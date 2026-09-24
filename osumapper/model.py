from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    width: int = 384
    encoder_layers: int = 6
    decoder_layers: int = 8
    heads: int = 6
    feedforward: int = 1536
    dropout: float = 0.1
    max_tokens: int = 1536
    checkpointing: bool = True

    @classmethod
    def tiny(cls):
        return cls(96, 2, 2, 3, 384, 0.0, 1536, False)


def positions(length, width, device, offset=0):
    index = torch.arange(offset, offset + length, device=device, dtype=torch.float32)[:, None]
    scale = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10000) / width))
    pos = torch.zeros(length, width, device=device)
    pos[:, 0::2], pos[:, 1::2] = torch.sin(index * scale), torch.cos(index * scale)
    return pos


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.dim = c.heads, c.width // c.heads
        self.q, self.k, self.v, self.out = [nn.Linear(c.width, c.width) for _ in range(4)]
        self.dropout = c.dropout

    def split(self, x):
        return x.view(x.shape[0], x.shape[1], self.heads, self.dim).transpose(1, 2)

    def forward(self, x, source=None, causal=False, cache=None, static=False):
        q = self.split(self.q(x))
        source = x if source is None else source
        if cache is not None and static:
            k, v = cache
        else:
            k, v = self.split(self.k(source)), self.split(self.v(source))
            if cache is not None:
                k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=causal and cache is None)
        y = y.transpose(1, 2).contiguous().view(x.shape)
        return self.out(y), (k, v)


class EncoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn = Attention(c)
        self.n1, self.n2 = nn.LayerNorm(c.width), nn.LayerNorm(c.width)
        self.ff = nn.Sequential(nn.Linear(c.width, c.feedforward), nn.GELU(), nn.Dropout(c.dropout), nn.Linear(c.feedforward, c.width))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x))[0])
        return x + self.drop(self.ff(self.n2(x)))


class DecoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn, self.cross_attn = Attention(c), Attention(c)
        self.n1, self.n2, self.n3 = [nn.LayerNorm(c.width) for _ in range(3)]
        self.ff = nn.Sequential(nn.Linear(c.width, c.feedforward), nn.GELU(), nn.Dropout(c.dropout), nn.Linear(c.feedforward, c.width))
        self.drop = nn.Dropout(c.dropout)

    def step(self, x, memory, cache=None):
        a, sa = self.self_attn(self.n1(x), causal=True, cache=None if cache is None else cache[0])
        x = x + self.drop(a)
        a, ca = self.cross_attn(self.n2(x), memory, cache=None if cache is None else cache[1], static=True)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n3(x)))
        return x, (sa, ca)

    def forward(self, x, memory):
        return self.step(x, memory)[0]


class Mapper(nn.Module):
    def __init__(self, vocab_size, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.frontend = nn.Sequential(nn.Conv1d(128, c.width, 5, padding=2), nn.GELU(), nn.Conv1d(c.width, c.width, 3, stride=2, padding=1), nn.GELU(), nn.Conv1d(c.width, c.width, 3, stride=2, padding=1), nn.GELU())
        self.encoder = nn.ModuleList([EncoderBlock(c) for _ in range(c.encoder_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(c) for _ in range(c.decoder_layers)])
        self.encoder_norm, self.decoder_norm = nn.LayerNorm(c.width), nn.LayerNorm(c.width)
        self.embedding = nn.Embedding(vocab_size, c.width, padding_idx=0)
        self.timing_projection = nn.Linear(3, c.width)
        self.beat_head = nn.Linear(c.width, 2)
        self.output = nn.Linear(c.width, vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def encode(self, mel, phase=None):
        x = self.frontend(mel).transpose(1, 2)
        x = x + positions(x.shape[1], x.shape[2], x.device).to(x.dtype)
        for block in self.encoder:
            x = checkpoint(block, x, use_reentrant=False) if self.training and self.config.checkpointing else block(x)
        x = self.encoder_norm(x)
        # Timing prediction cannot see the ground-truth timing conditioning.
        beat_logits = F.interpolate(self.beat_head(x).transpose(1, 2), size=mel.shape[-1], mode="linear", align_corners=False).transpose(1, 2)
        if phase is not None:
            pooled_phase = F.interpolate(phase.transpose(1, 2), size=x.shape[1], mode="linear", align_corners=False).transpose(1, 2)
            x = x + self.timing_projection(pooled_phase)
        return x, beat_logits

    def decode(self, tokens, memory):
        if tokens.shape[1] > self.config.max_tokens:
            raise ValueError("Decoder token budget exceeded")
        x = self.embedding(tokens)
        x = x + positions(x.shape[1], x.shape[2], x.device).to(x.dtype)
        for block in self.decoder:
            x = checkpoint(block, x, memory, use_reentrant=False) if self.training and self.config.checkpointing else block(x, memory)
        return self.output(self.decoder_norm(x))

    def forward(self, mel, tokens, phase=None):
        memory, beats = self.encode(mel, phase)
        return self.decode(tokens, memory), beats

    @torch.no_grad()
    def decode_step(self, tokens, memory, caches=None, offset=0):
        if offset + tokens.shape[1] > self.config.max_tokens:
            raise ValueError("Decoder token budget exceeded")
        x = self.embedding(tokens)
        x = x + positions(x.shape[1], x.shape[2], x.device, offset).to(x.dtype)
        new_caches = []
        for i, block in enumerate(self.decoder):
            x, cache = block.step(x, memory, None if caches is None else caches[i])
            new_caches.append(cache)
        return self.output(self.decoder_norm(x[:, -1:])), new_caches
