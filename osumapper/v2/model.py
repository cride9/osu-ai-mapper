"""Native-PyTorch Conformer + hierarchical dense GQA mapper, random init only."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    name: str = "v2-s"
    width: int = 512
    layers: int = 8
    heads: int = 8
    kv_heads: int = 2
    ff: int = 1408
    audio_width: int = 512
    audio_layers: int = 6
    audio_heads: int = 8
    frontend_channels: tuple = (32,64,128,256)
    context_width: int = 256
    planner_layers: int = 4
    history_layers: int = 2
    max_tokens: int = 2048
    history_tokens: int = 1024
    audio_ms: int = 64000
    dropout: float = .1
    checkpointing: bool = True

    @classmethod
    def preset(cls, name):
        if name == "v2-s": return cls()
        if name == "v2-l": return cls(name=name,width=640,layers=12,heads=10,kv_heads=2,ff=1728)
        if name == "tiny": return cls(name=name,width=32,layers=1,heads=4,kv_heads=1,ff=96,audio_width=32,audio_layers=1,audio_heads=4,frontend_channels=(4,8,16,32),context_width=32,planner_layers=1,history_layers=1,dropout=0,checkpointing=False)
        raise ValueError(f"Unknown V2 model: {name}")


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(width)); self.eps=eps
    def forward(self,x):
        y=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)
        return (y*self.weight.float()).to(x.dtype)


def rope(x, offset=0):
    d=x.shape[-1]
    theta=torch.arange(offset,offset+x.shape[-2],device=x.device,dtype=torch.float32)[:,None] * torch.exp(-math.log(10000)*torch.arange(0,d,2,device=x.device,dtype=torch.float32)/d)[None]
    c,s=theta.cos().to(x.dtype),theta.sin().to(x.dtype)
    a,b=x[...,0::2],x[...,1::2]
    return torch.stack([a*c-b*s,a*s+b*c],-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self,width,heads,kv_heads=None,source_width=None,dropout=.1,rotary=True):
        super().__init__(); self.heads=heads; self.kv_heads=kv_heads or heads; self.dim=width//heads
        if width%heads or heads%self.kv_heads or self.dim%2: raise ValueError("Invalid attention dimensions")
        source_width=source_width or width
        self.q=nn.Linear(width,width,bias=False)
        self.kv=nn.Linear(source_width,2*self.kv_heads*self.dim,bias=False)
        self.out=nn.Linear(width,width,bias=False)
        self.qnorm,self.knorm=RMSNorm(self.dim),RMSNorm(self.dim)
        self.dropout,self.rotary=dropout,rotary

    def forward(self,x,source=None,valid=None,causal=False,cache=None,offset=0,max_length=None):
        b,t,_=x.shape
        q=self.qnorm(self.q(x).view(b,t,self.heads,self.dim).transpose(1,2))
        if self.rotary: q=rope(q,offset)
        static=source is not None
        if static and cache is not None and "k" in cache:
            k,v=cache["k"],cache["v"]
        else:
            src=x if source is None else source
            kv=self.kv(src).view(b,src.shape[1],2,self.kv_heads,self.dim).permute(2,0,3,1,4)
            k,v=self.knorm(kv[0]),kv[1]
            if self.rotary: k=rope(k,offset)
            if cache is not None:
                if self.training: raise ValueError("KV cache is inference-only")
                if static: cache.update(k=k,v=v)
                else:
                    if max_length is None or offset+t>max_length: raise ValueError("KV capacity exceeded")
                    if "k" not in cache:
                        cache["k"]=torch.empty((b,self.kv_heads,max_length,self.dim),device=k.device,dtype=k.dtype)
                        cache["v"]=torch.empty_like(cache["k"])
                    cache["k"][:,:,offset:offset+t].copy_(k); cache["v"][:,:,offset:offset+t].copy_(v)
                    k,v=cache["k"][:,:,:offset+t],cache["v"][:,:,:offset+t]
                    cache["length"]=offset+t
        mask=None
        if valid is not None: mask=valid[:,None,None,:].bool()
        if causal and (mask is not None or offset):
            cm=torch.arange(k.shape[-2],device=x.device)[None,:] <= torch.arange(offset,offset+t,device=x.device)[:,None]
            mask=cm[None,None] if mask is None else mask & cm[None,None]
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=causal and mask is None,
                                        dropout_p=self.dropout if self.training else 0,enable_gqa=self.heads!=self.kv_heads)
        return self.out(y.transpose(1,2).reshape(b,t,-1))


class SwiGLU(nn.Module):
    def __init__(self,width,hidden):
        super().__init__(); self.up=nn.Linear(width,hidden*2,bias=False); self.down=nn.Linear(hidden,width,bias=False)
    def forward(self,x):
        a,b=self.up(x).chunk(2,-1)
        return self.down(F.silu(a)*b)


class Block(nn.Module):
    def __init__(self,width,heads,ff,kv_heads=None,cross=False,dropout=.1):
        super().__init__(); self.attn=Attention(width,heads,kv_heads,dropout=dropout)
        self.norm1,self.norm2=RMSNorm(width),RMSNorm(width)
        self.ff=SwiGLU(width,ff); self.drop=nn.Dropout(dropout)
        self.cross=Attention(width,heads,kv_heads,dropout=dropout,rotary=False) if cross else None
        if cross: self.norm_cross=RMSNorm(width)
    def forward(self,x,valid=None,memory=None,memory_valid=None,cache=None,offset=0,max_length=None):
        x=x+self.drop(self.attn(self.norm1(x),valid=valid,causal=self.cross is not None,cache=None if cache is None else cache[0],offset=offset,max_length=max_length))
        if self.cross is not None: x=x+self.drop(self.cross(self.norm_cross(x),source=memory,valid=memory_valid,cache=None if cache is None else cache[1]))
        return x+self.drop(self.ff(self.norm2(x)))


class ConvBlock(nn.Module):
    def __init__(self,cin,cout,time_stride):
        super().__init__(); groups=math.gcd(cout,8)
        self.net=nn.Sequential(nn.Conv2d(cin,cout,3,stride=(2,time_stride),padding=1),nn.GroupNorm(groups,cout),nn.SiLU(),nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(groups,cout))
        self.skip=nn.Conv2d(cin,cout,1,stride=(2,time_stride))
    def forward(self,x): return F.silu(self.net(x)+self.skip(x))


class Conformer(nn.Module):
    def __init__(self,width,heads,dropout):
        super().__init__(); hidden=math.ceil((8*width/3)/64)*64
        self.n=nn.ModuleList([RMSNorm(width) for _ in range(5)])
        self.f1,self.f2=SwiGLU(width,hidden),SwiGLU(width,hidden)
        self.attn=Attention(width,heads,dropout=dropout)
        self.conv=nn.Sequential(nn.Conv1d(width,width*2,1),nn.GLU(dim=1),nn.Conv1d(width,width,31,padding=15,groups=width),nn.GroupNorm(1,width),nn.SiLU(),nn.Conv1d(width,width,1))
        self.drop=nn.Dropout(dropout)
    def forward(self,x):
        x=x+.5*self.drop(self.f1(self.n[0](x)))
        x=x+self.drop(self.attn(self.n[1](x)))
        x=x+self.drop(self.conv(self.n[2](x).transpose(1,2)).transpose(1,2))
        return self.n[4](x+.5*self.drop(self.f2(self.n[3](x))))


class AudioEncoder(nn.Module):
    def __init__(self,c):
        super().__init__(); self.config=c
        blocks=[]; cin=1
        for i,cout in enumerate(c.frontend_channels):
            blocks.append(ConvBlock(cin,cout,2 if i<2 else 1)); cin=cout
        self.frontend=nn.Sequential(*blocks)
        self.projection=nn.Linear(cin*8,c.audio_width)
        self.layers=nn.ModuleList([Conformer(c.audio_width,c.audio_heads,c.dropout) for _ in range(c.audio_layers)])
        self.fine=nn.Sequential(nn.Conv1d(128,32,5,padding=2),nn.SiLU(),nn.Conv1d(32,2,1))
        self.beat=nn.Linear(c.audio_width,2)
        self.reconstruct=nn.Linear(c.audio_width,128)

    def forward(self,mel):
        x=self.frontend(mel[:,None]).permute(0,3,1,2).flatten(2)
        x=self.projection(x)
        for layer in self.layers:
            x=checkpoint(layer,x,use_reentrant=False) if self.training and self.config.checkpointing else layer(x)
        beats=F.interpolate(self.beat(x).transpose(1,2),size=mel.shape[-1],mode="linear",align_corners=False)+self.fine(mel)
        recon=F.interpolate(self.reconstruct(x).transpose(1,2),size=mel.shape[-1],mode="linear",align_corners=False)
        return x,beats.transpose(1,2),recon


class Mapper(nn.Module):
    def __init__(self,vocab_size,config=None):
        super().__init__(); self.config=c=config or ModelConfig()
        self.embedding=nn.Embedding(vocab_size,c.width,padding_idx=0)
        self.history_embedding=nn.Embedding(vocab_size,c.context_width,padding_idx=0)
        self.audio_projection=nn.Linear(c.audio_width,c.width)
        self.phase_projection=nn.Linear(3,c.width)
        self.global_in=nn.Linear(c.audio_width,c.context_width)
        self.style_in=nn.Linear(64,c.context_width)
        self.past_in=nn.Linear(64,c.context_width)
        heads=4
        self.planner=nn.ModuleList([Block(c.context_width,heads,c.context_width*3,dropout=c.dropout) for _ in range(c.planner_layers)])
        self.history=nn.ModuleList([Block(c.context_width,heads,c.context_width*3,dropout=c.dropout) for _ in range(c.history_layers)])
        self.global_out=nn.Linear(c.context_width,c.width)
        self.history_out=nn.Linear(c.context_width,c.width)
        self.plan_head=nn.Linear(c.context_width,64)
        self.time_in=nn.Linear(4,c.width,bias=False)
        self.modality=nn.Embedding(5,c.width)
        self.state_in=nn.Linear(8,c.width)
        self.decoder=nn.ModuleList([Block(c.width,c.heads,c.ff,c.kv_heads,True,c.dropout) for _ in range(c.layers)])
        self.norm=RMSNorm(c.width)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,std=.02)
            if getattr(m,"bias",None) is not None: nn.init.zeros_(m.bias)
            if isinstance(m,nn.Embedding) and m.padding_idx is not None:
                with torch.no_grad(): m.weight[m.padding_idx].zero_()

    def _run(self,layer,x,valid):
        return checkpoint(layer,x,valid,use_reentrant=False) if self.training and self.config.checkpointing else layer(x,valid)

    def memory(self,b):
        g=self.global_in(b["global_audio"])+self.style_in(b["style"])[:,None]
        for layer in self.planner: g=self._run(layer,g,b["global_valid"])
        plan=self.plan_head(g)
        h=self.history_embedding(b["history"])
        for layer in self.history: h=self._run(layer,h,b["history_valid"])
        # Pool history into 128 contiguous summaries after contextual encoding.
        n=128; mask=b["history_valid"].to(h.dtype)
        lengths=mask.sum(1).clamp_min(1)
        index=(torch.arange(h.shape[1],device=h.device)[None]*n/lengths[:,None]).long().clamp_max(n-1)
        pooled=h.new_zeros((h.shape[0],n,h.shape[2])).scatter_add(1,index[:,:,None].expand_as(h),h*mask[:,:,None])
        denom=h.new_zeros((h.shape[0],n)).scatter_add(1,index,mask)[:,:,None]
        h=pooled/denom.clamp_min(1e-6); hv=denom[:,:,0]>0
        chunks=[self.audio_projection(b["local_audio"])+self.phase_projection(b["phase"]), self.global_out(g), self.history_out(h),
                self.global_out(self.past_in(b["past"])), self.state_in(b["state"])[:,None]]
        valids=[b["local_valid"],b["global_valid"],hv,b["past_valid"],torch.ones((g.shape[0],1),device=g.device,dtype=torch.bool)]
        times=[b["local_times"],b["global_times"],torch.zeros(h.shape[:2],device=h.device),b["past_times"],torch.zeros((g.shape[0],1),device=g.device)]
        for i,x in enumerate(chunks):
            t=times[i]/64000
            tf=torch.stack([t,torch.sin(t*2*math.pi),torch.cos(t*2*math.pi),torch.ones_like(t)],-1)
            chunks[i]=x+self.time_in(tf.to(x.dtype))+self.modality.weight[i].to(x.dtype)
        return torch.cat(chunks,1),torch.cat(valids,1),plan

    def hidden(self,tokens,memory,memory_valid,valid=None):
        if tokens.shape[1]>self.config.max_tokens: raise ValueError("V2 decoder budget exceeded")
        x=self.embedding(tokens)
        for layer in self.decoder:
            if self.training and self.config.checkpointing:
                x=checkpoint(layer,x,valid,memory,memory_valid,use_reentrant=False)
            else: x=layer(x,valid,memory,memory_valid)
        return self.norm(x)

    def project(self,hidden): return F.linear(hidden,self.embedding.weight)

    def forward(self,b):
        memory,mask,plan=self.memory(b)
        return self.hidden(b["tokens"],memory,mask,b.get("token_valid")),plan

    @torch.no_grad()
    def step(self,tokens,memory,memory_valid,caches=None,offset=0):
        caches=caches or [[{},{}] for _ in self.decoder]
        x=self.embedding(tokens)
        for layer,cache in zip(self.decoder,caches):
            x=layer(x,memory=memory,memory_valid=memory_valid,cache=cache,offset=offset,max_length=self.config.max_tokens)
        return self.project(self.norm(x[:,-1])),caches


def token_loss(model,hidden,labels,chunk_size=256):
    valid=labels!=-100
    selected=hidden[valid]; target=labels[valid]
    if not len(target): return hidden.sum()*0,0
    total=hidden.new_zeros((),dtype=torch.float32)
    def part(h,y): return F.cross_entropy(model.project(h).float(),y,reduction="sum")
    for start in range(0,len(target),chunk_size):
        h,y=selected[start:start+chunk_size],target[start:start+chunk_size]
        total=total+(checkpoint(part,h,y,use_reentrant=False) if torch.is_grad_enabled() and h.requires_grad else part(h,y))
    return total,len(target)


def parameter_counts(model):
    return {"total":sum(p.numel() for p in model.parameters()),"trainable":sum(p.numel() for p in model.parameters() if p.requires_grad)}
