import copy

import pytest
import torch
from torch.nn import functional as F

from osumapper.v2.benchmark import synthetic
from osumapper.v2.model import AudioEncoder,Mapper,ModelConfig,token_loss
from osumapper.v2.tokenizer import Tokenizer


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def test_audio_encoder_finite_backward():
    model=AudioEncoder(ModelConfig.preset('tiny'))
    mel=torch.randn(2,128,128)
    features,beats,recon=model(mel)
    assert features.shape==(2,32,32) and beats.shape==(2,128,2) and recon.shape==mel.shape
    loss=beats.square().mean()+recon.square().mean(); loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_chunked_supervised_projection_loss_and_gradients():
    torch.manual_seed(7)
    model=Mapper(len(Tokenizer()),ModelConfig.preset('tiny'))
    h=torch.randn(2,12,32,requires_grad=True); labels=torch.randint(1,len(Tokenizer()),(2,12)); labels[:,:4]=-100
    loss,count=token_loss(model,h,labels,3)
    (loss/count).backward(); grad=h.grad.clone(); wgrad=model.embedding.weight.grad.clone()
    h.grad=None; model.zero_grad(set_to_none=True)
    expected=F.cross_entropy(model.project(h).reshape(-1,len(Tokenizer())),labels.flatten(),ignore_index=-100)
    expected.backward()
    torch.testing.assert_close(loss/count,expected)
    torch.testing.assert_close(h.grad,grad)
    torch.testing.assert_close(model.embedding.weight.grad,wgrad,atol=1e-7,rtol=1e-5)


def test_gqa_incremental_preallocated_cache_matches_teacher_forcing():
    torch.manual_seed(9)
    cfg=ModelConfig.preset('tiny'); model=Mapper(len(Tokenizer()),cfg).eval()
    b=synthetic(cfg,1,torch.device('cpu'),small=True)
    with torch.no_grad():
        memory,mask,_=model.memory(b)
        expected=model.project(model.hidden(b['tokens'],memory,mask))
        cache=None; got=[]; pointers=None
        for i in range(b['tokens'].shape[1]):
            logits,cache=model.step(b['tokens'][:,i:i+1],memory,mask,cache,i); got.append(logits)
            if pointers is None: pointers=[c[0]['k'].data_ptr() for c in cache]
            assert pointers==[c[0]['k'].data_ptr() for c in cache]
            assert cache[0][0]['k'].shape[1]==cfg.kv_heads
        torch.testing.assert_close(torch.stack(got,1),expected,atol=2e-6,rtol=1e-5)


def test_masked_padding_and_distant_context():
    torch.manual_seed(13)
    c=ModelConfig.preset('tiny'); model=Mapper(len(Tokenizer()),c).eval()
    b=synthetic(c,1,torch.device('cpu'),small=True)
    with torch.no_grad():
        out,_=model(b)
        padded={k:v.clone() for k,v in b.items()}
        padded['tokens']=F.pad(b['tokens'],(0,16)); padded['token_valid']=F.pad(b['token_valid'],(0,16),value=False)
        padded['history']=F.pad(b['history'],(0,37)); padded['history_valid']=F.pad(b['history_valid'],(0,37),value=False)
        got,_=model(padded)
        torch.testing.assert_close(out,got[:,:out.shape[1]],atol=2e-6,rtol=1e-5)
        b['global_audio']=b['global_audio']*0+3
        changed,_=model(b)
        assert not torch.allclose(out,changed,atol=1e-4)


def test_checkpointed_forward_backward():
    c=ModelConfig.preset('tiny'); c.checkpointing=True
    model=Mapper(len(Tokenizer()),c).train(); b=synthetic(c,1,torch.device('cpu'),small=True)
    h,p=model(b); loss,n=token_loss(model,h,b['labels']); (loss/n+p.square().mean()).backward()
    assert torch.isfinite(loss) and model.decoder[0].attn.kv.weight.grad is not None
