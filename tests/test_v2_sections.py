import numpy as np
import torch

from osumapper import mapio
from osumapper.v2.features import STRIDE_MS
from osumapper.v2.generation import GenerationConfig,section
from osumapper.v2.geometry import validate_map
from osumapper.v2.tokenizer import Tokenizer


class ScriptedModel:
    def __init__(self,tokens,vocabulary):
        self.tokens=tokens; self.vocabulary=vocabulary; self.calls=0
        self.config=type('Configuration',(),{'max_tokens':2048})()

    def memory(self,batch):
        return torch.zeros(1,1,8),torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1,64)

    def step(self,tokens,memory,memory_valid,caches=None,offset=0):
        next_token=self.tokens[self.calls]
        self.calls+=1
        logits=torch.full((1,self.vocabulary),-100.)
        logits[0,next_token]=100.
        return logits,caches or []


def test_slider_crosses_section_boundary_and_each_object_emitted_once():
    tok=Tokenizer(); device=torch.device('cpu')
    settings={'AR':8.,'OD':8.,'CS':5.,'HP':5.}
    source=mapio.Beatmap(timing=[mapio.TimingPoint(0,500)],difficulty={'CircleSize':'5','SliderMultiplier':'1.4'})
    slider=mapio.HitObject(100,100,15900,kind='slider',points=[(380,100)],length=280,repeats=1)
    circle=mapio.HitObject(240,200,17000)
    generated=mapio.Beatmap(timing=[mapio.TimingPoint(0,500)],difficulty=dict(source.difficulty))
    cfg=GenerationConfig(stars=4.,temperature=0)
    encoded=np.zeros((int(np.ceil(18000/STRIDE_MS)),32),np.float32)
    first=tok.object(source,slider,-32000)+[tok.ids['EOS']]
    model=ScriptedModel(first,len(tok))
    cursor=section(model,encoded,generated,0,16000,18000,cfg,settings,np.zeros(64,np.float32),device,torch.Generator().manual_seed(1),torch.float16)
    assert cursor==16000 and len(generated.objects)==1
    assert generated.objects[0].time+generated.duration(generated.objects[0])>cursor
    second=tok.object(source,circle,-16000)+[tok.ids['EOS']]
    model=ScriptedModel(second,len(tok))
    cursor=section(model,encoded,generated,cursor,18000,18000,cfg,settings,np.zeros(64,np.float32),device,torch.Generator().manual_seed(1),torch.float16)
    assert cursor==18000
    assert [o.time for o in generated.objects]==[15900,17000]
    validate_map(generated,18000)
