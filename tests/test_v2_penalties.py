import torch

from osumapper.mapio import HitObject
from osumapper.v2.penalties import out_of_bounds_loss
from osumapper.v2.quality import repetition_report
from osumapper.v2.tokenizer import Grammar,Tokenizer


def test_circle_grammar_respects_full_circle_radius():
    tok=Tokenizer(); grammar=Grammar(tok,0,0,16000,16000,cs=5)
    grammar.consume(tok.ids['CIRCLE']); grammar.consume(tok.t('T',100)); grammar.consume(tok.t('F',0))
    allowed=grammar.allowed()
    assert tok.t('XY',1024) not in allowed
    assert tok.t('XY',1024+16) in allowed
    assert tok.t('XY',1024+256) not in allowed


def test_training_probability_outside_playfield_is_penalized():
    tok=Tokenizer(); start=tok.ranges['XY'][0]
    seq=tok.condition(4,{},dict(CS=5))+[tok.ids['GEN'],tok.ids['CIRCLE'],tok.t('T',100),tok.t('F',0),tok.t('XY',1152),tok.t('XY',1120),tok.t('COMBO',0),tok.t('HS',0),tok.ids['END']]
    labels=torch.tensor(seq[1:]+[tok.ids['EOS']])[None]
    tokens=torch.tensor(seq)[None]
    class Projection:
        def __init__(self,edge): self.edge=edge
        def project(self,h):
            logits=torch.full((len(h),len(tok)),-20.,device=h.device)
            logits[:,start+1024+self.edge]=20.
            return logits+h[:,0,None]*0
    hidden=torch.zeros((1,len(seq),2),requires_grad=True)
    edge=out_of_bounds_loss(Projection(0),hidden,{'tokens':tokens,'labels':labels},tok)
    center=out_of_bounds_loss(Projection(128),hidden,{'tokens':tokens,'labels':labels},tok)
    assert edge>.9 and center<.01
    edge.backward()
    assert torch.isfinite(hidden.grad).all()


def test_long_bounce_and_stack_reported_but_short_pattern_allowed():
    bounce=[HitObject(100 if i%2 else 350,180,i*500) for i in range(50)]
    stack=[HitObject(200,150,i*500) for i in range(50)]
    assert repetition_report(bounce)['degenerate']
    assert repetition_report(stack)['degenerate']
    assert not repetition_report(bounce[:6])['degenerate']
