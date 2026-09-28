import copy

import numpy as np
import pytest

from osumapper.mapio import Beatmap,HitObject,TimingPoint,dumps,parse
from osumapper.v2.geometry import slider_path,validate_object
from osumapper.v2.tokenizer import Tokenizer,Grammar
from osumapper.v2.style import describe,reference_code


def test_long_window_all_curves_edges_and_timing_roundtrip():
    tok=Tokenizer(); bm=Beatmap(timing=[TimingPoint(0,500),TimingPoint(20000,-50,uninherited=False),TimingPoint(40000,400)])
    for i,curve in enumerate('BCLP'):
        bm.objects.append(HitObject(100,160,22000+i*5000,'slider',new_combo=True,combo_skip=2,hitsound=4,curve=curve,points=[(150,120),(150,120),(210,160)] if curve=='B' else [(150,120),(210,160)],repeats=2,length=140,edge_sounds=[2,4,8]))
    bm.objects.extend([HitObject(256,192,48000,'spinner',end_time=50000),HitObject(200,200,63001)])
    bm.breaks=[(51000,52000)]
    events=tok.records(bm,0,0,64000); sequence=sum([r for _,r in events],[])+[tok.ids['EOS']]
    grammar=Grammar(tok,0,0,64000,65000)
    for token in sequence:
        assert token in grammar.allowed(2048), (grammar.state,tok.names[token])
        grammar.consume(token)
    decoded=Beatmap(timing=copy.deepcopy(bm.timing)); tok.decode(sequence,decoded,0)
    assert [o.time for o in bm.objects]==[o.time for o in decoded.objects]
    assert decoded.objects[0].edge_sounds==[2,4,8]
    assert decoded.objects[0].points[0]==decoded.objects[0].points[1]
    for a,b in zip(bm.objects,decoded.objects): assert abs(bm.duration(a)-decoded.duration(b))<=.51
    assert parse(dumps(decoded)).objects[-1].time==63001
    with pytest.raises(ValueError): tok.time(64001,0)


def test_slider_expected_length_extension_and_trimming():
    s=HitObject(100,100,0,'slider',curve='L',points=[(200,100)],length=50)
    np.testing.assert_allclose(slider_path(s)[-1],[150,100])
    s.length=200; np.testing.assert_allclose(slider_path(s)[-1],[300,100]); validate_object(s,4)
    s.length=1000
    with pytest.raises(ValueError,match='Out-of-bounds'): validate_object(s,4)


def test_repeated_anchor_and_curved_extents():
    s=HitObject(100,100,0,'slider',points=[(200,100),(200,100),(200,200)],length=200)
    p=slider_path(s)
    assert np.any(np.all(np.isclose(p,[200,100]),axis=1))
    np.testing.assert_allclose(p[-1],[200,200],atol=.1)
    circle=HitObject(10,100,0)
    with pytest.raises(ValueError): validate_object(circle,4)
    # A control point can be outside while the played, trimmed curve is inside.
    s=HitObject(256,192,0,'slider',points=[(700,192),(256,220)],length=50)
    validate_object(s,4)


def test_style_shape_and_reference_excludes_target():
    bm=Beatmap(timing=[TimingPoint(0,500)],objects=[HitObject(100+i*20,100,1000+i*250) for i in range(10)])
    vector=describe(bm)
    assert vector.shape==(64,) and np.isfinite(vector).all()
    blocks=np.ones((50,64),np.float32); starts=np.arange(50)*2000
    selected=reference_code(blocks,starts,40000,56000)
    blocks[(starts>=8000)&(starts<72000)]=999
    np.testing.assert_array_equal(reference_code(blocks,starts,40000,56000),selected)
