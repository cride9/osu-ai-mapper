import numpy as np
import pytest

from osumapper import audio,mapio
from osumapper.v2.timing import estimate,TimingUncertain


def activations(duration,points):
    bm=mapio.Beatmap(timing=points)
    return audio.beat_targets(bm,round(duration/audio.FRAME_MS))


def test_no_hidden_default_bpm():
    with pytest.raises(TimingUncertain): estimate(np.zeros((1000,2)))
    points,report=estimate(np.zeros((1000,2)),strict=False)
    assert points==[] and report['uncertain']
    points,report=estimate(np.zeros((1,2)),bpm=173,offset=123)
    assert points[0].time==123 and points[0].beat_length==60000/173 and report['manual']


def test_pulse_offset_and_tempo():
    scores=activations(32000,[mapio.TimingPoint(123,500)])
    points,report=estimate(scores)
    assert report['confidence']>.55
    assert abs(60000/points[0].beat_length-120)<1
    phase=(points[0].time-123)%500
    assert min(phase,500-phase)<15


def test_piecewise_tempo_change():
    scores=activations(64000,[mapio.TimingPoint(100,500),mapio.TimingPoint(32100,400)])
    points,report=estimate(scores)
    assert len(points)>=2
    assert abs(60000/points[0].beat_length-120)<1
    assert abs(60000/points[-1].beat_length-150)<1
