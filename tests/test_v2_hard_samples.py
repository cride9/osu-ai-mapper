import csv

import numpy as np

from osumapper.v2.hard_samples import timing_summary, write_spikes


def test_timing_summary_counts_only_real_bpm_changes_inside_chunk():
    points=((0,500,True),(1000,-100,False),(2000,500,True),
            (4000,375,True),(8000,600,True))
    row=timing_summary(points,3000,7000)
    assert row['bpm']==160
    assert row['min_bpm']==100 and row['max_bpm']==160
    assert row['timing_point_count']==5
    assert row['uninherited_timing_point_count']==4
    assert row['bpm_change_count']==2
    assert row['bpm_changes_in_chunk']
    assert not timing_summary(points,1000,3000)['bpm_changes_in_chunk']


def test_utf8_csv_one_row_per_hard_window_and_single_header(tmp_path):
    meta={'source_map':{'beatmap_id':123,'set_id':45,'artist':'宇多田ヒカル',
                        'title':'Éjszaka','version':'難しい','map_path':'C:/音楽/map.osu'},
          'audio_path':'C:/音楽/audio.mp3','audio_hash':'hash','maps':['Éjszaka [難しい]'],
          'start_ms':1000,'duration_ms':5000,'timing_points':((0,500,True),)}
    target=np.zeros((2,400,2),np.float32)
    target[:,50,0]=1; target[:,150,0]=1; target[:,50,1]=1
    mask=np.ones_like(target)
    lines=[]
    batch=([[1.4,1.39,.1],[1.8,1.79,.1]], [meta,meta],target,mask)
    write_spikes(tmp_path,7,[batch],lines.append,total_samples=4)
    write_spikes(tmp_path,8,[batch],lines.append,total_samples=4)
    with (tmp_path/'hard_samples.csv').open(encoding='utf-8',newline='') as file:
        rows=list(csv.DictReader(file))
    assert len(rows)==4
    assert rows[0]['artist']=='宇多田ヒカル' and rows[0]['version']=='難しい'
    assert rows[0]['beatmap_id']=='123' and rows[0]['beatmapset_id']=='45'
    assert rows[0]['chunk_start_sec']=='1.0' and 4.98<float(rows[0]['chunk_end_sec'])<=5.0
    assert rows[0]['beat_target_count']=='2' and rows[0]['strong_beat_target_count']=='1'
    assert 0.5<float(rows[0]['beat_target_density'])<0.51
    assert any('SPIKE SUMMARY step 7: 2/4' in line for line in lines)
    assert (tmp_path/'hard_samples.csv').read_bytes().count('timestamp,step,'.encode())==1
