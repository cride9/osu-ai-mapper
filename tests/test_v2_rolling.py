import hashlib
import json
import sqlite3
from dataclasses import asdict

import numpy as np
import pytest
import torch

from test_v2_data import database,make_assets
from osumapper import mapio
from osumapper.data import load_json,atomic_json,digest
from osumapper.v2.data import prepare,records,snapshot,read_cached_map
from osumapper.v2.dataset import AudioDataset,MapDataset
from osumapper.v2.features import build_cache
from osumapper.v2.model import AudioEncoder,ModelConfig
from osumapper.v2.runtime import save_checkpoint
from osumapper.v2.rolling import RollingCache
from test_v2_training import fake_features,prepared as base_prepared


@pytest.fixture
def prepared_dataset(tmp_path):
    return base_prepared.__wrapped__(tmp_path)


def test_eviction_keeps_values_reproducible_and_handles_oversized_entries(tmp_path):
    cache=RollingCache(tmp_path/'cache.sqlite',max_gib=0.00001,min_free_gib=0)
    calls=[]
    def build(seed):
        calls.append(seed)
        return {'values':np.random.default_rng(seed).integers(0,256,7000,dtype=np.uint8)}
    a=cache.get_or_create('a',lambda:build(1))
    cache.get_or_create('b',lambda:build(2))
    again=cache.get_or_create('a',lambda:build(1))
    assert calls==[1,2,1]
    np.testing.assert_array_equal(a['values'],again['values'])
    large=cache.get_or_create('large',lambda:{'values':np.zeros(200000,np.float64)+np.arange(200000)})
    assert len(large['values'])==200000
    with cache.connect() as db:
        assert db.execute('SELECT sum(size) FROM entries').fetchone()[0]<=cache.capacity


def test_parallel_cache_readers_keep_entries_consistent(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path=tmp_path/'shared.sqlite'
    RollingCache(path,.001,min_free_gib=0)
    def read(_):
        return RollingCache(path,.001,min_free_gib=0).get_or_create('same',lambda:{'v':np.arange(1000)})['v']
    with ThreadPoolExecutor(max_workers=4) as workers:
        results=list(workers.map(read,range(12)))
    for result in results: np.testing.assert_array_equal(result,np.arange(1000))


def test_full_import_and_mapper_use_all_maps_under_tiny_cache_budget(tmp_path):
    torch.set_num_threads(1)
    root,op,ap,bm=make_assets(tmp_path)
    source=tmp_path/'source.sqlite'; db=database(source)
    for bid in range(1,4):
        bm.metadata['Version']=str(bid)
        path=op.with_name(f'{bid}.osu'); mapio.write(bm,path)
        db.execute('INSERT INTO beatmaps VALUES(?,?,?)',(bid,55,hashlib.md5(path.read_bytes()).hexdigest()))
        db.execute('INSERT INTO dataset VALUES(?,?,?, ?,1,1,?,?,?)',(bid,55,str(path),str(ap),'A','B','[]'))
    db.commit();db.close()
    dest=tmp_path/'full'
    report=prepare(source,root,dest,max_metadata_gib=1e-6,progress=lambda _:None)
    assert report['usable_maps']==3 and report['examined_rows']==3
    assert not report['metadata_budget_limited'] and report['unexamined_rows']==0
    frozen=snapshot(dest,tmp_path/'run'); rows=list(records(frozen)); split=rows[0]['split']
    assert len(read_cached_map(dest,rows[0]).objects)==5
    ad=AudioDataset(frozen,split)
    assert ad[0]['mel'].shape[0]==128
    config=ModelConfig.preset('tiny'); model=AudioEncoder(config)
    checkpoint=tmp_path/'audio.pt'
    save_checkpoint(checkpoint,dict(format='osu-v2',stage='audio',model_config=asdict(config),model=model.state_dict(),dataset_hash=frozen['hash']))
    features=build_cache(dest,checkpoint,max_gib=1e-6,progress=lambda _:None)
    assert features['recordings']==1 and 'selected_audio' not in features
    ds=MapDataset(frozen,features,split)
    assert len(ds)==3
    selected=[ds.choose(i,1)[0][0] for i in range(3)]
    assert len(set(selected))==3
    assert ds.choose(3,1)==MapDataset(frozen,features,split).choose(3,1)
    before=torch.get_rng_state().clone()
    first=ds[0]
    assert torch.equal(before,torch.get_rng_state())
    assert first['local_audio'].shape[1]==32
    ds.feature_data.cache_clear(); ds.encoded.cache_clear(); ds.arrays.cache_clear()
    second=ds[0]
    torch.testing.assert_close(first['local_audio'],second['local_audio'],rtol=0,atol=0)
    # A source mutation after eviction cannot silently alter the frozen dataset.
    changed=rows[0]['map_path']
    with open(changed,'a',encoding='utf-8') as file: file.write('\n// changed\n')
    ds.arrays.cache_clear()
    with pytest.raises(ValueError,match='source map changed'):
        read_cached_map(dest,rows[0])


def test_full_mapper_resume_preserves_weights_and_sample_order(prepared_dataset,tmp_path,monkeypatch):
    from pathlib import Path
    from osumapper.v2 import rolling
    from osumapper.v2.runtime import load_checkpoint
    from osumapper.v2.training import train,TrainConfig
    prepared=prepared_dataset
    m=load_json(prepared/'manifest.json')
    db=sqlite3.connect(m['index'])
    for mid,payload in db.execute('SELECT id,payload FROM records').fetchall():
        r=json.loads(payload)
        r.update(rolling_cache_gib=1e-6,map_hash=digest(r['map_path']),
                 window_tokens=np.load(prepared/'maps'/mid/'windows.npy')[:,2].tolist())
        db.execute('UPDATE records SET payload=? WHERE id=?',(json.dumps(r),mid))
    db.commit(); db.close()
    m.update(storage_mode='rolling',rolling_cache_gib=1e-6,index_hash=digest(m['index']))
    atomic_json(prepared/'manifest.json',m)
    features=fake_features(prepared,tmp_path)
    features.update(storage_mode='rolling',budget_gib=1e-6,min_free_gib=1)
    atomic_json(prepared/'features.json',features)
    def feature_data(frozen,manifest,row):
        folder=Path(manifest['root'])/row['audio_hash']
        return {'encoded':np.load(folder/'encoded.npy'),'timing_json':rolling.json_array({'points':[]})}
    monkeypatch.setattr(rolling,'feature_bundle',feature_data)
    cfg=TrainConfig(stage='mapper',model='tiny',steps=2,hours=1,effective_batch=1,device='cpu',validate_every=1,sample_every=0,validation_maps=1)
    whole=tmp_path/'whole'; resumed=tmp_path/'resumed'
    train(prepared,whole,cfg,progress=lambda _:None)
    train(prepared,resumed,cfg,progress=lambda _:None,cancelled=lambda:load_json(resumed/'status.json',{}).get('step',0)>=1)
    train(prepared,resumed,cfg,resume=resumed/'last.pt',progress=lambda _:None)
    a=load_checkpoint(whole/'last.pt'); b=load_checkpoint(resumed/'last.pt')
    for k in a['model']: torch.testing.assert_close(a['model'][k],b['model'][k],rtol=1e-6,atol=1e-7)
    assert a['sampler']==b['sampler']
