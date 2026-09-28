import json
import sqlite3
from dataclasses import asdict

import numpy as np
import pytest
import soundfile as sf
import torch

from osumapper import audio,mapio
from osumapper.data import atomic_json,digest,load_json
from osumapper.v2 import data
from osumapper.v2.dataset import MapDataset
from osumapper.v2.features import build_cache
from osumapper.v2.model import AudioEncoder,ModelConfig
from osumapper.v2.runtime import save_checkpoint,load_checkpoint
from osumapper.v2.tokenizer import VERSION
from osumapper.v2.training import train,TrainConfig


def test_amp_overflow_replays_same_update_without_changing_weights():
    from osumapper.v2.training import backward_with_recovery
    model=torch.nn.Linear(2,1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scaler=torch.amp.GradScaler('cpu',init_scale=16.)
    before={k:v.clone() for k,v in model.state_dict().items()}
    draws=[]
    def backward():
        x=torch.rand(3,2); draws.append(x.clone())
        loss=model(x).square().mean()
        scaler.scale(loss).backward()
        if len(draws)==1: model.weight.grad.fill_(float('inf'))
        return (loss.detach().reshape(1),)
    result,retries=backward_with_recovery(model,optimizer,scaler,backward,lambda _:None)
    assert retries==1 and scaler.get_scale()==8
    torch.testing.assert_close(draws[0],draws[1],rtol=0,atol=0)
    for k,v in model.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    assert not optimizer.state and torch.isfinite(result[0]).all()
    scaler.step(optimizer); scaler.update()
    assert all(s['step']==1 for s in optimizer.state.values())
    assert not torch.equal(model.weight,before['weight'])


@pytest.mark.parametrize('bad_loss,enabled',[(True,True),(False,True),(False,False)])
def test_nonfinite_failure_is_bounded_and_does_not_update_weights(bad_loss,enabled):
    from osumapper.v2.training import backward_with_recovery
    model=torch.nn.Linear(1,1)
    before=model.weight.detach().clone()
    optimizer=torch.optim.AdamW(model.parameters())
    scaler=torch.amp.GradScaler('cpu',enabled=enabled)
    calls=[]
    def backward():
        calls.append(1)
        loss=model(torch.ones(1,1)).sum()
        if bad_loss: loss=loss*float('nan')
        scaler.scale(loss).backward()
        model.weight.grad.fill_(float('inf'))
        return (loss.detach().reshape(1),)
    with pytest.raises(FloatingPointError,match='forward loss' if bad_loss else 'gradient'):
        backward_with_recovery(model,optimizer,scaler,backward,lambda _:None,max_retries=2)
    assert len(calls)==(3 if enabled and not bad_loss else 1)
    assert not optimizer.state and model.weight.grad is None
    torch.testing.assert_close(model.weight,before,rtol=0,atol=0)


@pytest.fixture
def prepared(tmp_path):
    torch.set_num_threads(1)
    root=tmp_path/'data'; root.mkdir()
    db=sqlite3.connect(root/'index.sqlite')
    db.execute('CREATE TABLE records(id TEXT PRIMARY KEY,split TEXT,group_id TEXT,stars REAL,windows INTEGER,payload TEXT)')
    for i,split in enumerate(('train','validation','test')):
        mid='map-'+split; ah='audio-'+split
        bm=mapio.Beatmap(timing=[mapio.TimingPoint(0,500)],objects=[mapio.HitObject(120+j*50,180,1000+j*500) for j in range(5)])
        ap=root/(ah+'.wav'); sf.write(ap,np.sin(np.arange(audio.SR*4)*(i+1)*.05)*.05,audio.SR)
        op=root/(mid+'.osu'); bm.general['AudioFilename']=ap.name; mapio.write(bm,op)
        windows=data._cache_events(bm,root/'maps'/mid,4000)
        mel=np.random.default_rng(i).normal(-5,1,(128,401)).astype(np.float16)
        data.save_array(root/'audio'/ah/'mel.npy',mel)
        data.save_array(root/'audio'/ah/'labels'/'beats.npy',audio.beat_targets(bm,401).astype(np.float16))
        data.save_array(root/'audio'/ah/'labels'/'mask.npy',np.ones((401,2),np.uint8))
        row=dict(id=mid,split=split,group='group-'+split,stars=2,windows=windows,sample_weight=1,audio_hash=ah,timing_labels='labels',duration_ms=4000,
                 settings=dict(AR=5,OD=5,CS=4,HP=5),styles=dict(aim=None,streams=None,rhythm=None),timing=[vars(p) for p in bm.timing],
                 map_path=str(op),audio_path=str(ap),title=split,artist='test',song_key=split)
        db.execute('INSERT INTO records VALUES(?,?,?,?,?,?)',(mid,split,row['group'],2,windows,json.dumps(row)))
    db.commit(); db.close()
    manifest=dict(version=data.VERSION,root=str(root),index=str(root/'index.sqlite'),index_hash=digest(root/'index.sqlite'),hash='test-new-only',audio_version=audio.VERSION,tokenizer_version=VERSION)
    atomic_json(root/'manifest.json',manifest)
    return root


def test_audio_checkpoint_and_feature_pipeline(prepared,tmp_path):
    cfg=TrainConfig(stage='audio',model='tiny',steps=1,hours=1,batch_size=1,effective_batch=1,device='cpu',validate_every=1,sample_every=0,validation_maps=1)
    result=train(prepared,tmp_path/'audio-run',cfg,progress=lambda _:None)
    state=load_checkpoint(result['checkpoint'])
    assert result['step']==1 and state['sampler']['next_update']==1
    assert np.isfinite(state['best_validation']) and state['optimizer']['state']
    features=build_cache(prepared,tmp_path/'audio-run'/'best.pt',device='cpu',progress=lambda _:None)
    assert features['recordings']==3
    frozen=data.snapshot(prepared,tmp_path/'mapper-snapshot'); ds=MapDataset(frozen,features)
    sample=ds[0]
    assert sample['local_audio'].shape[1]==32
    assert sample['history'].numel()>=1
    assert len(ds.choose(3,32))==32
    assert ds.choose(3,32)==ds.choose(3,32)


def test_audio_spike_log_names_the_actual_recording_and_window(prepared,tmp_path,monkeypatch):
    from osumapper.v2 import training
    def high_loss(model,b,return_samples=False):
        anchor=next(model.parameters()).sum()*0
        result=(anchor+1.2,anchor.detach()+.9,anchor.detach()+3.)
        if return_samples:
            return *result,torch.tensor([[1.2,.9,3.]]*len(b['mel']))
        return result
    monkeypatch.setattr(training,'audio_loss',high_loss)
    cfg=TrainConfig(stage='audio',model='tiny',steps=1,hours=1,batch_size=2,effective_batch=4,
                    device='cpu',validate_every=1,sample_every=0,validation_maps=1)
    run=tmp_path/'spike-run'
    train(prepared,run,cfg,progress=lambda _:None)
    spikes=[json.loads(line) for line in (run/'loss-spikes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(spikes)==4
    assert all(s['step']==1 and s['maps']==['test - train'] for s in spikes)
    assert all(s['total_loss']==pytest.approx(1.2) and s['beat_loss']==pytest.approx(.9) and s['auxiliary_loss']==pytest.approx(3) for s in spikes)
    with (run/'hard_samples.csv').open(encoding='utf-8',newline='') as file:
        import csv
        rows=list(csv.DictReader(file))
    assert len(rows)==4
    assert all(r['title']=='train' and r['artist']=='test' and float(r['chunk_end_sec'])>=float(r['chunk_start_sec']) for r in rows)


def test_per_sample_audio_diagnostics_leave_optimized_loss_and_gradient_unchanged():
    from osumapper.v2.training import audio_loss
    torch.manual_seed(18)
    model=AudioEncoder(ModelConfig.preset('tiny')).eval()
    batch=dict(mel=torch.randn(2,128,32),clean=torch.randn(2,128,32),
               masked=torch.ones(2,128,32,dtype=torch.bool),beats=torch.rand(2,32,2),
               beat_mask=torch.ones(2,32,2))
    baseline=audio_loss(model,batch)[0]
    baseline.backward()
    before=[p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    observed,_,_,individual=audio_loss(model,batch,return_samples=True)
    observed.backward()
    torch.testing.assert_close(observed,baseline)
    assert individual.shape==(2,3)
    for old,p in zip(before,model.parameters()):
        if old is not None: torch.testing.assert_close(p.grad,old)


def fake_features(prepared,tmp_path):
    cfg=ModelConfig.preset('tiny'); model=AudioEncoder(cfg)
    checkpoint=tmp_path/'audio.pt'
    save_checkpoint(checkpoint,dict(format='osu-v2',stage='audio',model_config=asdict(cfg),model=model.state_dict(),dataset_hash='test-new-only'))
    feature_root=prepared/'features'/'fixed'
    for split in ('train','validation','test'):
        data.save_array(feature_root/('audio-'+split)/'encoded.npy',np.random.default_rng(1).normal(size=(101,32)).astype(np.float16))
        data.save_array(feature_root/('audio-'+split)/'beats.npy',np.zeros((401,2),np.float16))
        atomic_json(feature_root/('audio-'+split)/'timing.json',{'points':[],'report':{'uncertain':True}})
    features=dict(root=str(feature_root),dataset_hash='test-new-only',encoder_hash=digest(checkpoint),encoder_checkpoint=str(checkpoint),model_config=asdict(cfg))
    atomic_json(prepared/'features.json',features)
    return features


def test_selected_song_samples_cache_audio_and_keep_training_rng(prepared,tmp_path,monkeypatch):
    from osumapper.v2 import evaluation,features,generation
    from osumapper.v2.model import Mapper
    from osumapper.v2.tokenizer import Tokenizer
    manifest=fake_features(prepared,tmp_path)
    frozen=data.snapshot(prepared,tmp_path/'snapshot')
    dataset=MapDataset(frozen,manifest)
    model=Mapper(len(Tokenizer()),ModelConfig.preset('tiny')).train()
    run=tmp_path/'run'; run.mkdir()
    selection={'audio':str(prepared/'audio-train.wav'),'stars':5,'seed':2026}
    atomic_json(run/'training-sample.json',selection)
    encodes=[]; sections=[]; exports=[]
    monkeypatch.setattr(evaluation.audio,'decode',lambda _:np.zeros(audio.SR*33,np.float32))
    monkeypatch.setattr(evaluation.audio,'spectrogram',lambda _:np.zeros((128,10),np.float32))
    def encode(encoder,mel,dest,device,**kwargs):
        encodes.append(device.type)
        data.save_array(dest/'encoded.npy',np.zeros((10,32),np.float16))
        data.save_array(dest/'beats.npy',np.zeros((10,2),np.float16))
    monkeypatch.setattr(features,'encode_recording',encode)
    monkeypatch.setattr(evaluation,'estimate',lambda _:([mapio.TimingPoint(0,500)],{'uncertain':False}))
    monkeypatch.setattr(generation,'_style',lambda *args:np.zeros(64,np.float32))
    def section(model,encoded,bm,start,end,*args,**kwargs):
        assert not model.training
        torch.rand(3)
        sections.append((start,end))
        bm.objects.append(mapio.HitObject(256,192,int(start+100)))
        return end
    monkeypatch.setattr(evaluation,'section',section)
    def export(bm,source,dest,metadata):
        exports.append((len(bm.objects),metadata))
        return str(dest/'sample.osz')
    monkeypatch.setattr(evaluation,'export',export)
    before=torch.get_rng_state().clone()
    for step in (1000,2000):
        evaluation.training_sample(model,dataset,run,step,torch.device('cpu'),torch.float16,lambda _:None)
    assert encodes==['cpu']
    assert sections==[(0,16000),(16000,32000),(32000,33000)]*2
    assert len(exports)==2 and all(n==3 and not m['reference_timing'] and not m['partial_song'] for n,m in exports)
    assert model.training
    assert torch.equal(before,torch.get_rng_state())


def test_mapper_resume_matches_uninterrupted_and_guards_dataset(prepared,tmp_path):
    fake_features(prepared,tmp_path)
    cfg=TrainConfig(stage='mapper',model='tiny',steps=2,hours=1,effective_batch=1,device='cpu',validate_every=1,sample_every=0,validation_maps=1)
    uninterrupted=tmp_path/'full'
    train(prepared,uninterrupted,cfg,progress=lambda _:None)
    interrupted=tmp_path/'split'
    result=train(prepared,interrupted,cfg,progress=lambda _:None,cancelled=lambda:load_json(interrupted/'status.json',{}).get('step',0)>=1)
    assert result['status']=='cancelled' and result['step']==1
    train(prepared,interrupted,cfg,resume=interrupted/'last.pt',progress=lambda _:None)
    a,b=load_checkpoint(uninterrupted/'last.pt'),load_checkpoint(interrupted/'last.pt')
    assert a['step']==b['step']==2
    for key in a['model']: torch.testing.assert_close(a['model'][key],b['model'][key],rtol=1e-6,atol=1e-7)
    manifest=load_json(interrupted/'dataset.json'); manifest['snapshot_hash']='changed'; atomic_json(interrupted/'dataset.json',manifest)
    with pytest.raises(ValueError,match='mismatch'):
        train(prepared,interrupted,cfg,resume=interrupted/'last.pt',progress=lambda _:None)
