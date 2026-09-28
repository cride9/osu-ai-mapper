"""Unseen-song evaluation, explicit timing modes and unfilled human quality gates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

from .. import audio,mapio
from ..data import atomic_json,difficulty,load_json,style_metrics,digest
from ..evaluation import match_events
from ..generation import export
from .data import records,read_cached_map
from .generation import GenerationConfig,generate,section
from .geometry import validate_map
from .runtime import load_checkpoint,random_state,restore_random
from .style import describe
from .timing import estimate
from .storage import local_file,parts
from .quality import repetition_report


def _panel(frozen,split,count,exclude_groups=(),eligible_audio=None):
    rows=sorted(records(frozen,split),key=lambda r:(r['stars'],r['id'])); unique={}
    for r in rows:
        if r['group'] not in exclude_groups and (eligible_audio is None or r['audio_hash'] in eligible_audio): unique.setdefault(r['group'],r)
    values=list(unique.values())
    return [values[i] for i in np.linspace(0,len(values)-1,min(count,len(values))).astype(int)] if values else []


def timing_scores(activations,reference,mask):
    result={}
    for channel,name in enumerate(('beat','downbeat')):
        pred,_=find_peaks(activations[:,channel],height=.5,distance=round(110/audio.FRAME_MS))
        truth,_=find_peaks(reference[:,channel],height=.5,distance=round(110/audio.FRAME_MS))
        pred=pred[mask[pred,channel]>0]; truth=truth[mask[truth,channel]>0]
        tp,npred,ntrue,errors=match_events(pred*audio.FRAME_MS,truth*audio.FRAME_MS,50)
        result[name]={'f1':2*tp/max(1,npred+ntrue),'precision':tp/max(1,npred),'recall':tp/max(1,ntrue),'median_error_ms':float(np.median(errors)) if errors else None,'matched':tp}
    return result


def mapping_metrics(bm):
    blocks=[describe(bm,a,a+2000) for a in range(0,max(1,bm.objects[-1].time+1),2000)]
    values=np.asarray(blocks)
    change=np.linalg.norm(np.diff(values,axis=0),axis=1) if len(values)>1 else np.zeros(1)
    boundary=np.array([(i+1)%8==0 for i in range(len(change))])
    return {'descriptor_variance':float(values.var(0).mean()),'boundary_style_change':float(change[boundary].mean()) if boundary.any() else None,
            'within_style_change':float(change[~boundary].mean()) if (~boundary).any() else None,'objects':len(bm.objects),**repetition_report(bm.objects)}


def evaluate(checkpoint_path,suite='full',split='test',count=32,output_dir=None,device='auto',v1_run=None,progress=print,cancelled=lambda:False):
    checkpoint_path=Path(checkpoint_path); saved=load_checkpoint(checkpoint_path)
    frozen=load_json(checkpoint_path.parent/'dataset.json'); features=load_json(checkpoint_path.parent/'features.json')
    dest=Path(output_dir or checkpoint_path.parent/('evaluation-'+split)); dest.mkdir(parents=True,exist_ok=True)
    if saved['dataset_hash']!=frozen['hash']: raise ValueError("Checkpoint/dataset mismatch")
    excluded=set()
    if v1_run:
        old=load_json(Path(v1_run)/'dataset.json',{})
        seen_audio={r['audio_hash'] for r in old.get('records',[]) if r['split']=='train'}
        seen_song={r.get('song_key') for r in old.get('records',[]) if r['split']=='train'}
        excluded={r['group'] for r in records(frozen) if r['audio_hash'] in seen_audio or r.get('song_key') in seen_song}
    eligible=set(features['selected_audio']) if features and 'selected_audio' in features else None
    panel=_panel(frozen,split,count,excluded,eligible)
    if not panel: raise ValueError("No unseen songs available for this evaluation")
    results=[]
    if suite in ('timing','full'):
        if not features: raise ValueError("Build encoder features first")
        for row in panel:
            if cancelled(): raise InterruptedError("Evaluation cancelled")
            folder=Path(features['root'])/row['audio_hash']; labels=Path(frozen['root'])/'audio'/row['audio_hash']/row['timing_labels']
            if features.get('storage_mode')=='rolling':
                from .rolling import feature_bundle,audio_bundle
                scores=feature_bundle(frozen,features,row)['beats']
                supervision=audio_bundle(frozen,row); target,mask=supervision['beats'],supervision['mask']
            else:
                scores=np.load(folder/'beats.npy',mmap_mode='r'); target=np.load(labels/'beats.npy',mmap_mode='r'); mask=np.load(labels/'mask.npy',mmap_mode='r')
            points,report=estimate(scores,strict=False)
            ref=[mapio.TimingPoint(**p) for p in row['timing'] if p['uninherited']]
            bpm_error=[]; offset_error=[]; octave_error=[]
            for p in points:
                r=next((q for q in reversed(ref) if q.time<=p.time),ref[0])
                ratio=r.beat_length/p.beat_length
                bpm_error.append(abs(60000/p.beat_length-60000/r.beat_length))
                octave_error.append(min(abs(ratio-1),abs(ratio-.5),abs(ratio-2)))
                phase=(p.time-r.time)/r.beat_length; offset_error.append(abs(phase-round(phase))*r.beat_length)
            results.append({'id':row['id'],'title':row['title'],'timing':timing_scores(scores,target,mask),'decoder':report,
                            'bpm_mae':float(np.mean(bpm_error)) if bpm_error else None,'phase_error_ms':float(np.mean(offset_error)) if offset_error else None,
                            'octave_tolerant_ratio_error':float(np.mean(octave_error)) if octave_error else None})
    maps=[]
    if suite in ('mapping','full'):
        if saved['stage']!='mapper': raise ValueError("Full map evaluation needs a mapper checkpoint")
        for row in panel:
            for timing_mode in ('automatic','reference'):
                if cancelled(): raise InterruptedError("Evaluation cancelled")
                source_map=row['map_path']
                if timing_mode=='reference' and parts(source_map):
                    source_map=str(dest/'reference'/f"{row['id']}.osu")
                    Path(source_map).parent.mkdir(parents=True,exist_ok=True)
                    if not Path(source_map).exists(): mapio.write(read_cached_map(frozen['root'],row),source_map)
                cfg=GenerationConfig(stars=row['stars'],seed=2026,candidates=1,device=device,timing_map=source_map if timing_mode=='reference' else None)
                folder=dest/'maps'/row['id'][:12]/timing_mode
                try:
                    audio_source=local_file(row['audio_path'],dest/'audio-source')
                    outputs=generate(checkpoint_path,audio_source,folder,cfg,progress,cancelled,cache_root=dest/'shared-audio-cache')
                    for archive in outputs:
                        meta=load_json(Path(archive).with_suffix('')/'generation.json')
                        osu=next(Path(archive).with_suffix('').glob('*.osu')); bm=mapio.read(osu)
                        maps.append({'id':row['id'],'timing_mode':timing_mode,'archive':archive,'requested_stars':row['stars'],'measured_stars':meta['measured_stars'],
                                     **mapping_metrics(bm),'playtest_timing':None,'playtest_flow':None,'playtest_fun':None,'reference_style':False})
                except ValueError as exc: maps.append({'id':row['id'],'timing_mode':timing_mode,'error':str(exc)})
    reference_maps=[]
    if suite=='full' and saved['stage']=='mapper':
        train_rows=sorted(records(frozen,'train'),key=lambda r:(r['stars'],r['id']))
        for row in panel[:4]:
            if cancelled(): raise InterruptedError("Evaluation cancelled")
            refs=[r for r in train_rows if abs(r['stars']-row['stars'])<1 and r['group']!=row['group']]
            if not refs: continue
            reference=refs[len(refs)//2]
            reference_path=dest/'reference'/f"style-{reference['id']}.osu"
            reference_path.parent.mkdir(parents=True,exist_ok=True)
            if not reference_path.exists(): mapio.write(read_cached_map(frozen['root'],reference),reference_path)
            cfg=GenerationConfig(stars=row['stars'],seed=2026,candidates=1,device=device,style_reference=str(reference_path))
            try:
                audio_source=local_file(row['audio_path'],dest/'audio-source')
                outputs=generate(checkpoint_path,audio_source,dest/'reference-style'/row['id'][:12],cfg,progress,cancelled,cache_root=dest/'shared-audio-cache')
                reference_maps.append({'id':row['id'],'reference_id':reference['id'],'archive':outputs[0],
                                       'playtest_timing':None,'playtest_flow':None,'playtest_fun':None})
            except ValueError as exc:
                reference_maps.append({'id':row['id'],'reference_id':reference['id'],'error':str(exc)})
    v1_maps=[]
    if suite=='full' and v1_run and (Path(v1_run)/'best.pt').is_file():
        from ..generation import GenerationConfig as V1Config, generate as v1_generate
        for row in panel:
            if cancelled(): raise InterruptedError("Evaluation cancelled")
            audio_source=local_file(row['audio_path'],dest/'audio-source')
            for timing_mode in ('automatic','reference'):
                source_map=row['map_path']
                if timing_mode=='reference' and parts(source_map):
                    source_map=str(dest/'reference'/f"{row['id']}.osu")
                    Path(source_map).parent.mkdir(parents=True,exist_ok=True)
                    if not Path(source_map).exists(): mapio.write(read_cached_map(frozen['root'],row),source_map)
                cfg=V1Config(stars=row['stars'],seed=2026,candidates=1,device=device,
                             timing_map=source_map if timing_mode=='reference' else None)
                try:
                    output=v1_generate(Path(v1_run)/'best.pt',audio_source,dest/'v1'/row['id'][:12]/timing_mode,cfg,progress,cancelled)[0]
                    bm=mapio.read(next(Path(output).with_suffix('').glob('*.osu')))
                    v1_maps.append({'id':row['id'],'timing_mode':timing_mode,'archive':output,
                                    'measured_stars':difficulty(bm)['stars'],**mapping_metrics(bm)})
                except ValueError as exc:
                    v1_maps.append({'id':row['id'],'timing_mode':timing_mode,'error':str(exc)})
    style_controls=[]
    if suite=='styles':
        if saved['stage']!='mapper': raise ValueError("Style control evaluation requires a mapper checkpoint")
        for row in panel[:4]:
            if cancelled(): raise InterruptedError("Evaluation cancelled")
            audio_source=local_file(row['audio_path'],dest/'audio-source')
            for control in ('aim','streams','rhythm'):
                outcomes={}
                for level in (0,2):
                    cfg=GenerationConfig(stars=row['stars'],preset='Custom',seed=2026,candidates=1,device=device,**{control:level})
                    folder=dest/'styles'/row['id'][:12]/control/str(level)
                    try:
                        output=generate(checkpoint_path,audio_source,folder,cfg,progress,cancelled,cache_root=dest/'shared-audio-cache')[0]
                        bm=mapio.read(next(Path(output).with_suffix('').glob('*.osu')))
                        outcomes[level]={'archive':output,'metrics':style_metrics(bm,difficulty(bm))}
                    except ValueError as exc: outcomes[level]={'error':str(exc)}
                delta=outcomes[2]['metrics'][control]-outcomes[0]['metrics'][control] if all('metrics' in outcomes[x] for x in (0,2)) else None
                style_controls.append({'id':row['id'],'control':control,'low':outcomes[0],'high':outcomes[2],
                                       'measured_delta':delta,'direction_matches':delta>0 if delta is not None else None})
    report={'split':split,'requested_songs':count,'actual_songs':len(panel),'timing':results,'maps':maps,'v1_overlap_excluded':len(excluded),
            'reference_style_maps':reference_maps,'style_controls':style_controls,'v1_maps':v1_maps,
            'valid_export_fraction':sum('archive' in m for m in maps)/max(1,len(maps)) if maps else None,'quality_verified':False,
            'human_gate':'Playtest 12 complete maps in osu!stable, including four reference-style maps; enter timing/flow/fun scores. Time budget or validation loss is not quality approval.'}
    atomic_json(dest/'report.json',report); progress(report)
    return report


@torch.inference_mode()
def training_sample(model,dataset,run,step,device,dtype,progress=print,cancelled=lambda:False):
    # Read at sample time so a local selection does not require changing checkpoints.
    selection=Path(run)/'training-sample.json'
    if not selection.exists(): selection=Path(run).parent.parent/'training-sample.json'
    if selection.exists():
        return external_training_sample(model,dataset,run,step,device,dtype,load_json(selection),progress,cancelled)
    row=dataset.row(0); reference=read_cached_map(dataset.root,row)
    start=max(0,reference.objects[0].time-1000); end=min(row['duration_ms'],start+16000)
    bm=mapio.Beatmap(timing=[p for p in reference.timing if p.uninherited],difficulty=dict(reference.difficulty))
    cfg=GenerationConfig(stars=row['stars'],temperature=.6,candidates=1)
    was_training=model.training; model.eval()
    try:
        section(model,dataset.encoded(row['audio_hash']),bm,start,end,row['duration_ms'],cfg,row['settings'],describe(reference),device,
                torch.Generator(device=device).manual_seed(2026),dtype,cancelled=cancelled)
        if not bm.objects: raise ValueError("Empty early sample")
        validate_map(bm,row['duration_ms']); stars=difficulty(bm)['stars']
        audio_source=local_file(row['audio_path'],Path(run)/'samples'/'audio-source')
        export(bm,audio_source,Path(run)/'samples'/f'step-{step}',{'style':'V2 training preview','requested_stars':row['stars'],'measured_stars':stars,'reference_timing':True,'partial_song':True,'checkpoint_step':step})
        progress(f"Saved reference-timing preview at step {step}")
    except ValueError as exc:
        atomic_json(Path(run)/'samples'/f'step-{step}'/'rejected.json',{'reason':str(exc),'reference_timing':True,'partial_song':True})
        progress(f"Training preview not exportable yet: {exc}")
    finally: model.train(was_training)


@torch.inference_mode()
def external_training_sample(model,dataset,run,step,device,dtype,selection,progress=print,cancelled=lambda:False):
    """Full-song preview using only the selected audio and the frozen encoder."""
    from .features import encode_recording,load_encoder,VERSION as FEATURE_VERSION
    from .data import identity
    from .generation import _style,choose_settings
    destination=Path(run)/'samples'/f'step-{step}'
    was_training=model.training; rng=random_state(); model.eval()
    metadata={'checkpoint_step':step,'reference_timing':False,'partial_song':False,
              'audio_source':selection.get('audio'),'style':'V2 automatic-timing training sample'}
    try:
        source=Path(selection['audio'])
        manifest=dataset.feature_manifest
        encoder_path=Path(manifest['encoder_checkpoint'])
        if digest(encoder_path)!=manifest['encoder_hash']: raise ValueError('Audio checkpoint identity changed')
        cache=Path(run)/'samples'/'audio-cache'/identity([digest(source),manifest['encoder_hash'],audio.VERSION,FEATURE_VERSION])
        if not (cache/'ready.json').exists():
            progress('Preparing selected sample audio on CPU; subsequent samples reuse its features')
            encoder,_=load_encoder(encoder_path)
            pcm=audio.decode(source); duration=len(pcm)*1000/audio.SR
            mel=audio.spectrogram(pcm); del pcm
            # Keep the training model and optimizer on GPU without adding encoder VRAM.
            encode_recording(encoder,mel,cache,torch.device('cpu'),cancelled=cancelled)
            atomic_json(cache/'ready.json',{'duration_ms':duration})
            del encoder,mel
        duration=load_json(cache/'ready.json')['duration_ms']
        encoded=np.load(cache/'encoded.npy',mmap_mode='r')
        timing,report=estimate(np.load(cache/'beats.npy',mmap_mode='r'))
        metadata['timing_report']=report
        cfg=GenerationConfig(stars=float(selection.get('stars',5)),seed=int(selection.get('seed',2026)),temperature=.6,candidates=1)
        settings=choose_settings(cfg)
        style=_style(dataset.frozen,cfg.stars,cfg.seed,None,0)
        bm=mapio.Beatmap(timing=timing)
        bm.difficulty.update({long:str(settings[short]) for short,long in [('AR','ApproachRate'),('OD','OverallDifficulty'),('CS','CircleSize'),('HP','HPDrainRate')]})
        generator=torch.Generator(device=device).manual_seed(cfg.seed)
        cursor=0
        while cursor<duration:
            if cancelled(): raise InterruptedError('Training sample cancelled')
            end=min(duration,cursor+16000)
            following=section(model,encoded,bm,cursor,end,duration,cfg,settings,style,device,generator,dtype,cancelled=cancelled)
            if following<=cursor: raise ValueError('Sample generation made no progress')
            cursor=following
        if not bm.objects: raise ValueError('Empty early sample')
        validate_map(bm,duration)
        if repetition_report(bm.objects)['degenerate']: raise ValueError('Repetitive training sample rejected')
        measured=difficulty(bm)['stars']
        metadata.update(requested_stars=cfg.stars,measured_stars=measured,seed=cfg.seed,
                        difficulty_miss=abs(measured-cfg.stars)>.5,encoder_hash=manifest['encoder_hash'])
        archive=export(bm,source,destination,metadata)
        progress(f'Saved selected-song sample at step {step}: {archive}')
    except (ValueError,OSError,KeyError) as exc:
        atomic_json(destination/'rejected.json',dict(metadata,reason=str(exc)))
        progress(f'Selected-song training sample rejected: {exc}')
    finally:
        model.train(was_training)
        restore_random(rng)
