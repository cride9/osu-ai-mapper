"""Grammar-constrained, stateful full-song generation with local geometry retries."""
from __future__ import annotations

import copy
import math
from dataclasses import asdict,dataclass
from pathlib import Path

import numpy as np
import torch

from .. import audio,mapio
from ..data import atomic_json,digest,difficulty,load_json,style_metrics
from ..generation import GenerationConfig as BaseConfig, choose_settings, export
from .data import identity,map_from_dict,records
from .dataset import collate,phase_at,summary,slice_array
from .features import encode_recording,load_encoder,STRIDE_MS,VERSION as FEATURE_VERSION
from .geometry import slider_path,validate_map,validate_object
from .model import Mapper,ModelConfig
from .runtime import device_for,gpu_session,load_checkpoint,precision_for
from .style import describe
from .timing import estimate
from .tokenizer import Tokenizer,Grammar
from .quality import generation_bias,repetition_report


@dataclass
class GenerationConfig(BaseConfig):
    style_reference: str|None=None
    style_strength: float=.7
    resume: bool=True
    precision: str="fp16"


def conditioning(encoded,bm,start,end,duration,stars,styles,settings,style):
    tok=Tokenizer(); base=int(start)-32000
    n=math.ceil(64000/STRIDE_MS); origin=round(base/STRIDE_MS)
    local,valid=slice_array(encoded,origin,n); times=origin*STRIDE_MS+np.arange(n)*STRIDE_MS
    glob,global_times=summary(encoded)
    prior=bm.copy(); prior.objects=[o for o in bm.objects if o.time<start]; prior.breaks=[b for b in bm.breaks if b[0]<start]
    hist=tok.context(prior,base,start,1024) or [tok.ids['CTX']]
    positions=np.arange(0,max(0,start-1999),max(2000,math.ceil(max(1,start)/128/2000)*2000))
    past=np.stack([describe(prior,t,t+2000) for t in positions]) if len(positions) else np.zeros((1,64),np.float32)
    if not len(positions): positions=np.zeros(1)
    busy=max([start,*[o.time+prior.duration(o) for o in prior.objects],*[b for _,b in prior.breaks]])
    x,y=.5,.5
    if prior.objects:
        last=prior.objects[-1]; xy=(last.x,last.y)
        if last.kind=='slider' and last.repeats%2: xy=slider_path(last)[-1]
        x,y=xy[0]/512,xy[1]/384
    prefix=tok.condition(stars,styles,settings)+tok.time(end-1,base)+[tok.ids['GEN']]
    values=dict(tokens=np.array(prefix,np.int64),labels=np.full(len(prefix),-100,np.int64),history=np.array(hist,np.int64),history_valid=np.ones(len(hist),bool),
                local_audio=local,local_valid=valid,local_times=times-base,phase=phase_at(prior.timing,times),global_audio=glob,global_times=global_times-base,
                global_valid=np.ones(len(glob),bool),plan=np.zeros((len(glob),64),np.float32),style=np.array(style,np.float32),
                past=past,past_valid=np.ones(len(past),bool),past_times=positions-base,
                state=np.array([min(1,(busy-start)/64000),x,y,start/max(duration,1),busy>start,0,1,0],np.float32))
    sample={k:torch.from_numpy(np.asarray(v,dtype=np.float32) if np.asarray(v).dtype.kind=='f' else np.asarray(v).copy()) for k,v in values.items()}
    return sample,prefix,base,busy


def _device_batch(sample,device):
    return {k:v[None].to(device) for k,v in sample.items()}


@torch.inference_mode()
def section(model,encoded,bm,start,end,duration,cfg,settings,style,device,generator,dtype,accepted=None,save=None,cancelled=lambda:False,token_budget=None):
    sample,prefix,base,busy=conditioning(encoded,bm,start,end,duration,cfg.stars,cfg.styles(),settings,style)
    budget=token_budget or model.config.max_tokens
    grammar=Grammar(Tokenizer(),base,start,end,duration,busy,cs=settings['CS'])
    tok=grammar.tok; done=list(accepted or [])
    for token in done:
        if token not in grammar.allowed(budget-len(prefix)):
            raise ValueError("Corrupt generation continuation")
        grammar.consume(token)
    with torch.autocast(device.type,dtype=dtype,enabled=device.type=='cuda'):
        memory,mask,plan=model.memory(_device_batch(sample,device))
        logits,caches=model.step(torch.tensor([prefix+done],device=device),memory,mask)
    offset=len(prefix)+len(done); completed=list(done); rng_at_commit=generator.get_state()
    event_start=len(done); grammar_before=copy.copy(grammar); logits_before=logits; offset_before=offset
    retries=0; indices_cache={}; added=0; forced=False
    while offset<budget:
        if cancelled():
            generator.set_state(rng_at_commit)
            if save: save(completed,plan[0].float().cpu().tolist())
            raise InterruptedError("Generation cancelled at the last complete object; resume the same settings")
        remaining=budget-offset
        allowed=grammar.allowed(remaining)
        if not allowed: break
        forced=grammar.state=='event' and remaining<32 and grammar.minimum<end
        key=tuple(allowed)
        if key not in indices_cache:
            if len(indices_cache)>512: indices_cache.clear()
            indices_cache[key]=torch.tensor(allowed,device=device)
        indices=indices_cache[key]; values=logits[0,indices].float()
        if grammar.kind=='CIRCLE':
            bias=generation_bias(bm.objects,grammar.state,indices,tok)
            if bias is not None: values=values+bias.to(values.device)
        if not torch.isfinite(values).all(): raise ValueError("Non-finite generation logits")
        if cfg.temperature<=0: token=int(indices[values.argmax()])
        else:
            prob=torch.softmax(values/cfg.temperature,0); prob,order=torch.sort(prob,descending=True)
            prob[prob.cumsum(0)-prob>=cfg.top_p]=0
            token=int(indices[order[torch.multinomial(prob,1,generator=generator)]].item())
        grammar.consume(token); done.append(token)
        if token==tok.ids['EOS']: break
        if token==tok.ids['END']:
            scratch=mapio.Beatmap(difficulty=dict(bm.difficulty),timing=copy.deepcopy(bm.timing))
            try:
                tok.decode(done[event_start:],scratch,base)
                scratch.validate(duration)
                if scratch.objects: validate_object(scratch.objects[0],settings['CS'])
            except (ValueError,IndexError,OverflowError) as exc:
                retries+=1
                if retries>3: raise ValueError(f"Object failed geometry after three retries: {exc}") from exc
                grammar=copy.copy(grammar_before); done=done[:event_start]; offset=offset_before; logits=logits_before
                continue
            if repetition_report(bm.objects+scratch.objects)['degenerate']:
                retries+=1
                if retries>3: raise ValueError("Persistent stack or two-point jump loop after three local retries")
                grammar=copy.copy(grammar_before); done=done[:event_start]; offset=offset_before; logits=logits_before
                continue
            bm.objects.extend(scratch.objects); bm.breaks.extend(scratch.breaks); bm.timing=scratch.timing
            completed=list(done); added+=1; rng_at_commit=generator.get_state(); retries=0
        with torch.autocast(device.type,dtype=dtype,enabled=device.type=='cuda'):
            logits,caches=model.step(torch.tensor([[token]],device=device),memory,mask,caches,offset)
        offset+=1
        if token==tok.ids['END']:
            event_start=len(done); grammar_before=copy.copy(grammar); logits_before=logits; offset_before=offset
            if save and added%16==0: save(completed,plan[0].float().cpu().tolist())
    if grammar.state not in ('done','event'): raise ValueError("Token budget ended inside an object")
    if forced or (offset>=budget and grammar.state=='event'):
        starts=[o.time for o in bm.objects if o.time>=start]+[a for a,b in bm.breaks if a>=start]
        if not starts: raise ValueError("No progress before token capacity")
        return max(starts)+1
    return end


def _style(frozen,stars,seed,reference,strength):
    rng=np.random.default_rng(seed); pool=[]
    for r in records(frozen,'train'):
        if abs(r['stars']-stars)<1: pool.append(r['id'])
    if not pool: pool=[r['id'] for r in records(frozen,'train')]
    if not pool: raise ValueError("No train-only style pool")
    mid=pool[int(rng.integers(len(pool)))]
    if frozen.get('storage_mode')=='rolling':
        from .rolling import map_bundle
        from .data import readonly
        import json
        db=readonly(frozen['index'])
        try: row=json.loads(db.execute('SELECT payload FROM records WHERE id=?',(mid,)).fetchone()[0])
        finally: db.close()
        a=map_bundle(frozen['root'],row)['style']
    else: a=np.load(Path(frozen['root'])/'maps'/mid/'style.npy',mmap_mode='r')
    code=np.asarray(a.mean(0),np.float32)
    if reference: code=(1-strength)*code+strength*describe(mapio.read(reference))
    return code


def generate(checkpoint_path,inputs,output_dir,cfg=None,progress=print,cancelled=lambda:False,cache_root=None):
    cfg=cfg or GenerationConfig(); cfg.styles()
    if not 0<=cfg.style_strength<=1 or cfg.candidates<1 or cfg.variants<1 or not 0<=cfg.stars<=20: raise ValueError("Invalid generation controls")
    if cfg.temperature<0 or not 0<cfg.top_p<=1: raise ValueError("Invalid sampling controls")
    for key in ('ar','od','cs','hp'):
        value=getattr(cfg,key)
        if value is not None and not 0<=value<=10: raise ValueError(f"{key} must be 0–10")
    checkpoint_path=Path(checkpoint_path); saved=load_checkpoint(checkpoint_path)
    if saved['stage']!='mapper': raise ValueError("Use a V2 mapper checkpoint for generation")
    frozen=load_json(checkpoint_path.parent/'dataset.json'); feature_manifest=load_json(checkpoint_path.parent/'features.json')
    if not frozen or not feature_manifest: raise ValueError("Keep dataset.json and features.json beside the checkpoint")
    encoder,encoder_state=load_encoder(feature_manifest['encoder_checkpoint'])
    if digest(feature_manifest['encoder_checkpoint'])!=feature_manifest['encoder_hash']: raise ValueError("Audio checkpoint identity changed")
    model=Mapper(len(Tokenizer()),ModelConfig(**saved['model_config'])); model.load_state_dict(saved['model'])
    del saved['model']; model.eval()
    for key in ('optimizer','scheduler','scaler','random'): saved.pop(key,None)
    device=device_for(cfg.device); dtype=precision_for(cfg.precision,device); torch.set_num_threads(4)
    root=Path(output_dir).resolve(); root.mkdir(parents=True,exist_ok=True)
    if isinstance(inputs,(str,Path)):
        p=Path(inputs); inputs=sorted(x for x in p.rglob('*') if x.suffix.lower() in audio.EXTENSIONS) if p.is_dir() else [p]
    if not inputs: raise ValueError("No songs selected")
    outputs=[]; failures=[]; checkpoint_hash=digest(checkpoint_path)
    settings=choose_settings(cfg)
    with gpu_session(device,root):
        for ap in inputs:
            if cancelled(): raise InterruptedError("Generation cancelled")
            ap=Path(ap).resolve(); ah=digest(ap)
            audio_dest=Path(cache_root or root/'audio-cache')/identity([ah,feature_manifest['encoder_hash'],audio.VERSION,FEATURE_VERSION])
            if not (audio_dest/'ready.json').exists():
                progress(f"Encoding {ap.name} once for all candidates")
                pcm=audio.decode(ap); duration=len(pcm)*1000/audio.SR
                activity=np.array([np.max(np.abs(pcm[i:i+audio.SR])) for i in range(0,len(pcm),audio.SR)],np.float32)
                mel=audio.spectrogram(pcm); del pcm
                encoder.to(device); model.cpu()
                if device.type=='cuda': torch.cuda.empty_cache()
                encode_recording(encoder,mel,audio_dest,device,dtype,cancelled)
                from .data import save_array
                save_array(audio_dest/'activity.npy',activity)
                atomic_json(audio_dest/'ready.json',{'duration_ms':duration}); del mel
            duration=load_json(audio_dest/'ready.json')['duration_ms']
            encoded=np.load(audio_dest/'encoded.npy',mmap_mode='r'); scores=np.load(audio_dest/'beats.npy',mmap_mode='r')
            activity=np.load(audio_dest/'activity.npy',mmap_mode='r') if (audio_dest/'activity.npy').is_file() else None
            if cfg.timing_map:
                timing=[p for p in mapio.read(cfg.timing_map).timing if p.uninherited]; timing_report={'reference':str(cfg.timing_map),'manual':True}
            else: timing,timing_report=estimate(scores,bpm=cfg.bpm,offset=cfg.offset)
            encoder.cpu()
            if device.type=='cuda': torch.cuda.empty_cache()
            model.to(device)
            for variant in range(cfg.variants):
                result_path=root/'continuations'/('variant-'+identity([checkpoint_hash,ah,asdict(cfg),variant])+'.json')
                prior_result=load_json(result_path) if cfg.resume else None
                if prior_result and Path(prior_result.get('archive','')).is_file():
                    outputs.append(prior_result['archive']); progress(f"Reused complete variant {variant+1} for {ap.name}")
                    continue
                style=_style(frozen,cfg.stars,cfg.seed+variant,cfg.style_reference,cfg.style_strength)
                candidates=[]; errors=[]
                for attempt in range(cfg.candidates):
                    seed=cfg.seed+variant*cfg.candidates+attempt
                    key=identity([checkpoint_hash,ah,asdict(cfg),variant,attempt,style.tolist(),[asdict(t) for t in timing]])
                    state_path=root/'continuations'/(key+'.json')
                    state=load_json(state_path) if cfg.resume else None
                    if state and state.get('status')=='rejected': errors.append(state['error']); continue
                    if state and state.get('status')=='complete':
                        bm=map_from_dict(state['map']); validate_map(bm,duration)
                        stats=difficulty(bm); repetition=repetition_report(bm.objects)
                        if not repetition['degenerate']: candidates.append((abs(stats['stars']-cfg.stars)+.6*repetition['penalty'],bm,stats,seed,repetition))
                        continue
                    generator=torch.Generator(device=device).manual_seed(seed)
                    if state:
                        bm=map_from_dict(state['map']); cursor=state['cursor']; saved_tokens=state.get('section_tokens',[])
                        generator.set_state(torch.tensor(state['rng'],dtype=torch.uint8))
                    else:
                        bm=mapio.Beatmap(timing=copy.deepcopy(timing)); cursor=0; saved_tokens=[]
                        bm.difficulty.update({long:str(settings[short]) for short,long in [('AR','ApproachRate'),('OD','OverallDifficulty'),('CS','CircleSize'),('HP','HPDrainRate')]})
                    def persist(tokens,plan=None,status='running'):
                        atomic_json(state_path,{'identity':key,'status':status,'cursor':cursor,'section_tokens':tokens,'map':asdict(bm),'style':style.tolist(),'plan':plan,
                                                'rng':generator.get_state().cpu().tolist(),'checkpoint':checkpoint_hash,'audio':ah,'timing_report':timing_report})
                    try:
                        while cursor<duration:
                            end=min(cursor+16000,math.ceil(duration))
                            progress(f"{ap.name} variant {variant+1}, candidate {attempt+1}: {cursor/1000:.1f}/{duration/1000:.1f}s")
                            if cancelled(): persist(saved_tokens); raise InterruptedError("Generation cancelled")
                            if activity is not None and not saved_tokens and max(activity[int(cursor//1000):max(int(cursor//1000)+1,int(math.ceil(end/1000)))],default=0)<1e-5:
                                cursor=end; persist([])
                                continue
                            previous=cursor
                            cursor=section(model,encoded,bm,cursor,end,duration,cfg,settings,style,device,generator,dtype,saved_tokens,persist,cancelled)
                            if cursor<=previous: raise ValueError("Generation did not advance")
                            saved_tokens=[]; persist([])
                        if not bm.objects: raise ValueError("Generated an empty song")
                        validate_map(bm,duration); stats=difficulty(bm)
                        repetition=repetition_report(bm.objects)
                        if repetition['degenerate']: raise ValueError("Persistent repetitive pattern in complete song")
                        candidates.append((abs(stats['stars']-cfg.stars)+.6*repetition['penalty'],bm,stats,seed,repetition)); persist([],status='complete')
                    except ValueError as exc:
                        errors.append(str(exc)); progress(f"Rejected candidate: {exc}")
                        atomic_json(state_path,{'status':'rejected','error':str(exc),'identity':key})
                if not candidates:
                    failures.append({'audio':str(ap),'variant':variant,'errors':errors}); continue
                _,bm,stats,seed,repetition=min(candidates,key=lambda x:x[0])
                miss=abs(stats['stars']-cfg.stars)
                metadata={'version':'v2','requested_stars':cfg.stars,'measured_stars':stats['stars'],'difficulty_miss':miss>.5,'repetition':repetition,'style':cfg.preset,'style_code':style.tolist(),
                          'settings':asdict(cfg),'checkpoint':str(checkpoint_path.resolve()),'checkpoint_sha256':checkpoint_hash,'checkpoint_step':saved['step'],
                          'encoder_hash':feature_manifest['encoder_hash'],'dataset_hash':frozen['hash'],'seed':seed,'timing':timing_report,'style_metrics':style_metrics(bm,stats),'candidate_errors':errors,'complete_song':True}
                archive=export(bm,ap,root,metadata)
                outputs.append(archive)
                atomic_json(result_path,{'archive':archive,'checkpoint':checkpoint_hash,'audio':ah,'variant':variant,'settings':asdict(cfg)})
                progress(f"Exported {stats['stars']:.2f} stars; requested {cfg.stars:.2f}"+(' (target missed)' if miss>.5 else ''))
    atomic_json(root/'generation-report.json',{'outputs':outputs,'failures':failures})
    if not outputs: raise ValueError("No valid complete maps; see generation-report.json")
    return outputs


def replace_training_history(model,dataset,batch,step,cfg,device,dtype):
    """On-policy complete previous-window rollouts; never use future target events."""
    metas=batch.get('meta',[])
    changed=[]; was_training=model.training; model.eval()
    try:
        for i,meta in enumerate(metas):
            rng=np.random.default_rng(cfg.seed+step*1009+meta['number'])
            if meta['window']==0 or rng.random()>=cfg.rollout_probability: continue
            row=dataset.row(meta['index']); a=dataset.arrays(row['id'])
            from .data import read_cached_map
            # This branch is occasional on-policy generation, not the training input path.
            reference=read_cached_map(dataset.root,row)
            previous_start=int(a['windows'][meta['window']-1,0]); current_start=meta['start']
            reference.objects=[o for o in reference.objects if o.time<previous_start]
            reference.breaks=[b for b in reference.breaks if b[0]<previous_start]
            gen=GenerationConfig(stars=row['stars'],preset='Custom',**row['styles'],temperature=.8,candidates=1)
            generator=torch.Generator(device=device).manual_seed(cfg.seed+step*997+meta['number'])
            try:
                section(model,dataset.encoded(row['audio_hash']),reference,previous_start,current_start,current_start,gen,row['settings'],batch['style'][i].numpy(),device,generator,dtype,token_budget=min(1024,model.config.max_tokens))
            except ValueError:
                # An unsuccessful rollout is an empty recent history, not secretly ground truth.
                reference.objects=[o for o in reference.objects if o.time<previous_start]
            history=Tokenizer().context(reference,meta['base'],current_start,1024) or [Tokenizer().ids['CTX']]
            changed.append((i,torch.tensor(history)))
    finally: model.train(was_training)
    if changed:
        n=max(batch['history'].shape[1],max(len(h) for _,h in changed))
        history=torch.zeros((len(metas),n),dtype=torch.long); valid=torch.zeros_like(history,dtype=torch.bool)
        history[:,:batch['history'].shape[1]]=batch['history']; valid[:,:batch['history'].shape[1]]=batch['history_valid']
        for i,h in changed:
            history[i].zero_(); valid[i].zero_(); history[i,:len(h)]=h; valid[i,:len(h)]=True
        batch=dict(batch,history=history,history_valid=valid)
    return batch
