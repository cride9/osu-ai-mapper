"""Measured 6 GB acceptance, including optimizer allocation and worst-case shapes."""
from __future__ import annotations

import gc
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ..data import atomic_json
from .model import AudioEncoder,Mapper,ModelConfig,parameter_counts
from .runtime import device_for,enforce_headroom,gpu_session,precision_for
from .tokenizer import Tokenizer
from .training import TrainConfig,make_optimizer,audio_loss,mapper_loss


def synthetic(config,batch,device,stage='mapper',small=False):
    frames=128 if small else 6416
    if stage=='audio':
        mel=torch.randn(batch,128,frames,device=device)
        return {'mel':mel,'clean':mel.clone(),'masked':torch.rand_like(mel)>.8,'beats':torch.zeros(batch,frames,2,device=device),'beat_mask':torch.ones(batch,frames,2,device=device)}
    seq=32 if small else config.max_tokens; history=16 if small else config.history_tokens
    local=frames//4; global_n=8 if small else 512; past=4 if small else 128
    ones=lambda n:torch.ones(batch,n,dtype=torch.bool,device=device)
    tokens=torch.randint(1,len(Tokenizer()),(batch,seq),device=device)
    return dict(tokens=tokens,labels=tokens.clone(),token_valid=ones(seq),history=torch.randint(1,len(Tokenizer()),(batch,history),device=device),history_valid=ones(history),
                local_audio=torch.randn(batch,local,config.audio_width,device=device),local_valid=ones(local),local_times=torch.arange(local,device=device).float()[None].expand(batch,-1)*40,
                phase=torch.randn(batch,local,3,device=device),global_audio=torch.randn(batch,global_n,config.audio_width,device=device),global_valid=ones(global_n),
                global_times=torch.arange(global_n,device=device).float()[None].expand(batch,-1)*2000,plan=torch.zeros(batch,global_n,64,device=device),
                style=torch.randn(batch,64,device=device),past=torch.randn(batch,past,64,device=device),past_valid=ones(past),past_times=torch.zeros(batch,past,device=device),state=torch.zeros(batch,8,device=device))


def trial(name='v2-s',stage='mapper',batch_size=1,device='auto',checkpointing=True,precision='fp16',steps=3,small=False):
    actual=device_for(device); cfg=ModelConfig.preset(name); cfg.checkpointing=checkpointing
    dtype=precision_for(precision,actual); torch.set_num_threads(4)
    with gpu_session(actual):
        if actual.type=='cuda': torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(actual)
        model=AudioEncoder(cfg) if stage=='audio' else Mapper(len(Tokenizer()),cfg)
        model.to(actual).train(); counts=parameter_counts(model)
        tc=TrainConfig(stage=stage,model=name,batch_size=batch_size,device=device)
        optimizer,backend=make_optimizer(model,tc,actual)
        scaler=torch.amp.GradScaler('cuda',enabled=actual.type=='cuda' and dtype==torch.float16)
        b=synthetic(cfg,batch_size,actual,stage,small); elapsed=[]; last=0
        for i in range(steps+1):
            tick=time.monotonic(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(actual.type,dtype=dtype,enabled=actual.type=='cuda'):
                if stage=='audio': loss,_,_=audio_loss(model,b)
                else:
                    loss,count,plan,boundary=mapper_loss(model,b); loss=loss/count+.1*plan+.05*boundary
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm): raise FloatingPointError("Nonfinite benchmark loss/gradients")
            scaler.step(optimizer); scaler.update()
            if actual.type=='cuda': torch.cuda.synchronize(actual)
            if i: elapsed.append(time.monotonic()-tick)
            last=float(loss.detach()); memory=enforce_headroom(actual)
        # Profile the actual native attention implementation, separate from throughput.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            with torch.no_grad(),torch.autocast(actual.type,dtype=dtype,enabled=actual.type=='cuda'):
                if stage=='audio': model(b['mel'])
                else: model(b)
        attention=sorted({event.key for event in prof.key_averages() if 'attention' in event.key})
        result={'model':name,'stage':stage,'batch_size':batch_size,'checkpointing':checkpointing,'precision':precision,'optimizer_backend':backend,
                'attention_backend_events':attention,'parameters':counts,'seconds_per_microbatch':sum(elapsed)/len(elapsed),'loss':last,'passed':True,
                'max_shape':not small,'target_tokens_per_second':batch_size*b['tokens'].shape[1]/(sum(elapsed)/len(elapsed)) if stage=='mapper' else None,**memory}
        del optimizer,model,b,loss; gc.collect()
        if actual.type=='cuda': torch.cuda.empty_cache()
        return result


def benchmark(output_dir,compare=('v2-s','v2-l'),device='auto',steps=3,batches=(1,2,4,8),hours=2,progress=print,cancelled=lambda:False):
    root=Path(output_dir); root.mkdir(parents=True,exist_ok=True)
    results=[]; began=time.monotonic()
    for stage,name in [('audio',compare[0])]+[('mapper',n) for n in compare]:
        for checkpointing in (True,False):
            for batch in batches:
                if cancelled(): raise InterruptedError("Benchmark cancelled")
                if time.monotonic()-began>=hours*3600: break
                progress(f"Benchmark {stage}/{name}: batch {batch}, checkpointing={checkpointing}")
                try: result=trial(name,stage,batch,device,checkpointing,steps=steps)
                except (RuntimeError,FloatingPointError) as exc:
                    result={'model':name,'stage':stage,'batch_size':batch,'checkpointing':checkpointing,'passed':False,'error':str(exc)}
                    results.append(result); progress(result); gc.collect()
                    if torch.cuda.is_initialized(): torch.cuda.empty_cache()
                    break
                results.append(result); progress(result)
                atomic_json(root/'benchmark.json',{'trials':results,'quality_verified':False})
    chosen={}
    for stage,name in [('audio',compare[0])]+[('mapper',n) for n in compare]:
        valid=[r for r in results if r['passed'] and r['stage']==stage and r['model']==name]
        if valid: chosen[f'{stage}/{name}']=min(valid,key=lambda r:r['seconds_per_microbatch']/r['batch_size'])
    report={'trials':results,'recommended':chosen,'limits':{'max_reserved_gib':4.8,'min_free_gib':.75},'quality_verified':False,'elapsed_seconds':time.monotonic()-began}
    atomic_json(root/'benchmark.json',report)
    if not chosen: raise RuntimeError("No configuration met the 6 GB acceptance limits; context was not shortened")
    return report


def overfit(data_root,output_dir,device='auto',steps=100,hours=.5,progress=print,cancelled=lambda:False):
    from .data import snapshot
    from .dataset import AudioDataset,collate
    from .training import move
    root=Path(output_dir); frozen=snapshot(data_root,root)
    actual=device_for(device); cfg=ModelConfig.preset('v2-s'); cfg.checkpointing=True
    ds=AudioDataset(frozen); sample=collate([ds[(0,0,0)]])
    values=[]; began=time.monotonic()
    with gpu_session(actual):
        model=AudioEncoder(cfg).to(actual).train(); b=move(sample,actual)
        optimizer,_=make_optimizer(model,TrainConfig(),actual)
        scaler=torch.amp.GradScaler('cuda',enabled=actual.type=='cuda')
        for step in range(steps):
            if cancelled(): raise InterruptedError('Overfit cancelled')
            if time.monotonic()-began>hours*3600: break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(actual.type,dtype=torch.float16,enabled=actual.type=='cuda'): loss,_,_=audio_loss(model,b)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm): raise FloatingPointError('Overfit produced nonfinite gradients')
            scaler.step(optimizer); scaler.update(); values.append(float(loss.detach()))
            enforce_headroom(actual)
            if step%10==0: progress(f'Overfit {step+1}/{steps}: {values[-1]:.4f}')
    count=min(10,len(values)//2)
    report={'losses':values,'passed':count>=3 and sum(values[-count:])/count<.9*sum(values[:count])/count,'elapsed_seconds':time.monotonic()-began}
    atomic_json(root/'report.json',report)
    return report


def benchmark_io(data_root,output_dir,progress=print,cancelled=lambda:False):
    from torch.utils.data import DataLoader
    from ..data import load_json
    from .data import snapshot
    from .dataset import MapDataset,collate
    from .training import UpdateSampler
    root=Path(output_dir); frozen=snapshot(data_root,root/'io-snapshot')
    ds=MapDataset(frozen,load_json(Path(data_root)/'features.json'))
    measurements=[]
    for workers in (0,2,4):
        if cancelled(): raise InterruptedError('Data benchmark cancelled')
        cfg=TrainConfig(steps=4,batch_size=1,workers=workers)
        loader=DataLoader(ds,batch_sampler=UpdateSampler(ds,cfg,0),collate_fn=collate,num_workers=workers,
                          **({'multiprocessing_context':'spawn','persistent_workers':True,'prefetch_factor':2} if workers else {}))
        iterator=iter(loader); tick=time.monotonic(); valid=0; steady=None
        try:
            for i,b in enumerate(iterator):
                if cancelled(): raise InterruptedError('Data benchmark cancelled')
                if i==16: steady=time.monotonic()
                if i>=16: valid+=int((b['labels']!=-100).sum())
        finally:
            if hasattr(iterator,'_shutdown_workers'): iterator._shutdown_workers()
        elapsed=time.monotonic()-tick
        measurements.append({'workers':workers,'startup_inclusive_seconds':elapsed,'target_tokens_per_second':valid/max(.001,time.monotonic()-(steady or tick))})
        progress(measurements[-1])
    best=max(measurements,key=lambda x:x['target_tokens_per_second'])
    if best['target_tokens_per_second']<1.1*measurements[0]['target_tokens_per_second']: best=measurements[0]
    report={'measurements':measurements,'recommended_workers':best['workers']}
    atomic_json(root/'data-loader.json',report); return report
