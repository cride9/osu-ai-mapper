"""Resumable two-stage training with bounded shapes and token-correct losses."""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..data import atomic_json, digest, load_json
from .data import snapshot, identity
from .dataset import AudioDataset, MapDataset, collate
from .model import AudioEncoder, Mapper, ModelConfig, parameter_counts, token_loss
from .runtime import device_for, enforce_headroom, gpu_session, load_checkpoint, precision_for, random_state, restore_random, save_checkpoint
from .tokenizer import Tokenizer, VERSION as TOKEN_VERSION
from .penalties import out_of_bounds_loss
from .hard_samples import SPIKE_THRESHOLD, write_spikes


@dataclass
class TrainConfig:
    stage: str = "mapper"
    model: str = "v2-s"
    steps: int = 100000
    hours: float = 120
    batch_size: int = 1
    effective_batch: int = 32
    workers: int = 0
    lr: float = 3e-4
    warmup: int = 500
    validate_every: int = 250
    sample_every: int = 1000
    checkpoint_minutes: float = 15
    seed: int = 42
    device: str = "auto"
    precision: str = "fp16"
    checkpointing: bool = True
    fused_optimizer: bool = True
    compile_model: bool = False
    rollout_after: int = 1000
    rollout_probability: float = .2
    validation_maps: int = 16


class UpdateSampler:
    def __init__(self,dataset,cfg,start): self.dataset,self.cfg,self.start=dataset,cfg,start
    def __iter__(self):
        for update in range(self.start,self.cfg.steps):
            selection=self.dataset.choose(update,self.cfg.effective_batch)
            for i in range(0,len(selection),self.cfg.batch_size): yield selection[i:i+self.cfg.batch_size]
    def __len__(self): return max(0,self.cfg.steps-self.start)*math.ceil(self.cfg.effective_batch/self.cfg.batch_size)


def move(batch,device):
    return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def make_optimizer(model,cfg,device):
    kwargs=dict(lr=cfg.lr,weight_decay=.01,eps=1e-8)
    if device.type=="cuda" and cfg.fused_optimizer:
        try: return torch.optim.AdamW(model.parameters(),fused=True,**kwargs),"fused"
        except (TypeError,RuntimeError): pass
    return torch.optim.AdamW(model.parameters(),foreach=False,**kwargs),"native"


def make_scheduler(optimizer,cfg):
    def factor(step):
        if step<cfg.warmup: return max(1,step)/max(1,cfg.warmup)
        return .1+.9*.5*(1+math.cos(math.pi*min(1,(step-cfg.warmup)/max(1,cfg.steps-cfg.warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)


def audio_loss(model,b,return_samples=False):
    _,beats,recon=model(b['mel'])
    raw=F.binary_cross_entropy_with_logits(beats.float(),b['beats'],reduction='none',pos_weight=beats.new_tensor([6,12],dtype=torch.float32))
    beat=(raw*b['beat_mask']).sum()/b['beat_mask'].sum().clamp_min(1)
    squared=(recon.float()-b['clean']).square()*b['masked']
    reconstruction=squared.sum()/b['masked'].sum().clamp_min(1)
    if not return_samples:
        return beat+.1*reconstruction,beat.detach(),reconstruction.detach()
    # Diagnostics only: the optimized batch loss above is unchanged. Each
    # individual loss is normalized by that window's supervised frame count.
    axes=(1,2)
    per_beat=(raw*b['beat_mask']).sum(axes)/b['beat_mask'].sum(axes).clamp_min(1)
    per_recon=squared.sum(axes)/b['masked'].sum(axes).clamp_min(1)
    individual=torch.stack((per_beat+.1*per_recon,per_beat,per_recon),dim=-1).detach()
    return beat+.1*reconstruction,beat.detach(),reconstruction.detach(),individual


def backward_with_recovery(model, optimizer, scaler, backward, progress=print, max_retries=8):
    """Replay an accumulated update after AMP overflow, without advancing data or weights."""
    rng = random_state()
    for attempt in range(max_retries + 1):
        if attempt:
            restore_random(rng)
        optimizer.zero_grad(set_to_none=True)
        result = backward()
        stats = result[0]
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not bool(torch.isfinite(stats).all()):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite forward loss; retained last completed checkpoint")
        if bool(torch.isfinite(norm)):
            return result, attempt
        optimizer.zero_grad(set_to_none=True)
        if not scaler.is_enabled() or attempt == max_retries:
            raise FloatingPointError("Persistent non-finite gradient; retained last completed checkpoint")
        # No optimizer/scheduler step on a failed attempt. Reset AMP's per-optimizer
        # state and lower the scale, including when only the gradient norm overflowed.
        scale = scaler.get_scale()
        scaler.update(new_scale=scale * scaler.get_backoff_factor())
        progress(f"AMP gradient overflow: retry {attempt + 1}/{max_retries}, scale {scale:g} -> {scaler.get_scale():g}")


def mapper_loss(model,b):
    hidden,plan=model(b)
    total,count=token_loss(model,hidden,b['labels'])
    valid=b['global_valid'][:,:,None]
    plan_loss=(F.smooth_l1_loss(plan.float(),b['plan'],reduction='none')*valid).sum()/(valid.sum()*64).clamp_min(1)
    boundary=out_of_bounds_loss(model,hidden,b)
    return total,count,plan_loss,boundary


@torch.inference_mode()
def validate(model,dataset,cfg,device,dtype):
    model.eval(); sums=np.zeros(3); examples=0
    indices=np.linspace(0,len(dataset)-1,min(cfg.validation_maps,len(dataset))).astype(int)
    for index in indices:
        b=move(collate([dataset[int(index)]]),device)
        with torch.autocast(device.type,dtype=dtype,enabled=device.type=='cuda'):
            if cfg.stage=='audio':
                loss,beat,recon=audio_loss(model,b); values=[loss.item(),beat.item(),recon.item()]
            else:
                total,count,plan,boundary=mapper_loss(model,b); values=[total.item()/max(1,count)+.1*plan.item()+.05*boundary.item(),total.item()/max(1,count),plan.item()]
        sums+=values; examples+=1
    model.train()
    return dict(zip(('loss','beat_loss','reconstruction_loss') if cfg.stage=='audio' else ('loss','token_loss','plan_loss'),(sums/max(1,examples)).tolist()))


def train(data_root,run_dir,cfg=None,resume=None,features=None,progress=print,cancelled=lambda:False):
    cfg=cfg or TrainConfig(); run=Path(run_dir).resolve(); run.mkdir(parents=True,exist_ok=True)
    if cfg.stage not in ('audio','mapper'): raise ValueError("Stage must be audio or mapper")
    if cfg.batch_size not in (1,2,4,8) or cfg.effective_batch%cfg.batch_size: raise ValueError("Physical batch must divide effective batch")
    if cfg.workers not in (0,2,4): raise ValueError("Workers must be 0, 2 or 4")
    if cfg.steps<1 or cfg.hours<=0 or not 0<=cfg.rollout_probability<=1: raise ValueError("Invalid training budget")
    state=load_checkpoint(resume) if resume else None
    if not resume and (run/'last.pt').exists(): raise ValueError("Run already exists; resume last.pt or choose a new directory")
    if state:
        if state['stage']!=cfg.stage or state['train_config']['model']!=cfg.model: raise ValueError("Resume stage/model mismatch")
        frozen=load_json(Path(resume).parent/'dataset.json')
        if not frozen or frozen['snapshot_hash']!=state['snapshot_hash']: raise ValueError("Frozen dataset/label mismatch")
        # Preserve schedule, sampling and architecture; only safe runtime/budget extensions change.
        saved=TrainConfig(**state['train_config'])
        for key in ('batch_size','workers','device','checkpointing','hours'): setattr(saved,key,getattr(cfg,key))
        cfg=saved
        atomic_json(run/'dataset.json',frozen)
    else: frozen=snapshot(data_root,run)
    if digest(frozen['index'])!=frozen['index_hash']: raise ValueError("Frozen index checksum mismatch")
    feature_manifest=None
    if cfg.stage=='mapper':
        feature_manifest=load_json(features or Path(data_root)/'features.json')
        if not feature_manifest or feature_manifest['dataset_hash']!=frozen['hash']: raise ValueError("Build matching frozen-encoder features before mapper training")
        if state and state.get('feature_hash')!=identity(feature_manifest): raise ValueError("Resume encoder/features mismatch")
        atomic_json(run/'features.json',feature_manifest)
    config=ModelConfig(**state['model_config']) if state else ModelConfig.preset(cfg.model)
    config.checkpointing=cfg.checkpointing
    if feature_manifest and config.audio_width!=feature_manifest['model_config']['audio_width']: raise ValueError("Encoder feature width does not match mapper preset")
    device=device_for(cfg.device); dtype=precision_for(cfg.precision,device)
    torch.set_num_threads(4)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    dataset=AudioDataset(frozen,seed=cfg.seed) if cfg.stage=='audio' else MapDataset(frozen,feature_manifest,seed=cfg.seed)
    validation=AudioDataset(frozen,'validation',cfg.seed) if cfg.stage=='audio' else MapDataset(frozen,feature_manifest,'validation',cfg.seed)
    start=state['step'] if state else 0; elapsed=state['elapsed_seconds'] if state else 0.
    best=state['best_validation'] if state else float('inf')
    with gpu_session(device,run):
        model=AudioEncoder(config) if cfg.stage=='audio' else Mapper(len(Tokenizer()),config)
        if state: model.load_state_dict(state['model'])
        model.to(device).train()
        optimizer,backend=make_optimizer(model,cfg,device); scheduler=make_scheduler(optimizer,cfg)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and dtype==torch.float16)
        if state:
            optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler']); restore_random(state['random'])
        callable_model=model
        if cfg.compile_model:
            try: callable_model=torch.compile(model)
            except Exception as exc: raise RuntimeError("Optional torch.compile unavailable; disable compile and use native eager") from exc
        atomic_json(run/'config.json',{'version':'v2','training':asdict(cfg),'model':asdict(config),'parameters':parameter_counts(model),'optimizer_backend':backend,'feature_hash':identity(feature_manifest) if feature_manifest else None})
        loader=DataLoader(dataset,batch_sampler=UpdateSampler(dataset,cfg,start),collate_fn=collate,num_workers=cfg.workers,pin_memory=device.type=='cuda',**({'persistent_workers':True,'prefetch_factor':2,'multiprocessing_context':'spawn'} if cfg.workers else {}))
        iterator=iter(loader); now=time.monotonic(); last_save=now; base_elapsed=elapsed; step=start; ema=None
        def checkpoint_state():
            return {'format':'osu-v2','stage':cfg.stage,'model_config':asdict(config),'train_config':asdict(cfg),'tokenizer_version':TOKEN_VERSION,
                    'dataset_hash':frozen['hash'],'snapshot_hash':frozen['snapshot_hash'],'feature_hash':identity(feature_manifest) if feature_manifest else None,
                    'step':step,'elapsed_seconds':elapsed,'best_validation':best,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'scaler':scaler.state_dict(),'random':random_state(),'sampler':{'seed':cfg.seed,'next_update':step},'backend':{'optimizer':backend,'precision':cfg.precision,'torch':torch.__version__,'device':str(device)}}
        outcome='complete'
        try:
            while step<cfg.steps and elapsed<cfg.hours*3600:
                if cancelled(): outcome='cancelled'; break
                tick=time.monotonic(); data_tick=tick
                batches=[next(iterator) for _ in range(cfg.effective_batch//cfg.batch_size)]
                data_seconds=time.monotonic()-data_tick
                optimizer.zero_grad(set_to_none=True)
                total_tokens=sum(int((b['labels']!=-100).sum()) for b in batches) if cfg.stage=='mapper' else 0
                def backward_update():
                    stats=torch.zeros(3,device=device); padding=0; slots=0; rollout_seconds=0; batch_losses=[]
                    for batch in batches:
                        if cfg.stage=='mapper' and step>=cfg.rollout_after and cfg.rollout_probability:
                            from .generation import replace_training_history
                            rt=time.monotonic(); batch=replace_training_history(model,dataset,batch,step,cfg,device,dtype); rollout_seconds+=time.monotonic()-rt
                        b=move(batch,device)
                        with torch.autocast(device.type,dtype=dtype,enabled=device.type=='cuda'):
                            if cfg.stage=='audio':
                                loss,aux1,aux2,individual=audio_loss(callable_model,b,return_samples=True)
                                # Keep these on GPU until the complete update succeeds. Failed
                                # AMP retries are never written as duplicate spike records.
                                batch_losses.append((individual,batch))
                                loss=loss/len(batches); stats+=torch.stack([loss.detach(),aux1/len(batches),aux2/len(batches)])
                            else:
                                total,count,plan,boundary=mapper_loss(callable_model,b)
                                loss=total/max(1,total_tokens)+(.1*plan+.05*boundary)/len(batches)
                                stats+=torch.stack([loss.detach(),total.detach()/max(1,total_tokens),(plan.detach()+boundary.detach())/len(batches)])
                                padding+=int((batch['tokens']==0).sum()); slots+=batch['tokens'].numel()
                        scaler.scale(loss).backward()
                    return stats, padding, slots, rollout_seconds, batch_losses
                (stats, padding, slots, rollout_seconds, batch_losses), amp_retries = backward_with_recovery(
                    model, optimizer, scaler, backward_update, progress)
                scaler.step(optimizer); scaler.update(); scheduler.step(); step+=1
                if batch_losses:
                    batches_for_log=[]
                    for individual,batch in batch_losses:
                        values=individual.float().cpu().tolist()
                        hard=[i for i,v in enumerate(values) if v[0]>SPIKE_THRESHOLD]
                        if hard:
                            targets=batch['beats'][hard].numpy()
                            masks=batch['beat_mask'][hard].numpy()
                            batches_for_log.append(([values[i] for i in hard],
                                                    [batch['meta'][i] for i in hard],targets,masks))
                    if batches_for_log:
                        write_spikes(run,step,batches_for_log,progress,total_samples=cfg.effective_batch)
                elapsed=base_elapsed+time.monotonic()-now
                values=stats.detach().cpu().tolist(); ema=values[0] if ema is None else .95*ema+.05*values[0]
                memory=enforce_headroom(device)
                seconds=time.monotonic()-tick
                metrics={'stage':cfg.stage,'step':step,'step_target':cfg.steps,'status':'running','loss':values[0],'loss_ema':ema,
                         'beat_loss' if cfg.stage=='audio' else 'token_loss':values[1], 'auxiliary_loss':values[2], 'elapsed_seconds':elapsed,'hours_target':cfg.hours,
                         'seconds_per_update':seconds,'target_tokens_per_second':total_tokens/max(seconds,1e-8),'padding_fraction':padding/max(1,slots),
                         'amp_retries':amp_retries,'grad_scale':scaler.get_scale(),
                         'data_seconds':data_seconds,'rollout_seconds':rollout_seconds,'lr':scheduler.get_last_lr()[0],
                         'eta_seconds':max(0,min((cfg.steps-step)*seconds,cfg.hours*3600-elapsed)),**memory}
                if frozen.get('storage_mode')=='rolling':
                    visited=step if cfg.stage=='audio' else step*cfg.effective_batch
                    metrics.update(dataset_units=len(dataset),completed_dataset_cycles=visited//len(dataset),
                                   current_cycle_fraction=(visited%len(dataset))/len(dataset),
                                   coverage_unit='recordings' if cfg.stage=='audio' else 'maps')
                if step%cfg.validate_every==0 or step==cfg.steps:
                    result=validate(model,validation,cfg,device,dtype); metrics['validation']=result
                    value=result.get('token_loss',result['loss'])
                    if value<best:
                        best=value; save_checkpoint(run/'best.pt',checkpoint_state())
                atomic_json(run/'status.json',metrics)
                with (run/'metrics.jsonl').open('a',encoding='utf-8') as log: log.write(json.dumps(metrics,allow_nan=False)+'\n')
                if step%10==0 or step==start+1: progress(f"V2 {cfg.stage} step {step}: loss {values[0]:.4f}, {metrics['target_tokens_per_second']:.0f} target tokens/s, reserved {memory['reserved_gib']:.2f} GiB")
                if time.monotonic()-last_save>=cfg.checkpoint_minutes*60 or (cfg.sample_every and step%cfg.sample_every==0):
                    save_checkpoint(run/'last.pt',checkpoint_state()); last_save=time.monotonic()
                if cfg.stage=='mapper' and cfg.sample_every and step%cfg.sample_every==0:
                    from .evaluation import training_sample
                    training_sample(model,validation,run,step,device,dtype,progress,cancelled)
                elapsed=base_elapsed+time.monotonic()-now
            elapsed=base_elapsed+time.monotonic()-now
            save_checkpoint(run/'last.pt',checkpoint_state())
            if not (run/'best.pt').exists():
                result=validate(model,validation,cfg,device,dtype); best=result.get('token_loss',result['loss']); save_checkpoint(run/'best.pt',checkpoint_state())
        except BaseException as exc:
            # A failed/partial update is not a resumable boundary. Keep prior checkpoint.
            atomic_json(run/'status.json',{'status':'failed','stage':cfg.stage,'step':step,'error':str(exc),'checkpoint':str(run/'last.pt') if (run/'last.pt').exists() else None})
            raise
        finally:
            if hasattr(iterator,'_shutdown_workers'): iterator._shutdown_workers()
        result={'status':outcome,'stage':cfg.stage,'step':step,'elapsed_seconds':elapsed,'best_validation':best,'checkpoint':str(run/'last.pt'),
                'completion_reason':'cancelled' if outcome=='cancelled' else 'step_budget' if step>=cfg.steps else 'time_budget','quality_verified':False}
        atomic_json(run/'status.json',result); progress(result)
        return result
