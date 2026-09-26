"""One encoder pass per recording/checkpoint; immutable shared feature caches."""
from __future__ import annotations

import math
import shutil
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch

from .. import audio
from ..data import atomic_json, digest, load_json
from .data import identity, records, save_array
from .model import AudioEncoder, ModelConfig
from .timing import estimate
from .storage import decode_audio

VERSION="encoder-features-v2.1"
STRIDE_MS=audio.FRAME_MS*4


def normalized_window(mel,start,length):
    out=np.full((128,length),np.log(1e-8),np.float32)
    a,b=max(0,start),min(mel.shape[1],start+length)
    if b>a: out[:,a-start:b-start]=mel[:,a:b]
    return (out+5)/7


@torch.inference_mode()
def encode_recording(encoder,mel,dest,device,precision=torch.float16,cancelled=lambda:False):
    encoder.eval(); n=mel.shape[1]
    width=encoder.config.audio_width
    length=math.ceil(encoder.config.audio_ms/audio.FRAME_MS/4)*4
    step=length//2//4*4
    encoded=np.zeros((math.ceil(n/4),width),np.float32); weight=np.zeros(len(encoded),np.float32)
    scores=np.zeros((n,2),np.float32); score_weight=np.zeros(n,np.float32)
    taper=np.maximum(np.hanning(length),.02).astype(np.float32)
    for start in range(0,n,step):
        if cancelled(): raise InterruptedError("Feature caching cancelled")
        x=torch.from_numpy(normalized_window(mel,start,length))[None].to(device)
        with torch.autocast(device.type,dtype=precision,enabled=device.type=="cuda"):
            f,b,_=encoder(x)
        f=f[0].float().cpu().numpy(); b=b[0].float().sigmoid().cpu().numpy()
        a=start//4; count=min(len(f),len(encoded)-a); w=taper[::4][:count]
        encoded[a:a+count]+=f[:count]*w[:,None]; weight[a:a+count]+=w
        count=min(length,n-start); scores[start:start+count]+=b[:count]*taper[:count,None]; score_weight[start:start+count]+=taper[:count]
    encoded/=np.maximum(weight[:,None],1e-8); scores/=np.maximum(score_weight[:,None],1e-8)
    dest=Path(dest); save_array(dest / "encoded.npy",encoded.astype(np.float16)); save_array(dest / "beats.npy",scores.astype(np.float16))
    points,report=estimate(scores,strict=False)
    atomic_json(dest / "timing.json",{"points":[vars(p) for p in points],"report":report})
    return encoded,scores


def load_encoder(path):
    state=torch.load(path,map_location="cpu",weights_only=False)
    if state.get("format")!="osu-v2" or state.get("stage")!="audio": raise ValueError("Expected a locally trained V2 audio checkpoint")
    encoder=AudioEncoder(ModelConfig(**state["model_config"])); encoder.load_state_dict(state["model"])
    metadata={key:state[key] for key in ("dataset_hash","model_config","step") if key in state}
    return encoder,metadata


def build_cache(data_root,checkpoint,device="cpu",max_gib=8.0,min_free_gib=8.0,progress=print,cancelled=lambda:False):
    from .runtime import device_for, gpu_session
    manifest=load_json(Path(data_root)/"manifest.json")
    encoder,state=load_encoder(checkpoint)
    if state["dataset_hash"]!=manifest["hash"]: raise ValueError("Audio checkpoint belongs to a different dataset")
    eid=digest(checkpoint); root=Path(manifest["root"])
    config_key=identity([eid,audio.VERSION,VERSION,state["model_config"]])
    destination=root/"features"/config_key
    actual=device_for(device)
    unique={}
    for r in records(manifest):
        prior=unique.get(r["audio_hash"])
        if prior is None or r["sample_weight"]>prior["sample_weight"]: unique[r["audio_hash"]]=r
    if max_gib<=0 or min_free_gib<1: raise ValueError("Feature cache needs positive size and at least 1 GiB disk reserve")
    if manifest.get('storage_mode')=='rolling':
        destination=destination/'rolling'
        destination.mkdir(parents=True,exist_ok=True)
        result={'version':VERSION,'storage_mode':'rolling','dataset_hash':manifest['hash'],
                'encoder_hash':eid,'encoder_checkpoint':str(Path(checkpoint).resolve()),'model_config':state['model_config'],
                'root':str(destination),'frame_ms':STRIDE_MS,'recordings':len(unique),
                'available_recordings':len(unique),'budget_gib':max_gib,'min_free_gib':min_free_gib}
        atomic_json(destination/'manifest.json',result); atomic_json(root/'features.json',result)
        progress(f'Rolling feature cache ready: all {len(unique)} recordings eligible, {max_gib:g} GiB cache. Missing features are computed on CPU when requested.')
        return result
    capacity=int(max_gib*1024**3)
    available=shutil.disk_usage(root).free-int(min_free_gib*1024**3)
    if available<=0: raise RuntimeError("Insufficient free disk space for the requested reserve")
    capacity=min(capacity,available)
    # Preserve song-split and difficulty coverage when disk allows only a cohort.
    by_split={}
    for split in ("train","validation","test"):
        bands=defaultdict(list)
        for row in unique.values():
            if row["split"]==split: bands[int(row["stars"])].append(row)
        ordered=[sorted(bands[b],key=lambda r:(-r["sample_weight"],r["audio_hash"])) for b in sorted(bands)]
        by_split[split]=[r for group in zip_longest(*ordered) for r in group if r is not None] if ordered else []
    selected=[]; estimated=0; chosen=set()
    def cost(row):
        frames=math.ceil(row["duration_ms"]/STRIDE_MS)
        return frames*encoder.config.audio_width*2+math.ceil(row["duration_ms"]/audio.FRAME_MS)*4+32768
    for split,share in (("validation",.05),("test",.05),("train",.9)):
        spent=0
        for row in by_split[split]:
            amount=cost(row)
            if spent+amount<=capacity*share:
                selected.append(row); chosen.add(row["audio_hash"]); estimated+=amount; spent+=amount
    for row in (*by_split["train"],*by_split["validation"],*by_split["test"]):
        amount=cost(row)
        if row["audio_hash"] not in chosen and estimated+amount<=capacity:
            selected.append(row); chosen.add(row["audio_hash"]); estimated+=amount
    if not selected or not any(r["split"]=="train" for r in selected):
        raise RuntimeError("Feature budget is too small for training examples")
    progress(f"Disk-limited feature cohort: {len(selected)}/{len(unique)} recordings, estimated {estimated/1024**3:.2f} GiB")
    with gpu_session(actual,root.parent):
        encoder.to(actual)
        for i,row in enumerate(selected):
            if cancelled(): raise InterruptedError("Feature caching cancelled")
            ah=row["audio_hash"]
            dest=destination/ah
            if not (dest/"ready.json").exists():
                mel_path=root/"audio"/ah/"mel.npy"
                mel=np.load(mel_path,mmap_mode="r") if mel_path.is_file() else audio.spectrogram(decode_audio(row["audio_path"]))
                encode_recording(encoder,mel,dest,actual,cancelled=cancelled)
                atomic_json(dest/"ready.json",{"audio_hash":ah,"encoder_hash":eid,"frame_ms":STRIDE_MS,"audio_frames":mel.shape[1]})
            if shutil.disk_usage(root).free<int(min_free_gib*1024**3):
                raise RuntimeError("Feature cache reached the minimum free-disk reserve; completed recordings are reusable")
            progress(f"Encoded {i+1}/{len(selected)} selected recordings")
    result={"version":VERSION,"dataset_hash":manifest["hash"],"encoder_hash":eid,"encoder_checkpoint":str(Path(checkpoint).resolve()),"model_config":state["model_config"],"root":str(destination),"frame_ms":STRIDE_MS,"recordings":len(selected),
            "selected_audio":sorted(r["audio_hash"] for r in selected),"available_recordings":len(unique),"budget_gib":max_gib,"min_free_gib":min_free_gib,"estimated_gib":estimated/1024**3}
    atomic_json(destination/"manifest.json",result)
    atomic_json(root/"features.json",result)
    return result
