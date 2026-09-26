"""Immutable mmap samples and update-indexed randomness, independent of workers."""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .. import audio, mapio
from ..data import load_json
from .data import readonly, records, resolved
from .features import STRIDE_MS, normalized_window
from .style import reference_code
from .tokenizer import Tokenizer
from .storage import decode_audio

BUCKETS=(256,512,1024,1536,2048)


def bucket(length):
    for b in BUCKETS:
        if length<=b: return b
    raise ValueError("Sequence exceeds V2 maximum length")


def slice_array(a,start,count,fill=0):
    out=np.full((count,*a.shape[1:]),fill,dtype=np.float32)
    left,right=max(0,start),min(len(a),start+count)
    valid=np.zeros(count,bool)
    if right>left: out[left-start:right-start]=a[left:right]; valid[left-start:right-start]=True
    return out,valid


def summary(encoded,max_tokens=512):
    stride=max(round(2000/STRIDE_MS),math.ceil(len(encoded)/max_tokens))
    values=np.stack([np.asarray(encoded[a:a+stride],np.float32).mean(0) for a in range(0,len(encoded),stride)])
    times=np.arange(len(values))*stride*STRIDE_MS
    return values,times


def phase_at(points,times):
    result=np.zeros((len(times),3),np.float32)
    reds=[p for p in points if p.uninherited]
    for i,p in enumerate(reds):
        end=reds[i+1].time if i+1<len(reds) else float("inf")
        use=(times>=(p.time if i else -float("inf"))) & (times<end)
        phase=(times[use]-p.time)/p.beat_length
        result[use]=np.stack([np.sin(phase*2*np.pi),np.cos(phase*2*np.pi),np.cos(phase*2*np.pi/p.meter)],-1)
    return result


class MapDataset(Dataset):
    def __init__(self,frozen,feature_manifest,split="train",seed=42):
        self.frozen,self.feature_manifest,self.split,self.seed=frozen,feature_manifest,split,seed
        self.root=Path(frozen["root"]); self.tok=Tokenizer()
        self.entries=[]; self.groups=defaultdict(list)
        selected=set(feature_manifest.get("selected_audio",[])) if "selected_audio" in feature_manifest else None
        for r in records(frozen,split):
            r=resolved(r,frozen)
            if r.get("excluded"): continue
            if feature_manifest.get('storage_mode')!='rolling' and selected is not None and r["audio_hash"] not in selected: continue
            feature_dir=Path(feature_manifest["root"])/r["audio_hash"]
            if feature_manifest.get('storage_mode')!='rolling' and not (feature_dir/("ready.json" if selected is not None else "encoded.npy")).exists(): continue
            i=len(self.entries)
            self.entries.append((r["id"],r["group"],int(r["stars"]),r["sample_weight"],r["windows"]))
            self.groups[r["group"]].append(i)
        if not self.entries: raise ValueError(f"No {split} maps in frozen V2 dataset")
        self.group_names=sorted(self.groups)
        bands=defaultdict(int)
        for _,_,band,_,_ in self.entries: bands[band]+=1
        self.band_counts=dict(bands)

    def __len__(self): return len(self.entries)

    @lru_cache(maxsize=16)
    def row(self,index):
        db=readonly(self.frozen["index"])
        try: r=json.loads(db.execute("SELECT payload FROM records WHERE id=?",(self.entries[index][0],)).fetchone()[0])
        finally: db.close()
        return resolved(r,self.frozen)

    @lru_cache(maxsize=32)
    def arrays(self,mid):
        if self.frozen.get('storage_mode')=='rolling':
            from .rolling import map_bundle
            db=readonly(self.frozen['index'])
            try: row=json.loads(db.execute('SELECT payload FROM records WHERE id=?',(mid,)).fetchone()[0])
            finally: db.close()
            return map_bundle(self.root,row)
        dest=self.root/"maps"/mid
        return {k:np.load(dest/(k+".npy"),mmap_mode="r") for k in ("tokens","offsets","times","ends","windows","style","style_times")}

    @lru_cache(maxsize=8)
    def encoded(self,ah):
        if self.feature_manifest.get('storage_mode')=='rolling': return self.feature_data(ah)['encoded']
        return np.load(Path(self.feature_manifest["root"])/ah/"encoded.npy",mmap_mode="r")

    @lru_cache(maxsize=2)
    def feature_data(self,ah):
        from .rolling import feature_bundle
        db=readonly(self.frozen['index'])
        try: row=json.loads(db.execute("SELECT payload FROM records WHERE json_extract(payload,'$.audio_hash')=? LIMIT 1",(ah,)).fetchone()[0])
        finally: db.close()
        return feature_bundle(self.frozen,self.feature_manifest,row)

    @lru_cache(maxsize=2)
    def epoch_order(self,epoch):
        # Every map once per cycle, interleaving difficulties across shuffled songs.
        rng=random.Random(f'{self.seed}:all-maps:{epoch}')
        groups=list(self.group_names); rng.shuffle(groups)
        queues=[list(self.groups[g]) for g in groups]
        for queue in queues: rng.shuffle(queue)
        return [queue[i] for i in range(max(map(len,queues))) for queue in queues if i<len(queue)]

    def choose(self,update,count):
        rng=random.Random(f"{self.seed}:selection:{update}")
        selected=[]
        for number in range(count):
            # Equal song opportunity; inverse band-frequency weights within a song.
            if self.frozen.get('storage_mode')=='rolling':
                epoch,position=divmod(update*count+number,len(self.entries))
                index=self.epoch_order(epoch)[position]
            else:
                group=self.groups[rng.choice(self.group_names)]
                weights=[self.entries[i][3]/math.sqrt(self.band_counts[self.entries[i][2]]) for i in group]
                index=rng.choices(group,weights=weights,k=1)[0]
            win=rng.randrange(self.entries[index][4])
            length=self.row(index)['window_tokens'][win] if self.frozen.get('storage_mode')=='rolling' else int(self.arrays(self.entries[index][0])["windows"][win,2])
            selected.append((index,win,update,number,bucket(length)))
        return sorted(selected,key=lambda s:s[-1])

    def _events(self,a,left,right,base,limit=None):
        pieces=[]; size=0
        indices=range(left,right) if limit is None else range(right-1,left-1,-1)
        for i in indices:
            record=np.array(a["tokens"][a["offsets"][i]:a["offsets"][i+1]],copy=True)
            if limit is not None and size+len(record)>limit: break
            record[1:3]=self.tok.time(int(a["times"][i]),base)
            pieces.append(record); size+=len(record)
        if limit is not None: pieces.reverse()
        return np.concatenate(pieces) if pieces else np.array([],np.int64)

    def __getitem__(self,key):
        if isinstance(key,int): key=(key,0,0,0,2048)
        index,window,update,number,_=key
        row=self.row(index); a=self.arrays(row["id"])
        start,end,_=a["windows"][window]; base=int(start)-32000
        rng=random.Random(f"{self.seed}:sample:{update}:{number}:{row['id']}:{window}")
        styles=dict(row["styles"])
        if self.split=="train" and rng.random()<.2: styles={}
        prefix=self.tok.condition(row["stars"],styles,row["settings"])+self.tok.time(int(end)-1,base)+[self.tok.ids["GEN"]]
        left,right=np.searchsorted(a["times"],[start,end])
        target=self._events(a,left,right,base)
        sequence=np.r_[prefix,target,self.tok.ids["EOS"]].astype(np.int64)
        labels=sequence[1:].copy(); labels[:len(prefix)-1]=-100
        hleft=np.searchsorted(a["times"],max(0,base)); history=self._events(a,hleft,left,base,1024)
        if not len(history): history=np.array([self.tok.ids["CTX"]])
        encoded=self.encoded(row["audio_hash"])
        n=math.ceil(64000/STRIDE_MS); origin=round(base/STRIDE_MS)
        local,local_valid=slice_array(encoded,origin,n)
        local_times=np.arange(n)*STRIDE_MS+origin*STRIDE_MS
        points=[mapio.TimingPoint(**p) for p in row["timing"]]
        predicted=False
        if self.split=="train" and update>=1000 and rng.random()<.5:
            if self.feature_manifest.get('storage_mode')=='rolling':
                from .rolling import array_json
                result=array_json(self.feature_data(row['audio_hash'])['timing_json'])
            else: result=load_json(Path(self.feature_manifest["root"])/row["audio_hash"]/"timing.json")
            if result["points"]:
                points=[mapio.TimingPoint(**p) for p in result["points"]]; predicted=True
            else: points=[]
        phase=phase_at(points,local_times)
        glob,global_times=summary(encoded)
        style=a["style"]
        plan=[]
        for j,t in enumerate(global_times):
            left=min(len(style)-1,max(0,int(t//2000)))
            right=min(len(style),max(left+1,int(math.ceil((global_times[j+1] if j+1<len(global_times) else row['duration_ms'])/2000))))
            plan.append(np.asarray(style[left:right],np.float32).mean(0))
        plan=np.stack(plan)
        plan=np.nan_to_num(plan)
        past_limit=np.searchsorted(a["style_times"]+2000,start,side="right")
        past=np.asarray(a["style"][:past_limit: max(1,math.ceil(max(1,past_limit)/128))],np.float32)
        past_times=np.asarray(a["style_times"][:past_limit: max(1,math.ceil(max(1,past_limit)/128))],np.float32)
        if not len(past): past=np.zeros((1,64),np.float32); past_times=np.zeros(1,np.float32)
        active=np.where((a["times"]<start)&(a["ends"]>start))[0]
        busy=max([float(start),*a["ends"][active]])
        last=left-1; x,y=.5,.5
        if last>=0:
            record=a["tokens"][a["offsets"][last]:a["offsets"][last+1]]
            if record[0]!=self.tok.ids["BREAK"]: x=(self.tok.v("XY",record[3])-1024)*2/512; y=(self.tok.v("XY",record[4])-1024)*2/384
        state=np.array([min(1,(busy-start)/64000),x,y,start/max(row['duration_ms'],1),bool(len(active)),predicted,1,0],np.float32)
        result=dict(tokens=sequence[:-1],labels=labels,history=history.astype(np.int64),history_valid=np.ones(len(history),bool),
                    local_audio=local,local_valid=local_valid,local_times=local_times-base,phase=phase,global_audio=glob,global_times=global_times-base,
                    global_valid=np.ones(len(glob),bool),plan=plan,style=reference_code(a['style'],a['style_times'],start,end),past=past,past_times=past_times-base,past_valid=np.ones(len(past),bool),state=state)
        result={k:torch.from_numpy(np.asarray(v,dtype=np.float32) if np.asarray(v).dtype.kind=='f' else np.asarray(v).copy()) for k,v in result.items()}
        result["meta"]={"index":index,"window":window,"start":int(start),"end":int(end),"base":base,"row_id":row['id'],"update":update,"number":number}
        return result


class AudioDataset(Dataset):
    def __init__(self,frozen,split="train",seed=42):
        self.frozen=frozen; self.root=Path(frozen['root']); self.seed=seed; self.split=split
        seen=set(); self.rows=[]; by_audio={}
        for raw in records(frozen,split):
            r=resolved(raw,frozen)
            if r.get('excluded'): continue
            name=' - '.join(part for part in (r.get('artist',''),r.get('title','')) if part)
            version=r.get('version','')
            label=f'{name} [{version}]' if version else name
            if label: by_audio.setdefault(r['audio_hash'],set()).add(label)
            if r['audio_hash'] in seen: continue
            seen.add(r['audio_hash'])
            self.rows.append({k:r[k] for k in ('audio_hash','timing_labels','duration_ms','audio_path')})
            self.rows[-1]['source_map']={k:r.get(k) for k in ('beatmap_id','set_id','artist','title','version','map_path')}
            self.rows[-1]['timing_points']=tuple((float(p['time']),float(p['beat_length']),bool(p['uninherited']))
                                          for p in r.get('timing',()) if p.get('beat_length'))
        for r in self.rows:
            r['maps']=sorted(by_audio.get(r['audio_hash'],()))
        if not self.rows: raise ValueError(f"No {split} audio")
    def __len__(self): return len(self.rows)
    def choose(self,update,count):
        rng=random.Random(f"{self.seed}:audio:{update}")
        selected=self.epoch_order(update//len(self))[update%len(self)] if self.frozen.get('storage_mode')=='rolling' else rng.randrange(len(self))
        return [(selected,update,i) for i in range(count)]

    @lru_cache(maxsize=2)
    def epoch_order(self,epoch):
        order=list(range(len(self))); random.Random(f'{self.seed}:all-audio:{epoch}').shuffle(order)
        return order

    @lru_cache(maxsize=2)
    def audio_data(self,index):
        from .rolling import audio_bundle
        return audio_bundle(self.frozen,self.rows[index])

    @lru_cache(maxsize=2)
    def mel(self,audio_hash,audio_path):
        cached=self.root/'audio'/audio_hash/'mel.npy'
        if cached.is_file(): return np.load(cached,mmap_mode='r')
        return audio.spectrogram(decode_audio(audio_path))
    def __getitem__(self,key):
        index,update,number=key if not isinstance(key,int) else (key,0,0)
        r=self.rows[index]; dest=self.root/'audio'/r['audio_hash']
        if self.frozen.get('storage_mode')=='rolling':
            bundle=self.audio_data(index); mel,target,mask=bundle['mel'],bundle['beats'],bundle['mask']
        else:
            mel=self.mel(r['audio_hash'],r['audio_path']); target=np.load(dest/r['timing_labels']/'beats.npy',mmap_mode='r'); mask=np.load(dest/r['timing_labels']/'mask.npy',mmap_mode='r')
        rng=np.random.default_rng(int.from_bytes(__import__('hashlib').sha256(f"{self.seed}:{index}:{update}:{number}".encode()).digest()[:8],'little'))
        n=math.ceil(64000/audio.FRAME_MS/4)*4
        start=int(rng.integers(0,max(1,mel.shape[1]-n+1))) if self.split=='train' else max(0,(mel.shape[1]-n)//2)
        features=normalized_window(mel,start,n); truth,_=slice_array(target,start,n); weights,valid=slice_array(mask,start,n)
        clean=features.copy(); hidden=np.zeros_like(features,dtype=bool)
        if self.split=='train':
            # Time stretching is aligned for every supervision channel.
            factor=float(rng.uniform(.9,1.1)); positions=np.arange(n)*factor
            features=np.stack([np.interp(positions,np.arange(n),f,left=np.log(1e-8)/7,right=(np.log(1e-8)+5)/7) for f in features]).astype(np.float32)
            truth=np.stack([np.interp(positions,np.arange(n),f,right=0) for f in truth.T],-1).astype(np.float32)
            weights=np.stack([np.interp(positions,np.arange(n),f,right=0) for f in weights.T],-1).astype(np.float32)
            clean=features.copy()
            for _ in range(8):
                a=int(rng.integers(0,n-30)); hidden[:,a:a+30]=True
            features[hidden]=0
        else: hidden[:,::10]=True; features[hidden]=0
        result={k:torch.from_numpy(v) for k,v in dict(mel=features,clean=clean,masked=hidden,beats=truth,beat_mask=weights).items()}
        result['meta']={'audio_hash':r['audio_hash'],'maps':r['maps'],'start_ms':round(start*audio.FRAME_MS),
                        'duration_ms':r['duration_ms'],'audio_path':r['audio_path'],
                        'source_map':r['source_map'],'timing_points':r['timing_points'],
                        'update':update,'sample':number}
        return result


def collate(samples):
    if 'mel' in samples[0]: return {k:[s[k] for s in samples] if k=='meta' else torch.stack([s[k] for s in samples]) for k in samples[0]}
    result={}; lengths={k:max(len(s[k]) for s in samples) for k in samples[0] if k not in ('meta','style','state')}
    lengths['tokens']=lengths['labels']=bucket(lengths['tokens'])
    for key in samples[0]:
        if key=='meta': result[key]=[s[key] for s in samples]; continue
        if key in ('style','state'): result[key]=torch.stack([s[key] for s in samples]); continue
        shape=(len(samples),lengths[key],*samples[0][key].shape[1:])
        out=torch.full(shape,-100 if key=='labels' else 0,dtype=samples[0][key].dtype)
        for i,s in enumerate(samples): out[i,:len(s[key])]=s[key]
        result[key]=out
    result['token_valid']=result['tokens']!=0
    return result
