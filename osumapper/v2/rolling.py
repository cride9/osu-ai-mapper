"""Bounded, process-safe derived caches. Eviction never changes dataset membership."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np


class RollingCache:
    def __init__(self, path, max_gib, min_free_gib=8):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.capacity = int(max_gib * 1024**3)
        self.reserve = int(min_free_gib * 1024**3)
        if self.capacity <= 0: raise ValueError('Cache budget must be positive')
        with self.connect() as db:
            db.execute('PRAGMA auto_vacuum=FULL')
            db.execute('CREATE TABLE IF NOT EXISTS entries(key TEXT PRIMARY KEY, touched REAL, size INTEGER, payload BLOB)')
            db.execute('CREATE INDEX IF NOT EXISTS lru ON entries(touched)')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=600)
        page_size=db.execute('PRAGMA page_size').fetchone()[0]
        db.execute(f'PRAGMA max_page_count={max(64,int(self.capacity/page_size)+64)}')
        try:
            with db: yield db
        finally: db.close()

    def get_or_create(self, key, build):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM entries WHERE key=?', (key,)).fetchone()
            if row:
                db.execute('UPDATE entries SET touched=? WHERE key=?', (time.time(), key))
        if row:
            with np.load(io.BytesIO(row[0]), allow_pickle=False) as bundle:
                return {k:bundle[k] for k in bundle.files}
        value = build()
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **value)
        payload = buffer.getvalue()
        # Oversized recordings remain usable without expanding the disk cache.
        if len(payload) > self.capacity: return value
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM entries WHERE key=?', (key,)).fetchone(): return value
            used = db.execute('SELECT COALESCE(SUM(size),0) FROM entries').fetchone()[0]
            while used + len(payload) > self.capacity:
                oldest = db.execute('SELECT key,size FROM entries ORDER BY touched LIMIT 1').fetchone()
                if oldest is None: break
                db.execute('DELETE FROM entries WHERE key=?', (oldest[0],))
                used -= oldest[1]
            # SQLite reuses freed pages; allow room for a new payload and journal.
            if shutil.disk_usage(self.path.parent).free < self.reserve + 2 * len(payload):
                return value
            try: db.execute('INSERT INTO entries VALUES(?,?,?,?)', (key,time.time(),len(payload),payload))
            except sqlite3.OperationalError as exc:
                if getattr(exc,'sqlite_errorcode',None)!=sqlite3.SQLITE_FULL: raise
                # Storage is a cache, so a full SQLite page budget must not drop data.
                db.rollback()
        return value


def json_array(value):
    return np.frombuffer(json.dumps(value,ensure_ascii=False).encode('utf-8'),dtype=np.uint8)


def array_json(value):
    return json.loads(value.tobytes().decode('utf-8'))


def map_bundle(root, row, beatmap=None):
    from dataclasses import asdict
    from .data import _cache_events
    from .storage import read_bytes
    from .. import mapio
    root=Path(root)
    cache=RollingCache(root/'rolling'/'maps.sqlite',row['rolling_cache_gib']*.6)
    def build():
        bm=beatmap
        if bm is None:
            raw=read_bytes(row['map_path'],8*1024**2)
            if hashlib.sha256(raw).hexdigest()!=row['map_hash']:
                raise ValueError('Frozen source map changed; prepare a new dataset')
            try: text=raw.decode('utf-8-sig')
            except UnicodeDecodeError: text=raw.decode('cp1252')
            bm=mapio.parse(text)
        with tempfile.TemporaryDirectory(prefix='map-',dir=root/'rolling') as folder:
            dest=Path(folder)
            _cache_events(bm,dest,row['duration_ms'])
            arrays={p.stem:np.load(p,allow_pickle=False) for p in dest.glob('*.npy')}
        arrays['map_json']=json_array(asdict(bm))
        return arrays
    return cache.get_or_create(row['id'],build)


def audio_bundle(frozen, row):
    from .. import audio,mapio
    from .data import identity,readonly
    from .storage import decode_audio
    root=Path(frozen['root'])
    cache=RollingCache(root/'rolling'/'audio.sqlite',frozen['rolling_cache_gib']*.4)
    key=identity([row['audio_hash'],row['timing_labels'],audio.VERSION])
    def build():
        pcm=decode_audio(row['audio_path'])
        if audio.pcm_hash(pcm)!=row['audio_hash']: raise ValueError('Frozen audio changed; prepare again')
        mel=audio.spectrogram(pcm).astype(np.float16)
        db=readonly(frozen['index'])
        try: spec=json.loads(db.execute('SELECT payload FROM audio_labels WHERE audio_hash=?',(row['audio_hash'],)).fetchone()[0])
        finally: db.close()
        low=high=total=None
        for timing in spec['timings']:
            target=audio.beat_targets(mapio.Beatmap(timing=[mapio.TimingPoint(**p) for p in timing]),mel.shape[1])
            low=target.copy() if low is None else np.minimum(low,target)
            high=target.copy() if high is None else np.maximum(high,target)
            total=target.copy() if total is None else total+target
        times=np.arange(mel.shape[1])*audio.FRAME_MS
        mask=(high-low<.35)&((times>=max(0,spec['first']-4000))&(times<=spec['last']+2000))[:,None]
        return dict(mel=mel,beats=(total/len(spec['timings'])).astype(np.float16),mask=mask.astype(np.uint8))
    return cache.get_or_create(key,build)


def feature_bundle(frozen, manifest, row):
    from .data import identity
    from .features import encode_recording
    from ..data import load_json
    import torch
    root=Path(manifest['root'])
    cache=RollingCache(root/'features.sqlite',manifest['budget_gib'],manifest['min_free_gib'])
    key=identity([row['audio_hash'],manifest['encoder_hash'],manifest['version']])
    def build():
        # CPU misses never compete for the mapper's dedicated VRAM.
        encoder=frozen_encoder(manifest['encoder_checkpoint'],manifest['encoder_hash'])
        if frozen.get('storage_mode')=='rolling': mel=audio_bundle(frozen,row)['mel']
        else:
            from .. import audio
            from .storage import decode_audio
            mel=audio.spectrogram(decode_audio(row['audio_path']))
        with tempfile.TemporaryDirectory(prefix='encoder-',dir=root) as folder:
            dest=Path(folder)
            encode_recording(encoder,mel,dest,torch.device('cpu'))
            return dict(encoded=np.load(dest/'encoded.npy'),beats=np.load(dest/'beats.npy'),
                        timing_json=json_array(load_json(dest/'timing.json')))
    return cache.get_or_create(key,build)


@lru_cache(maxsize=1)
def frozen_encoder(path,checksum):
    from ..data import digest
    from .features import load_encoder
    import torch
    if digest(path)!=checksum: raise ValueError('Frozen encoder checkpoint changed')
    # Initializing a CPU encoder on a cache miss must not perturb trainer dropout.
    with torch.random.fork_rng(devices=[]): encoder,_=load_encoder(path)
    return encoder.eval()
