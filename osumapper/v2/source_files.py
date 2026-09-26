"""Shared verification for the bundled, repaired osu_files.py downloader."""
from __future__ import annotations

import hashlib
import json
import posixpath
import time
import zipfile
from functools import lru_cache
from pathlib import Path,PurePosixPath

from .. import audio,mapio
from .data import identity


def ensure_schema(db):
    columns={r[1] for r in db.execute('PRAGMA table_info(files)')}
    for name,kind in [('verified_key','TEXT'),('valid','INTEGER DEFAULT 0'),('validation_error','TEXT'),('actual_md5','TEXT')]:
        if name not in columns: db.execute(f'ALTER TABLE files ADD COLUMN {name} {kind}')
    db.commit()


def _target(root,relative):
    relative=relative.replace('\\','/')
    if ':' in relative or PurePosixPath(relative).is_absolute(): raise ValueError('Unsafe archive path')
    dest=(root/relative).resolve()
    if not dest.is_relative_to(root.resolve()): raise ValueError('Archive path escapes its beatmapset')
    return dest


@lru_cache(maxsize=256)
def _audio_duration(path,size,mtime):
    pcm=audio.decode(path)
    return len(pcm)*1000/audio.SR


def verify(db,beatmap_id):
    row=db.execute('SELECT f.*, b.checksum FROM files f JOIN beatmaps b ON b.id=f.beatmap_id WHERE f.beatmap_id=?',(beatmap_id,)).fetchone()
    if row is None: return False
    row=dict(row)
    key=None
    try:
        op=Path(row['osu_path'] or ''); ap=Path(row['audio_path'] or '')
        os_,as_=op.stat(),ap.stat()
        key=identity([str(op.resolve()),os_.st_size,os_.st_mtime_ns,str(ap.resolve()),as_.st_size,as_.st_mtime_ns,row['checksum']])
        if row.get('verified_key')==key: return bool(row.get('valid'))
        raw=op.read_bytes(); checksum=hashlib.md5(raw).hexdigest()
        if row['checksum'] and checksum.lower()!=row['checksum'].lower(): raise ValueError('Map version differs from selected metadata')
        bm=mapio.read(op)
        if not bm.objects: raise ValueError('Map has no hit objects')
        referenced=(op.parent/bm.general['AudioFilename'].replace('\\','/')).resolve()
        if referenced!=ap.resolve(): raise ValueError('AudioFilename and database audio_path disagree')
        duration=_audio_duration(str(ap.resolve()),as_.st_size,as_.st_mtime_ns)
        bm.validate(duration)
        db.execute('UPDATE files SET valid=1, verified_key=?, validation_error=NULL, actual_md5=? WHERE beatmap_id=?',(key,checksum,beatmap_id)); db.commit()
        return True
    except (OSError,ValueError,KeyError) as exc:
        db.execute('UPDATE files SET valid=0, verified_key=?, validation_error=? WHERE beatmap_id=?',(key,str(exc),beatmap_id)); db.commit()
        return False


def process_osz(db,osz_path,out_root,allow_updated=False):
    if allow_updated: raise ValueError('V2 requires exact metadata checksums. Refresh metadata instead of --allow-updated.')
    ensure_schema(db); matched=[]; unmatched=0; out_root=Path(out_root).resolve()
    with zipfile.ZipFile(osz_path) as z:
        if sum(i.file_size for i in z.infolist())>2*1024**3: raise ValueError('Archive decompression limit exceeded')
        members={posixpath.normpath(n.replace('\\','/')).casefold():n for n in z.namelist()}
        for member in z.infolist():
            if not member.filename.lower().endswith('.osu'): continue
            if member.file_size>8*1024**2: raise ValueError('Oversized .osu member')
            raw=z.read(member); md5=hashlib.md5(raw).hexdigest()
            row=db.execute('SELECT id,set_id FROM beatmaps WHERE checksum=?',(md5,)).fetchone()
            if row is None: unmatched+=1; continue
            bm=mapio.parse(raw.decode('utf-8-sig',errors='strict'))
            set_root=out_root/str(row['set_id'])
            archive_dir=posixpath.dirname(member.filename.replace('\\','/'))
            osu_path=_target(set_root,posixpath.join(archive_dir,f"{row['id']}.osu"))
            audio_relative=posixpath.normpath(posixpath.join(archive_dir,bm.general['AudioFilename'].replace('\\','/')))
            ap=_target(set_root,audio_relative)
            name=members.get(audio_relative.casefold())
            osu_path.parent.mkdir(parents=True,exist_ok=True); osu_path.write_bytes(raw)
            sha=None
            if name is not None:
                entry=z.getinfo(name)
                if not 0<entry.file_size<=200*1024**2: raise ValueError('Audio member outside size limit')
                if (entry.external_attr>>16)&0o170000==0o120000: raise ValueError('Audio symlink is forbidden')
                content=z.read(name); sha=hashlib.sha256(content).hexdigest()
                ap.parent.mkdir(parents=True,exist_ok=True)
                if not ap.exists() or hashlib.sha256(ap.read_bytes()).hexdigest()!=sha: ap.write_bytes(content)
            db.execute('INSERT OR REPLACE INTO files(beatmap_id,osu_path,audio_path,audio_sha256,downloaded_at,actual_md5) VALUES(?,?,?,?,?,?)',
                       (row['id'],str(osu_path),str(ap) if name is not None else None,sha,time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),md5)); db.commit()
            if verify(db,row['id']): matched.append(row['id'])
    return matched,unmatched


def sync_dataset(db):
    ensure_schema(db)
    for row in db.execute('SELECT beatmap_id FROM files').fetchall(): verify(db,row[0])
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='dataset'").fetchone(): return None
    db.execute('''UPDATE dataset SET
        osu_path=(SELECT osu_path FROM files WHERE beatmap_id=dataset.beatmap_id),
        audio_path=(SELECT audio_path FROM files WHERE beatmap_id=dataset.beatmap_id),
        ready=CASE WHEN EXISTS(SELECT 1 FROM files WHERE beatmap_id=dataset.beatmap_id AND valid=1) THEN 1 ELSE 0 END''')
    db.commit()
    return db.execute('SELECT COALESCE(SUM(ready),0), COUNT(*) FROM dataset').fetchone()


def completion(db,set_id,all_sets=False):
    use_dataset=not all_sets and db.execute("SELECT 1 FROM sqlite_master WHERE name='dataset'").fetchone()
    query='SELECT beatmap_id FROM dataset WHERE set_id=?' if use_dataset else 'SELECT id FROM beatmaps WHERE set_id=?'
    ids=[r[0] for r in db.execute(query,(set_id,))]
    valid=sum(verify(db,i) for i in ids)
    return valid,len(ids)


def pending_sets(db,a):
    sync_dataset(db)
    use_dataset=not a.all_sets and db.execute("SELECT 1 FROM sqlite_master WHERE name='dataset'").fetchone()
    if use_dataset:
        sql='''SELECT d.set_id,COUNT(*) AS expected,MAX(d.sample_weight) AS w FROM dataset d
               LEFT JOIN files f ON f.beatmap_id=d.beatmap_id LEFT JOIN set_downloads sd ON sd.set_id=d.set_id
               WHERE COALESCE(f.valid,0)=0 AND COALESCE(sd.attempts,0)<? AND COALESCE(sd.state,'')!='gone'
               GROUP BY d.set_id ORDER BY w DESC LIMIT ?'''
    else:
        sql='''SELECT b.set_id,COUNT(*) AS expected,1 AS w FROM beatmaps b LEFT JOIN files f ON f.beatmap_id=b.id
               LEFT JOIN set_downloads sd ON sd.set_id=b.set_id WHERE COALESCE(f.valid,0)=0 AND COALESCE(sd.attempts,0)<?
               AND COALESCE(sd.state,'')!='gone' GROUP BY b.set_id ORDER BY b.set_id LIMIT ?'''
    return db.execute(sql,(a.max_attempts,a.limit or -1)).fetchall()
