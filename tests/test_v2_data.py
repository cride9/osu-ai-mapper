import hashlib
import sqlite3
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from osumapper import audio,mapio
from osumapper.data import load_json,set_label
from osumapper.v2.data import prepare,records,snapshot,assign_groups
from osumapper.v2.source_files import process_osz,sync_dataset,pending_sets,completion
from osumapper.v2.dataset import AudioDataset
from osumapper.v2.storage import decode_audio,parts


def database(path):
    db=sqlite3.connect(path); db.row_factory=sqlite3.Row
    db.executescript('''CREATE TABLE beatmaps(id INTEGER PRIMARY KEY,set_id INTEGER,checksum TEXT);
    CREATE TABLE files(beatmap_id INTEGER PRIMARY KEY,osu_path TEXT,audio_path TEXT,audio_sha256 TEXT,downloaded_at TEXT);
    CREATE TABLE dataset(beatmap_id INTEGER PRIMARY KEY,set_id INTEGER,osu_path TEXT,audio_path TEXT,ready INTEGER,sample_weight REAL,artist TEXT,title TEXT,tags_json TEXT);
    CREATE TABLE set_downloads(set_id INTEGER PRIMARY KEY,state TEXT,attempts INTEGER);''')
    return db


def make_assets(tmp_path):
    root=tmp_path/'files'; folder=root/'55'; (folder/'音楽').mkdir(parents=True)
    waveform=np.zeros(audio.SR*4,np.float32); waveform[audio.SR:]=.1*np.sin(2*np.pi*440*np.arange(audio.SR*3)/audio.SR)
    ap=folder/'音楽'/'árvíz.wav'; sf.write(ap,waveform,audio.SR)
    bm=mapio.Beatmap(timing=[mapio.TimingPoint(0,500)],objects=[mapio.HitObject(100+i*40,180,1000+i*500) for i in range(5)])
    bm.general['AudioFilename']='音楽/árvíz.wav'; bm.metadata.update(Title='Árvíz',Artist='音楽')
    op=folder/'1.osu'; mapio.write(bm,op)
    return root,op,ap,bm


def test_selected_database_import_and_frozen_labels(tmp_path):
    root,op,ap,bm=make_assets(tmp_path); dbpath=tmp_path/'source.sqlite'; db=database(dbpath)
    md5=hashlib.md5(op.read_bytes()).hexdigest()
    db.execute('INSERT INTO beatmaps VALUES(1,55,?)',(md5,))
    db.execute('INSERT INTO dataset VALUES(1,55,?,?,1,1,?,?,?)',(str(op),str(ap),'音楽','Árvíz','[]'))
    db.commit(); db.close()
    mapio.write(bm,root/'unselected.osu')
    before=dbpath.read_bytes(); dest=tmp_path/'v2'
    report=prepare(dbpath,root,dest,progress=lambda _:None)
    assert report['usable_maps']==1 and report['unique_audio']==1
    assert before==dbpath.read_bytes()
    manifest=load_json(dest/'manifest.json'); row=list(records(manifest))[0]
    assert row['styles']=={'aim':None,'streams':None,'rhythm':None}
    run=tmp_path/'run'; frozen=snapshot(dest,run)
    set_label(dest,row['id'],{'aim':2,'streams':0,'rhythm':None})
    assert snapshot(dest,tmp_path/'run2')['snapshot_hash']!=frozen['snapshot_hash']
    assert load_json(run/'dataset.json')['labels']=={}
    prepare(dbpath,root,dest,progress=lambda _:None)
    assert load_json(dest/'manifest.json')['hash']==manifest['hash']


def test_missing_final_selection_and_checksum_mismatch(tmp_path):
    db=sqlite3.connect(tmp_path/'empty.sqlite'); db.execute('CREATE TABLE meta (name TEXT)'); db.close()
    with pytest.raises(ValueError,match='dataset'):
        prepare(tmp_path/'empty.sqlite',tmp_path,tmp_path/'dest')
    root,op,ap,_=make_assets(tmp_path); db=database(tmp_path/'wrong.sqlite')
    db.execute('INSERT INTO beatmaps VALUES(1,55,?)',('0'*32,))
    db.execute('INSERT INTO dataset VALUES(1,55,?,?,1,1,?,?,?)',(str(op),str(ap),'A','B','[]')); db.commit(); db.close()
    with pytest.raises(ValueError,match='No usable'):
        prepare(tmp_path/'wrong.sqlite',root,tmp_path/'bad',progress=lambda _:None)
    assert 'checksum' in load_json(tmp_path/'bad'/'rejected.json')[0]['reason']


def test_duplicate_groups_transitive_and_reproducible():
    rows=[dict(id='a',audio_hash='a',fingerprint='f1',song_key='x',set_id=1),dict(id='b',audio_hash='b',fingerprint='f1',song_key='y',set_id=2),dict(id='c',audio_hash='c',fingerprint='f2',song_key='z',set_id=2)]
    assign_groups(rows); assert len({r['group'] for r in rows})==1
    reverse=[dict(r) for r in reversed(rows)]; assign_groups(reverse)
    assert {r['id']:r['split'] for r in rows}=={r['id']:r['split'] for r in reverse}


def test_extraction_preserves_relative_audio_and_partial_done_retries(tmp_path):
    root,op,ap,bm=make_assets(tmp_path)
    db=database(tmp_path/'files.sqlite')
    db.execute('INSERT INTO beatmaps VALUES(1,55,?)',(hashlib.md5(op.read_bytes()).hexdigest(),))
    db.execute('INSERT INTO beatmaps VALUES(2,55,?)',('1'*32,))
    for bid in (1,2): db.execute('INSERT INTO dataset VALUES(?,55,NULL,NULL,0,1,?,?,?)',(bid,'A','B','[]'))
    db.execute("INSERT INTO set_downloads VALUES(55,'done',0)"); db.commit()
    archive=tmp_path/'archive.osz'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('chart.osu',op.read_bytes()); z.writestr('音楽/árvíz.wav',ap.read_bytes())
    ids,unmatched=process_osz(db,archive,tmp_path/'out')
    assert ids==[1]
    row=db.execute('SELECT * FROM files WHERE beatmap_id=1').fetchone()
    mapped=mapio.read(row['osu_path'])
    assert (Path(row['osu_path']).parent/mapped.general['AudioFilename']).resolve()==Path(row['audio_path'])
    assert completion(db,55)==(1,2)
    assert tuple(sync_dataset(db))==(1,2)
    pending=pending_sets(db,SimpleNamespace(all_sets=False,max_attempts=3,limit=0))
    assert len(pending)==1 and pending[0]['set_id']==55
    Path(row['audio_path']).unlink()
    assert tuple(sync_dataset(db))==(0,2)
    with pytest.raises(ValueError,match='exact'):
        process_osz(db,archive,tmp_path/'out',allow_updated=True)
    db.close()


def test_archive_selected_import_without_extraction_or_mel_copy(tmp_path):
    source,op,ap,_=make_assets(tmp_path)
    root=tmp_path/'archive-only'; (root/'_osz').mkdir(parents=True)
    with zipfile.ZipFile(root/'_osz'/'55.osz','w') as z:
        z.writestr('chart.osu',op.read_bytes())
        z.writestr('音楽/árvíz.wav',ap.read_bytes())
    dbpath=tmp_path/'selected.sqlite'; db=database(dbpath)
    db.execute('INSERT INTO beatmaps VALUES(1,55,?)',(hashlib.md5(op.read_bytes()).hexdigest(),))
    db.execute('INSERT INTO dataset VALUES(1,55,NULL,NULL,0,1,?,?,?)',('音楽','Árvíz','[]'))
    db.commit();db.close()
    output=tmp_path/'prepared'
    report=prepare(dbpath,root,output,progress=lambda _:None)
    assert report['usable_maps']==1
    row=list(records(load_json(output/'manifest.json')))[0]
    assert parts(row['map_path']) is not None and parts(row['audio_path']) is not None
    assert np.allclose(decode_audio(row['audio_path'])[:100],audio.decode(ap)[:100],atol=1e-4)
    assert not list((output/'audio'/row['audio_hash']).glob('mel.npy'))
    assert AudioDataset(snapshot(output,tmp_path/'run'))[0]['mel'].shape[0]==128
    assert not (root/'55').exists()


def test_metadata_budget_freezes_a_reported_subset(tmp_path):
    root,op,ap,bm=make_assets(tmp_path)
    second=op.with_name('2.osu')
    bm.metadata['Version']='Another difficulty'; mapio.write(bm,second)
    dbpath=tmp_path/'budget.sqlite'; db=database(dbpath)
    for bid,path in ((1,op),(2,second)):
        db.execute('INSERT INTO beatmaps VALUES(?,?,?)',(bid,55,hashlib.md5(path.read_bytes()).hexdigest()))
        db.execute('INSERT INTO dataset VALUES(?,?,?, ?,1,1,?,?,?)',(bid,55,str(path),str(ap),'A','B','[]'))
    db.commit();db.close()
    report=prepare(dbpath,root,tmp_path/'bounded',max_metadata_gib=1e-7,progress=lambda _:None,full_dataset=False)
    assert report['metadata_budget_limited']
    assert report['selected_rows']==2 and report['examined_rows']==1
    assert report['unexamined_rows']==1 and report['usable_maps']==1
