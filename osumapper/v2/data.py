"""Read-only script-output import into immutable V2, content-addressed caches."""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .. import audio, mapio
from ..data import atomic_json, digest, difficulty, load_json, normalized, style_metrics
from .style import describe
from .tokenizer import Tokenizer, VERSION as TOKEN_VERSION, STYLES
from .storage import decode_audio, index_archives, parse_map, parts, read_bytes, referenced_audio

VERSION = "dataset-v2.2-streaming"


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def readonly(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def save_array(path, array):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array, allow_pickle=False)
    tmp.replace(path)


def map_from_dict(value):
    return mapio.Beatmap(general=value["general"], metadata=value["metadata"], difficulty=value["difficulty"],
                        timing=[mapio.TimingPoint(**p) for p in value["timing"]],
                        objects=[mapio.HitObject(**o) for o in value["objects"]], breaks=[tuple(b) for b in value["breaks"]])


def read_cached_map(root, row):
    if 'rolling_cache_gib' in row:
        from .rolling import map_bundle,array_json
        return map_from_dict(array_json(map_bundle(root,row)['map_json']))
    return map_from_dict(load_json(Path(root) / "maps" / row["id"] / "map.json"))


def _resolve_file(value, root, set_id, kind):
    candidate = Path(value or "")
    options = [candidate if candidate.is_absolute() else root / candidate, root / str(set_id) / candidate.name]
    for path in options:
        path = path.resolve()
        if path.is_relative_to(root) and path.is_file():
            return path
    raise ValueError(f"Missing {kind} under files-root: {value}")


def _cache_events(bm, dest, duration):
    tok = Tokenizer()
    events = sorted([(o.time, tok.object(bm, o, o.time), o.time + bm.duration(o)) for o in bm.objects]
                    + [(a, [tok.ids["BREAK"], *tok.time(a, a), *tok.integer(b-a), tok.ids["END"]], b) for a,b in bm.breaks], key=lambda e:e[0])
    times, flat, offsets, ends = [], [], [0], []
    for at, tokens, end in events:
        if len(tokens) > 2000:
            raise ValueError("Single event cannot fit the V2 decoder")
        times.append(at); ends.append(end); flat.extend(tokens); offsets.append(len(flat))
    for name, values, dtype in (("tokens", flat, np.int32), ("offsets", offsets, np.int64), ("times", times, np.int64), ("ends", ends, np.float64)):
        save_array(dest / f"{name}.npy", np.asarray(values, dtype=dtype))
    starts = np.arange(0, duration, 2000)
    descriptors = np.stack([describe(bm, at, at+2000) for at in starts])
    save_array(dest / "style.npy", descriptors)
    save_array(dest / "style_times.npy", starts)
    # Reserve condition/end/GEN/EOS, shorten on complete simultaneous-event boundaries.
    windows, start = [], 0
    while start < duration:
        end = min(start + 16000, duration)
        left, right = np.searchsorted(times, [start, end])
        used = 0
        for i in range(left, right):
            size = offsets[i+1]-offsets[i]
            if used + size > 2016:
                end = times[i]
                if end <= start:
                    raise ValueError("Simultaneous events exceed V2 token budget")
                right = np.searchsorted(times, end)
                used = offsets[right]-offsets[left]
                break
            used += size
        windows.append((start, end, used+13))
        start = end
    save_array(dest / "windows.npy", np.asarray(windows, dtype=np.int64))
    atomic_json(dest / "map.json", asdict(bm))
    return len(windows)


def assign_groups(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i, r in enumerate(rows):
        for k in ("audio_hash", "fingerprint", "song_key", "set_id"):
            value = r.get(k)
            if value is None or value == "" or (k == "set_id" and int(value) <= 0):
                continue
            key = (k, str(value))
            if key in seen: parent[find(i)] = find(seen[key])
            else: seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows): groups[find(i)].append(row)
    for records in groups.values():
        gid = identity(sorted({r["audio_hash"] for r in records}))
        bucket = int(gid[:8],16) % 100
        split = "train" if bucket < 90 else "validation" if bucket < 95 else "test"
        for r in records: r.update(group=gid, split=split)


def prepare(source_db, files_root, destination, limit=None, archive_index=True, cache_mel=False, max_metadata_gib=4.0, progress=print, cancelled=lambda:False, full_dataset=True):
    root, files_root = Path(destination).resolve(), Path(files_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    prior=load_json(root/'manifest.json')
    if prior and (prior.get('storage_mode')=='rolling')!=bool(full_dataset):
        raise ValueError('Use a new directory (for example dataset-full) to change storage mode; existing runs keep their frozen dataset')
    if full_dataset and cache_mel:
        raise ValueError('Full-library mode uses a bounded rotating audio cache; omit cache_mel')
    if shutil.disk_usage(root).free < 8*1024**3:
        raise RuntimeError("V2 preparation requires at least 8 GiB free for metadata and checkpoints")
    if (root / "manifest.json").exists() and load_json(root / "manifest.json").get("version") != VERSION:
        raise ValueError("Use a separate V2 dataset directory; V1 data is protected")
    db = readonly(source_db)
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='dataset'").fetchone():
        db.close(); raise ValueError("Missing final 'dataset' table. Run osu_dataset.py build first.")
    # An archive sidecar makes already downloaded .osz usable without expanding them.
    archive_db=None
    if archive_index:
        archive_path=index_archives(source_db,files_root,root,progress,cancelled)
        archive_db=readonly(archive_path)
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    has_stars="stars" in {r[1] for r in db.execute("PRAGMA table_info(dataset)")}
    checksum="b.checksum" if "beatmaps" in tables else "NULL"
    join="LEFT JOIN beatmaps b ON b.id=d.beatmap_id" if "beatmaps" in tables else ""
    if has_stars:
        query=(f"SELECT * FROM (SELECT d.*, {checksum} AS expected_checksum, "
               f"ROW_NUMBER() OVER (PARTITION BY CAST(d.stars AS INT) ORDER BY d.sample_weight DESC,d.beatmap_id) AS sampling_rank, "
               f"CAST(d.stars AS INT) AS sampling_band FROM dataset d {join}) ORDER BY sampling_rank,sampling_band,beatmap_id")
    else:
        query=f"SELECT d.*, {checksum} AS expected_checksum FROM dataset d {join} ORDER BY d.beatmap_id"
    total = db.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
    candidate_bands={str(b):n for b,n in db.execute('SELECT CAST(stars AS INT),COUNT(*) FROM dataset GROUP BY CAST(stars AS INT)')} if has_stars else {}
    accepted, rejected, seen = [], [], set()
    audio_paths, audio_bytes, timing_maps = {}, {}, defaultdict(list)
    if max_metadata_gib<=0: raise ValueError("Metadata cache budget must be positive")
    metadata_limit=int(max_metadata_gib*1024**3*.85)
    cached_bytes=0; budget_limited=False; visited=0
    try:
        for index, raw in enumerate(db.execute(query)):
            if limit and index >= limit: break
            if not full_dataset and cached_bytes>=metadata_limit:
                budget_limited=True; progress(f"Metadata budget reached after {index} selected rows; frozen subset will be built")
                break
            visited=index+1
            if cancelled(): raise InterruptedError("Preparation cancelled; completed immutable caches retained")
            row = dict(raw)
            try:
                archive = archive_db.execute("SELECT osu_ref,audio_ref FROM sources WHERE beatmap_id=?",(row["beatmap_id"],)).fetchone() if archive_db else None
                try:
                    path = _resolve_file(row.get("osu_path"), files_root, row["set_id"], "osu")
                    path = str(path)
                except ValueError:
                    if archive is None: raise
                    path=archive[0]
                content = read_bytes(path,8*1024**2)
                actual_md5 = hashlib.md5(content).hexdigest()
                if row.get("expected_checksum") and actual_md5.lower() != row["expected_checksum"].lower():
                    raise ValueError("Map checksum differs from metadata; refresh the selected version")
                bm = parse_map(path)
                if not bm.objects: raise ValueError("Empty beatmap")
                ap = referenced_audio(path,bm.general["AudioFilename"])
                if parts(ap) is None and not Path(ap).is_relative_to(files_root):
                    raise ValueError("AudioFilename escapes files-root")
                if parts(path) is None and row.get("audio_path"):
                    declared = _resolve_file(row.get("audio_path"), files_root, row["set_id"], "audio")
                    if ap != str(declared) and hashlib.sha256(read_bytes(ap)).digest() != hashlib.sha256(read_bytes(declared)).digest():
                        raise ValueError("DB audio differs from the map's AudioFilename")
                key = str(ap)
                if key not in audio_paths:
                    bh = hashlib.sha256(read_bytes(ap)).hexdigest()
                    info_file = root / "audio" / (bh + ".json")
                    info = audio_bytes.get(bh) or load_json(info_file)
                    if not info or info.get("audio_version") != audio.VERSION or (cache_mel and not (root / "audio" / info["audio_hash"] / "mel.npy").exists()):
                        pcm = decode_audio(ap); ah = audio.pcm_hash(pcm)
                        dest = root / "audio" / ah
                        if cache_mel and not (dest / "mel.npy").exists(): save_array(dest / "mel.npy", audio.spectrogram(pcm).astype(np.float16))
                        info = dict(audio_hash=ah, fingerprint=audio.fingerprint(pcm), duration_ms=len(pcm)*1000/audio.SR, audio_version=audio.VERSION, bytes_sha256=bh)
                        atomic_json(info_file, info)
                    audio_bytes[bh] = info; audio_paths[key] = info
                info = audio_paths[key]
                bm.validate(info["duration_ms"])
                if any(len(o.points)>64 for o in bm.objects): raise ValueError("More than 64 native slider anchors")
                mid = identity([actual_md5, info["audio_hash"], TOKEN_VERSION, VERSION])
                if mid in seen: continue
                dest = root / "maps" / mid
                if full_dataset:
                    from .rolling import map_bundle
                    bundle=map_bundle(root,dict(id=mid,rolling_cache_gib=max_metadata_gib,duration_ms=info['duration_ms']),bm)
                    count=len(bundle['windows'])
                    window_tokens=bundle['windows'][:,2].tolist()
                elif not (dest / "ready.json").exists():
                    count = _cache_events(bm, dest, info["duration_ms"])
                    atomic_json(dest / "ready.json", {"windows":count})
                else: count = load_json(dest / "ready.json")["windows"]
                if not full_dataset: cached_bytes+=sum(p.stat().st_size for p in dest.glob('*') if p.is_file())
                diff = difficulty(bm)
                tags = json.loads(row.get("tags_json") or "[]")
                styles = {k:None for k in STYLES}; provenance = {}
                aliases = {"aim":"aim", "jumps":"aim", "streams":"streams", "stream":"streams", "complex rhythm":"rhythm", "rhythm complexity":"rhythm"}
                for t in tags:
                    label = aliases.get(str(t.get("tag", "")).casefold())
                    if label and t.get("votes",0)>=5:
                        styles[label] = 2; provenance[label] = {"source":"tag", "tag":t["tag"], "votes":t["votes"], "confidence":min(1,t["votes"]/20)}
                record = {"id":mid, "beatmap_id":row["beatmap_id"], "set_id":row["set_id"], "map_path":str(path), "map_hash":hashlib.sha256(content).hexdigest(), "map_md5":actual_md5,
                          "expected_checksum":row.get("expected_checksum"), "audio_path":key, **info, **diff, "windows":count,
                          "song_key":normalized(row.get("artist") or bm.metadata.get("Artist",""))+"|"+normalized(row.get("title") or bm.metadata.get("Title","")),
                          "title":bm.metadata.get("Title",""), "artist":bm.metadata.get("Artist",""), "version":bm.metadata.get("Version",""),
                          "settings":{s:float(bm.difficulty.get(k,5)) for s,k in [("AR","ApproachRate"),("OD","OverallDifficulty"),("CS","CircleSize"),("HP","HPDrainRate")]},
                          "styles":styles, "label_provenance":provenance, "metrics":style_metrics(bm,diff), "tags":tags,
                          "timing":[asdict(p) for p in bm.timing], "slider_multiplier":float(bm.difficulty.get("SliderMultiplier",1.4)),
                          "sample_weight":float(np.clip(row.get("sample_weight") or 1,.2,1)), "source_metadata":row}
                if full_dataset:
                    record.update(rolling_cache_gib=max_metadata_gib,window_tokens=window_tokens,
                                  first_object=bm.objects[0].time,last_object=max(o.time+bm.duration(o) for o in bm.objects))
                accepted.append(record); seen.add(mid); timing_maps[info["audio_hash"]].append(mid)
            except (ValueError,OSError,KeyError,IndexError,OverflowError,zipfile.BadZipFile,subprocess.TimeoutExpired) as exc:
                rejected.append({"beatmap_id":row.get("beatmap_id"), "reason":str(exc)})
            if index % 25 == 0:
                if shutil.disk_usage(root).free < 8*1024**3:
                    raise RuntimeError("Free disk space fell below the 8 GiB V2 preparation reserve")
                progress(f"Import {index+1}/{total}: {len(accepted)} usable, {len(rejected)} rejected")
                atomic_json(root / "prepare-progress.json", dict(processed=index+1,total=total,accepted=len(accepted),rejected=len(rejected)))
    finally:
        db.close()
        if archive_db: archive_db.close()
    atomic_json(root / "rejected.json", rejected)
    if not accepted: raise ValueError("No usable V2 maps; inspect rejected.json")
    assign_groups(accepted)
    # Train-only suggestion thresholds; held-out maps never tune the labels.
    for band in {int(r["stars"]) for r in accepted}:
        training = [r for r in accepted if r["split"]=="train" and int(r["stars"])==band]
        if len(training)<3: continue
        thresholds = {k:np.quantile([r["metrics"][k] for r in training],[1/3,2/3]) for k in STYLES}
        for r in accepted:
            if int(r["stars"])!=band: continue
            for k in STYLES:
                if r["styles"][k] is None:
                    r["styles"][k]=int(sum(r["metrics"][k]>thresholds[k]))
                    r["label_provenance"][k]={"source":"train-band-statistics", "confidence":.5}
    audio_rows = defaultdict(list)
    for r in accepted: audio_rows[r["audio_hash"]].append(r)
    label_specs={}
    for ah, mids in timing_maps.items():
        if cancelled(): raise InterruptedError("Preparation cancelled")
        if full_dataset:
            grids={}
            for r in audio_rows[ah]:
                red=[p for p in r['timing'] if p['uninherited']]
                signature=tuple((p['time'],p['beat_length'],p['meter']) for p in red)
                grids[signature]=red
            label_specs[ah]=dict(timings=list(grids.values()),first=min(r['first_object'] for r in audio_rows[ah]),last=max(r['last_object'] for r in audio_rows[ah]))
            for r in audio_rows[ah]: r['timing_labels']=identity(sorted(mids))
            continue
        dest = root / "audio" / ah
        frames = int(np.floor(audio_rows[ah][0]["duration_ms"] / audio.FRAME_MS + 1e-7)) + 1
        low, high, target, signatures = None, None, None, set()
        first, last, count = float("inf"), 0, 0
        for mid in mids:
            bm = map_from_dict(load_json(root / "maps" / mid / "map.json"))
            signature = tuple((p.time,p.beat_length,p.meter) for p in bm.timing if p.uninherited)
            first=min(first,bm.objects[0].time); last=max(last,max(o.time+bm.duration(o) for o in bm.objects))
            if signature in signatures: continue
            signatures.add(signature)
            arr=audio.beat_targets(bm,frames)
            low=arr.copy() if low is None else np.minimum(low,arr)
            high=arr.copy() if high is None else np.maximum(high,arr)
            target=arr.copy() if target is None else target+arr
            count+=1
        mask=(high-low<.35) & ((np.arange(frames)*audio.FRAME_MS>=max(0,first-4000)) & (np.arange(frames)*audio.FRAME_MS<=last+2000))[:,None]
        # Supervision is versioned by the precise set of source map grids.
        label_id=identity(sorted(mids)); label_dir=dest / label_id
        save_array(label_dir / "beats.npy", (target/count).astype(np.float16)); save_array(label_dir / "mask.npy", mask.astype(np.uint8))
        for r in audio_rows[ah]: r["timing_labels"]=label_id
    report={"selected_rows":total,"examined_rows":visited,"unexamined_rows":max(0,total-visited),"metadata_budget_limited":budget_limited,
            "metadata_budget_gib":max_metadata_gib,"indexed_map_cache_gib":cached_bytes/1024**3,
            "usable_maps":len(accepted),"unique_audio":len(timing_maps),"song_groups":len({r['group'] for r in accepted}),
            "audio_hours":sum(rs[0]["duration_ms"] for rs in audio_rows.values())/3600000,
            "splits":dict(Counter(r["split"] for r in accepted)),"star_bands":dict(Counter(str(int(r["stars"])) for r in accepted)),"rejected":len(rejected),
            "source_modes":dict(Counter("osz" if parts(r["map_path"]) else "extracted" for r in accepted)),"mel_cache_persisted":cache_mel}
    report.update(candidate_star_bands=candidate_bands,storage_mode='rolling' if full_dataset else 'fixed',
                  full_selection_examined=visited==total,
                  selection_note='All usable selected rows remain eligible; cache eviction does not exclude maps' if full_dataset else 'Fixed disk-limited subset')
    imported_source_bands=Counter(str(int(r['source_metadata']['stars'])) for r in accepted if r['source_metadata'].get('stars') is not None)
    report['coverage_by_source_star_band']={band:{'candidates':count,'usable':imported_source_bands[band],
                                               'usable_fraction':imported_source_bands[band]/count} for band,count in candidate_bands.items()}
    if full_dataset: report['indexed_map_cache_gib']=sum(p.stat().st_size for p in (root/'rolling').glob('*.sqlite'))/1024**3
    descriptor={"version":VERSION,"audio_version":audio.VERSION,"tokenizer_version":TOKEN_VERSION,"source_db":str(Path(source_db).resolve()),"records":accepted}
    if full_dataset: descriptor.update(storage_mode='rolling',rolling_cache_gib=max_metadata_gib)
    dataset_id=identity(descriptor)
    version_dir=root / "versions" / dataset_id; version_dir.mkdir(parents=True,exist_ok=True)
    index_path=version_dir / "index.sqlite"
    if not index_path.exists():
        (version_dir / "index.tmp.sqlite").unlink(missing_ok=True)
        out=sqlite3.connect(version_dir / "index.tmp.sqlite")
        out.execute("CREATE TABLE records (id TEXT PRIMARY KEY, split TEXT, group_id TEXT, stars REAL, windows INTEGER, payload TEXT)")
        out.executemany("INSERT INTO records VALUES(?,?,?,?,?,?)",[(r['id'],r['split'],r['group'],r['stars'],r['windows'],json.dumps(r,ensure_ascii=False)) for r in accepted])
        if full_dataset:
            out.execute("CREATE INDEX audio_lookup ON records(json_extract(payload,'$.audio_hash'))")
            out.execute('CREATE TABLE audio_labels(audio_hash TEXT PRIMARY KEY,payload TEXT)')
            out.executemany('INSERT INTO audio_labels VALUES(?,?)',((ah,json.dumps(spec)) for ah,spec in label_specs.items()))
        out.commit(); out.close(); (version_dir / "index.tmp.sqlite").replace(index_path)
    manifest={k:v for k,v in descriptor.items() if k!="records"}
    manifest.update(hash=dataset_id,index=str(index_path),index_hash=digest(index_path),root=str(root),report=report)
    atomic_json(version_dir / "manifest.json",manifest); atomic_json(root / "manifest.json",manifest); atomic_json(root / "report.json",report)
    progress(report)
    return report


def records(manifest, split=None):
    db=readonly(manifest["index"])
    try:
        for row in db.execute("SELECT payload FROM records" + (" WHERE split=?" if split else "") + " ORDER BY id",(split,) if split else ()):
            yield json.loads(row[0])
    finally: db.close()


def snapshot(data_root, run_dir):
    manifest=load_json(Path(data_root) / "manifest.json")
    if not manifest or manifest.get("version")!=VERSION: raise ValueError("Prepare a V2 dataset first")
    if digest(manifest["index"])!=manifest["index_hash"]: raise ValueError("Prepared dataset index was modified")
    frozen=dict(manifest, labels=load_json(Path(data_root) / "labels.json",{}))
    frozen["snapshot_hash"]=identity(frozen)
    atomic_json(Path(run_dir) / "dataset.json",frozen)
    return frozen


def resolved(row, frozen):
    label=frozen.get("labels",{}).get(row["id"])
    if label: row={**row,"styles":label["styles"],"excluded":label.get("excluded",False),"label_source":"human"}
    return row
