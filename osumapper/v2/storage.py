"""Read selected map/audio members directly from .osz, without extraction."""
from __future__ import annotations

import hashlib
import os
import posixpath
import sqlite3
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np

from .. import audio, mapio


def make_ref(archive, member):
    return "osz://" + quote(str(Path(archive).resolve()), safe="") + "#" + quote(member, safe="")


def parts(ref):
    if not str(ref).startswith("osz://"):
        return None
    archive, sep, member = str(ref)[6:].partition("#")
    if not sep:
        raise ValueError("Incomplete .osz member reference")
    return Path(unquote(archive)), unquote(member)


def read_bytes(ref, maximum=200 * 1024**2):
    source = parts(ref)
    if source is None:
        path = Path(ref)
        if path.stat().st_size > maximum:
            raise ValueError("Source file exceeds size limit")
        return path.read_bytes()
    archive, member = source
    with zipfile.ZipFile(archive) as package:
        info = package.getinfo(member)
        if not 0 < info.file_size <= maximum:
            raise ValueError("Archive member exceeds size limit")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("Archive symlink is not accepted")
        return package.read(info)


def decode_audio(ref, sr=audio.SR):
    source = parts(ref)
    if source is None:
        return audio.decode(ref, sr=sr)
    raw = read_bytes(ref)
    args = [audio.ffmpeg(), "-nostdin", "-v", "error", "-i", "pipe:0", "-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    result = subprocess.run(args, input=raw, capture_output=True, timeout=600,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise ValueError("Audio decode failed: " + result.stderr.decode(errors="replace")[-500:])
    pcm = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not len(pcm) or not np.isfinite(pcm).all():
        raise ValueError("Empty or non-finite archive audio")
    return pcm


def parse_map(ref):
    if parts(ref) is None:
        return mapio.read(ref)
    return mapio.parse(read_bytes(ref, 8 * 1024**2).decode("utf-8-sig"))


def local_file(ref,cache_dir):
    """Materialize only one requested archive member for export/playback."""
    source=parts(ref)
    if source is None:
        return str(Path(ref).resolve())
    cache_dir=Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True)
    suffix=Path(source[1]).suffix.lower()
    name=hashlib.sha256(ref.encode()).hexdigest()+suffix
    path=cache_dir/name
    if not path.is_file():
        tmp=path.with_suffix(path.suffix+".tmp")
        tmp.write_bytes(read_bytes(ref))
        tmp.replace(path)
    return str(path)


def referenced_audio(osu_ref, filename):
    source = parts(osu_ref)
    if source is None:
        return str((Path(osu_ref).parent / filename.replace("\\", "/")).resolve())
    archive, member = source
    relative = posixpath.normpath(posixpath.join(posixpath.dirname(member), filename.replace("\\", "/")))
    if relative.startswith("../") or relative.startswith("/"):
        raise ValueError("AudioFilename escapes archive")
    with zipfile.ZipFile(archive) as package:
        matches = {name.replace("\\", "/").casefold(): name for name in package.namelist()}
    actual = matches.get(relative.casefold())
    if actual is None:
        raise ValueError("AudioFilename missing from archive")
    return make_ref(archive, actual)


def index_archives(source_db, files_root, destination, progress=print, cancelled=lambda: False):
    """Incremental sidecar index. Original SQLite and archives stay untouched."""
    from .data import readonly

    root = Path(files_root).resolve()
    archives = root / "_osz"
    index = Path(destination) / "archive-index.sqlite"
    index.parent.mkdir(parents=True, exist_ok=True)
    out = sqlite3.connect(index)
    out.execute("CREATE TABLE IF NOT EXISTS sources(beatmap_id INTEGER PRIMARY KEY, osu_ref TEXT NOT NULL, audio_ref TEXT NOT NULL, archive TEXT NOT NULL, actual_md5 TEXT NOT NULL)")
    out.execute("CREATE TABLE IF NOT EXISTS scanned(archive TEXT PRIMARY KEY, size INTEGER, mtime_ns INTEGER, matched INTEGER)")
    source = readonly(source_db)
    try:
        expected = {}
        for sid, bid, checksum in source.execute("SELECT d.set_id,d.beatmap_id,b.checksum FROM dataset d JOIN beatmaps b ON b.id=d.beatmap_id"):
            if checksum:
                expected.setdefault(int(sid), {})[checksum.lower()] = int(bid)
        paths = sorted(archives.glob("*.osz")) if archives.is_dir() else []
        matched = 0
        for number, path in enumerate(paths, 1):
            if cancelled():
                raise InterruptedError("Archive indexing cancelled; completed entries remain")
            try:
                sid = int(path.name.split(".")[0])
            except ValueError:
                continue
            targets = expected.get(sid)
            if not targets:
                continue
            stat = path.stat()
            old = out.execute("SELECT size,mtime_ns FROM scanned WHERE archive=?", (str(path),)).fetchone()
            if old == (stat.st_size, stat.st_mtime_ns):
                continue
            this = 0
            try:
                out.execute("SAVEPOINT one_archive")
                out.execute("DELETE FROM sources WHERE archive=?", (str(path),))
                with zipfile.ZipFile(path) as package:
                    if sum(i.file_size for i in package.infolist()) > 2 * 1024**3:
                        raise ValueError("Archive decompression limit exceeded")
                    names = {n.replace("\\", "/").casefold(): n for n in package.namelist()}
                    for member in package.infolist():
                        if not member.filename.lower().endswith(".osu") or member.file_size > 8 * 1024**2:
                            continue
                        raw = package.read(member)
                        md5 = hashlib.md5(raw).hexdigest()
                        bid = targets.get(md5)
                        if bid is None:
                            continue
                        bm = mapio.parse(raw.decode("utf-8-sig"))
                        relative = posixpath.normpath(posixpath.join(posixpath.dirname(member.filename.replace("\\", "/")),
                                                                   bm.general["AudioFilename"].replace("\\", "/")))
                        if relative.startswith("../") or relative.startswith("/"):
                            continue
                        actual = names.get(relative.casefold())
                        if actual is None:
                            continue
                        audio_info = package.getinfo(actual)
                        if not 0 < audio_info.file_size <= 200 * 1024**2:
                            continue
                        out.execute("INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?)",
                                    (bid, make_ref(path, member.filename), make_ref(path, actual), str(path), md5))
                        this += 1
                out.execute("INSERT OR REPLACE INTO scanned VALUES(?,?,?,?)", (str(path), stat.st_size, stat.st_mtime_ns, this))
                out.execute("RELEASE one_archive")
                out.commit()
                matched += this
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                out.execute("ROLLBACK TO one_archive")
                out.execute("RELEASE one_archive")
                progress(f"Archive {path.name} skipped: {exc}")
            if number % 100 == 0:
                progress(f"Archive index {number}/{len(paths)}: +{matched} matching selected maps")
        return index
    finally:
        source.close()
        out.close()
