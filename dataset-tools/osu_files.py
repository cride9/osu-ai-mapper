#!/usr/bin/env python3
"""
osu_files.py - get the actual beatmap files for the SQLite dataset made by osu_dataset.py.

  download   fetch .osz files straight from osu.ppy.sh using your own logged-in session
             cookie, in dataset priority order (works for graveyarded maps too, since it
             hits osu!'s own /download route rather than a third-party mirror)
  index      scan an existing folder of .osz files (e.g. made by your own downloader app)
  stats      progress overview

Getting the cookie (do this in a normal browser, logged into your own osu! account):
  1. Open osu.ppy.sh and make sure you're logged in.
  2. Open DevTools (F12) -> Application/Storage tab -> Cookies -> https://osu.ppy.sh.
  3. Copy the VALUE of the cookie named `osu_session`. Do not copy `XSRF-TOKEN` or any
     other cookie - only `osu_session` is needed here.
  4. Pass it as --cookie "<value>" or set the OSU_SESSION environment variable.
  This is your personal login session, not an API key - treat it like a password:
  don't commit it, don't share it, and it will stop working if you log out or after
  it expires (then just repeat the steps above for a fresh one).

For every .osz it extracts ONLY the .osu files and the audio they reference (never videos,
storyboards, skins), matches each .osu to the DB by MD5 (== the `checksum` the osu! API returned,
so the file is guaranteed to be the version the metadata/tags/star were taken from), and fills
the `files` table. The `dataset` table is updated too (osu_path, audio_path, ready).

Layout:  <out>/<set_id>/<beatmap_id>.osu   and   <out>/<set_id>/<audio file name>

Mirror URL template contains {set_id}, e.g.  --mirror "https://your-mirror.example/d/{set_id}"
Be polite to the mirror: default is 1 request at a time with a 2 s delay. Check its own rules.
"""
from __future__ import annotations

# Allow direct invocation using the project Python, without reinstalling V1.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import os
import re
import sys
import time
import zipfile
from pathlib import Path

from osu_dataset import connect, utcnow

MAX_AUDIO_BYTES = 200 * 1024 * 1024
USER_AGENT = "osu_dataset/1.0 (personal ML research, sequential downloads)"

DDL = """
CREATE TABLE IF NOT EXISTS set_downloads(
    set_id INTEGER PRIMARY KEY,
    state TEXT NOT NULL,                 -- done | gone | error
    attempts INTEGER NOT NULL DEFAULT 0,
    matched INTEGER, expected INTEGER,
    last_error TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS index_seen(path TEXT PRIMARY KEY, size INTEGER, mtime REAL);
"""


# --------------------------------------------------------------------------- extraction
def safe_name(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/"))
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip(" .")
    return base or "audio"


def osu_header(data: bytes):
    text = data.decode("utf-8-sig", errors="replace")[:8000]
    a = re.search(r"^AudioFilename:\s*(.+?)\s*$", text, re.M)
    b = re.search(r"^BeatmapID:\s*(\d+)", text, re.M)
    return (a.group(1) if a else None), (int(b.group(1)) if b else None)


def process_osz(db, osz_path, out_root, allow_updated=False):
    from osumapper.v2.source_files import process_osz as checked
    return checked(db, osz_path, out_root, allow_updated)


def sync_dataset(db):
    from osumapper.v2.source_files import sync_dataset as checked
    return checked(db)


# --------------------------------------------------------------------------- download mode
class Gone(Exception):
    pass


def fetch_osz(session, url, dest: Path, headers, timeout=180):
    import requests
    last = ""
    for attempt in range(4):
        try:
            r = session.get(url, headers=headers, stream=True, timeout=timeout)
        except requests.RequestException as e:  # network error
            time.sleep(min(60, 2 ** attempt * 3))
            last = str(e)
            continue
        if r.status_code in (404, 410):
            raise Gone(str(r.status_code))
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "60") or 60)
            print(f"  mirror says 429, sleeping {wait}s", flush=True)
            time.sleep(wait)
            last = "429"
            continue
        if 500 <= r.status_code < 600:
            time.sleep(min(60, 2 ** attempt * 3))
            last = str(r.status_code)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        part = dest.with_suffix(".part")
        with open(part, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        if not zipfile.is_zipfile(part):
            part.unlink(missing_ok=True)
            raise RuntimeError("response is not a zip/.osz - cookie likely expired, log in again and pass a fresh --cookie")
        part.replace(dest)
        return
    raise RuntimeError(f"retries exhausted ({last})")


def pending_sets(db, a):
    from osumapper.v2.source_files import pending_sets as checked
    return checked(db, a)


def record(db, set_id, state, matched, expected, err):
    with db:
        db.execute("INSERT INTO set_downloads(set_id,state,attempts,matched,expected,last_error,updated_at)"
                   " VALUES(?,?,?,?,?,?,?) ON CONFLICT(set_id) DO UPDATE SET state=excluded.state,"
                   " attempts=set_downloads.attempts+?, matched=excluded.matched, expected=excluded.expected,"
                   " last_error=excluded.last_error, updated_at=excluded.updated_at",
                   (set_id, state, 1 if state in ("error", "partial") else 0, matched, expected, err, utcnow(),
                    1 if state in ("error", "partial") else 0))


def cmd_download(db, a):
    sources = []
    for m in a.mirror or []:
        if "{set_id}" not in m:
            raise SystemExit(f"--mirror must contain {{set_id}}: {m!r}")
        sources.append((f"mirror:{m.split('/')[2]}", m, {}))
    cookie = a.cookie or os.environ.get("OSU_SESSION")
    if a.use_osu or (not sources and cookie):
        if not cookie:
            raise SystemExit("--use-osu needs --cookie or OSU_SESSION env var (see the script docstring)")
        sources.append(("osu", "https://osu.ppy.sh/beatmapsets/{set_id}/download" +
                        ("?noVideo=1" if a.no_video else ""), {"cookie": cookie}))
    if not sources:
        raise SystemExit("nothing to do: pass --mirror, or --use-osu with --cookie/OSU_SESSION")

    import requests
    session = requests.Session()
    for name, _, extra in sources:
        if name == "osu":
            session.cookies.set("osu_session", extra["cookie"], domain="osu.ppy.sh")
    headers = {"User-Agent": a.user_agent}
    for h in a.header or []:
        k, _, v = h.partition(":")
        headers[k.strip()] = v.strip()
    out_root, osz_dir = Path(a.out), Path(a.out) / "_osz"
    osz_dir.mkdir(parents=True, exist_ok=True)
    todo = pending_sets(db, a)
    print(f"{len(todo)} sets to download" + (" (dataset priority order)" if todo else ""))
    t0, done, consecutive = time.monotonic(), 0, 0
    try:
        for row in todo:
            if a.max_minutes and (time.monotonic() - t0) / 60 >= a.max_minutes:
                print("time budget reached")
                break
            sid, expected = row["set_id"], row["expected"]
            dest = osz_dir / f"{sid}.osz"
            state, err, matched = "done", None, 0
            tried, used = [], None
            for name, tmpl, extra in sources:
                url = tmpl.format(set_id=sid)
                req_headers = dict(headers)
                if name == "osu":
                    req_headers["Referer"] = f"https://osu.ppy.sh/beatmapsets/{sid}"
                try:
                    fetch_osz(session, url, dest, req_headers)
                    err, used = None, name
                    break
                except Gone:
                    tried.append(f"{name}: gone")
                    err = "gone"
                    continue
                except RuntimeError as e:
                    tried.append(f"{name}: {e}")
                    err = str(e)
                    continue
            else:
                state = "gone" if err == "gone" else "error"
                record(db, sid, state, 0, expected, "; ".join(tried)[:300])
                done += 1
                consecutive += 1
                print(f"[{done}/{len(todo)}] set {sid}: {state} - {'; '.join(tried)}", flush=True)
                if consecutive >= 10:
                    raise SystemExit("10 consecutive errors, stopping (check the source(s) / rate limit)")
                time.sleep(a.delay)
                continue
            try:
                ids, unmatched = process_osz(db, dest, out_root, a.allow_updated)
                from osumapper.v2.source_files import completion
                matched, expected = completion(db, sid, a.all_sets)
                if matched < expected:
                    state, err = "partial", f"Only {matched}/{expected} selected difficulties verified"
                if matched == 0:
                    state, err = "error", f"no .osu matched the DB checksums ({unmatched} files); map updated?"
                elif unmatched:
                    err = f"{unmatched} .osu not matched (newer/older version than the API metadata)"
            except (zipfile.BadZipFile, OSError, ValueError) as e:
                state, err = "error", str(e)[:300]
            finally:
                if not a.keep_osz and state == "done":
                    dest.unlink(missing_ok=True)
            record(db, sid, state, matched, expected, err)
            done += 1
            consecutive = consecutive + 1 if state in ("error", "partial") else 0
            print(f"[{done}/{len(todo)}] set {sid}: {state} {matched}/{expected}" + (f" - {err}" if err else ""),
                  flush=True)
            if consecutive >= 10:
                raise SystemExit("10 consecutive errors, stopping (check the source(s) / rate limit)")
            if done % 50 == 0:
                sync_dataset(db)
            time.sleep(a.osu_delay if used == "osu" else a.delay)
    except KeyboardInterrupt:
        print("\ninterrupted - progress is saved, run again to resume")
    finally:
        r = sync_dataset(db)
        if r:
            print(f"dataset ready: {r[0] or 0}/{r[1]} diffs")


# --------------------------------------------------------------------------- index mode
def cmd_index(db, a):
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.dir).rglob("*.osz"))
    print(f"{len(files)} .osz files found")
    n_ok = n_skip = n_bad = total = 0
    try:
        for i, p in enumerate(files, 1):
            st = p.stat()
            # Recheck partial archives even when their size/mtime is unchanged.
            try:
                ids, unmatched = process_osz(db, p, out_root, a.allow_updated)
            except (zipfile.BadZipFile, OSError) as e:
                n_bad += 1
                print(f"[{i}/{len(files)}] {p.name}: unreadable ({e})")
                continue
            with db:
                db.execute("INSERT OR REPLACE INTO index_seen(path,size,mtime) VALUES(?,?,?)",
                           (str(p), st.st_size, st.st_mtime))
            n_ok += 1
            total += len(ids)
            if i % 100 == 0 or not ids:
                print(f"[{i}/{len(files)}] {p.name}: matched {len(ids)}, unmatched {unmatched}", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted - progress is saved")
    r = sync_dataset(db)
    print(f"indexed {n_ok} archives ({total} diffs matched), skipped {n_skip}, unreadable {n_bad}")
    if r:
        print(f"dataset ready: {r[0] or 0}/{r[1]} diffs")


# --------------------------------------------------------------------------- stats
def cmd_stats(db, a):
    print("files:", db.execute("SELECT COUNT(*) FROM files").fetchone()[0],
          "| with audio:", db.execute("SELECT COUNT(*) FROM files WHERE audio_path IS NOT NULL").fetchone()[0])
    print("downloads:", {r[0]: r[1] for r in db.execute("SELECT state, COUNT(*) FROM set_downloads GROUP BY state")})
    r = sync_dataset(db)
    if r:
        print(f"dataset ready: {r[0] or 0}/{r[1]}")
    size = sum(f.stat().st_size for f in Path(a.out).rglob("*") if f.is_file()) if Path(a.out).exists() else 0
    print(f"disk usage of {a.out}: {size / 1e9:.2f} GB")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="osu_dataset.sqlite")
    ap.add_argument("--out", default="osu_files")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("--mirror", action="append",
                    help='URL template with {set_id}; repeat for fallback chain, e.g. '
                         '--mirror "https://beatconnect.io/b/{set_id}/" --mirror "https://catboy.best/d/{set_id}"')
    d.add_argument("--use-osu", action="store_true", help="also try osu.ppy.sh (cookie) as a fallback after mirrors")
    d.add_argument("--cookie", help="value of your osu_session cookie (or set OSU_SESSION env var)")
    d.add_argument("--user-agent", default=USER_AGENT)
    d.add_argument("--no-video", action="store_true", help="request the smaller no-video .osz from osu.ppy.sh")
    d.add_argument("--header", action="append", help='extra header, e.g. "Cf-Clearance: x"')
    d.add_argument("--delay", type=float, default=1.5, help="seconds between mirror downloads")
    d.add_argument("--osu-delay", type=float, default=6.0, help="seconds between osu.ppy.sh (cookie) downloads")
    d.add_argument("--limit", type=int, default=0)
    d.add_argument("--max-minutes", type=float, default=0)
    d.add_argument("--max-attempts", type=int, default=3)
    d.add_argument("--all-sets", action="store_true", help="ignore the dataset table, take every crawled set")
    d.add_argument("--keep-osz", action="store_true", help="keep the full .osz in <out>/_osz")
    d.add_argument("--allow-updated", action="store_true",
                   help="accept a .osu whose MD5 differs if its BeatmapID matches (file newer/older than the API data)")

    i = sub.add_parser("index")
    i.add_argument("--dir", required=True, help="folder containing .osz files")
    i.add_argument("--force", action="store_true")
    i.add_argument("--allow-updated", action="store_true")

    sub.add_parser("stats")
    a = ap.parse_args()
    db = connect(a.db)
    db.executescript(DDL)
    {"download": cmd_download, "index": cmd_index, "stats": cmd_stats}[a.cmd](db, a)


if __name__ == "__main__":
    main()
