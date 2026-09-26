#!/usr/bin/env python3
"""
osu! metadata crawler -> SQLite dataset (osu!standard, ranked/loved).

Pipeline (each step is resumable, everything lives in one SQLite file):

  discover   walk beatmapsets/search, fill crawl_queue with set IDs
  fetch      GET beatmapsets/{id} for queued sets in priority order,
             store sets / beatmaps / tags / beatmap_tags
  build      filter + score + split -> `dataset` table (+ v_train/v_val/v_test)
  stats      quick overview of the DB

Credentials come from environment variables (osu! API v2 OAuth app):
  OSU_CLIENT_ID, OSU_CLIENT_SECRET

Rate limit: osu! asks for <= 60 requests/minute. --min-interval is clamped to >= 1.0 s.
Only `pip install requests` is needed (and only for discover/fetch).
"""
from __future__ import annotations

# Allow direct invocation using the project Python, without reinstalling V1.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import zlib
from datetime import datetime, timezone

API_BASE = "https://osu.ppy.sh/api/v2"
TOKEN_URL = "https://osu.ppy.sh/oauth/token"
MIN_INTERVAL_FLOOR = 1.0

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS crawl_queue(
    set_id          INTEGER PRIMARY KEY,
    status          TEXT,
    favourite_count INTEGER,
    play_count      INTEGER,
    ranked_date     TEXT,
    state           TEXT NOT NULL DEFAULT 'pending',   -- pending | done | gone | error
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    discovered_at   TEXT,
    fetched_at      TEXT
);
CREATE INDEX IF NOT EXISTS ix_queue_state ON crawl_queue(state);

CREATE TABLE IF NOT EXISTS sets(
    id INTEGER PRIMARY KEY,
    status TEXT,
    artist TEXT, artist_unicode TEXT, title TEXT, title_unicode TEXT,
    creator TEXT, creator_id INTEGER, source TEXT, tags_text TEXT,
    favourite_count INTEGER, play_count INTEGER,
    rating REAL, rating_votes INTEGER, rating_sum INTEGER, ratings_json TEXT,
    genre_id INTEGER, genre TEXT, language_id INTEGER, language TEXT,
    nsfw INTEGER, video INTEGER, storyboard INTEGER, bpm REAL,
    ranked_date TEXT, submitted_date TEXT, last_updated TEXT,
    fetched_at TEXT,
    raw_zlib BLOB              -- trimmed API response (no user data), zlib-compressed JSON
);

CREATE TABLE IF NOT EXISTS beatmaps(
    id INTEGER PRIMARY KEY,
    set_id INTEGER NOT NULL,
    version TEXT, mode TEXT,
    stars REAL,                -- website star rating at fetched_at
    ar REAL, cs REAL, od REAL, hp REAL, bpm REAL,
    total_length INTEGER, hit_length INTEGER,
    count_circles INTEGER, count_sliders INTEGER, count_spinners INTEGER,
    max_combo INTEGER,
    playcount INTEGER, passcount INTEGER,
    status TEXT, checksum TEXT, lazer_only INTEGER,
    user_id INTEGER, mapper_ids TEXT, mapper_names TEXT,
    fail_json TEXT, exit_json TEXT,   -- 100-bucket fail / exit histograms along the map
    last_updated TEXT, fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_beatmaps_set ON beatmaps(set_id);

CREATE TABLE IF NOT EXISTS tags(
    id INTEGER PRIMARY KEY, name TEXT, category TEXT, ruleset_id INTEGER, description TEXT
);

CREATE TABLE IF NOT EXISTS beatmap_tags(
    beatmap_id INTEGER NOT NULL, tag_id INTEGER NOT NULL, votes INTEGER NOT NULL,
    PRIMARY KEY(beatmap_id, tag_id)
);

-- filled by your own downloader later (the API has no file download)
CREATE TABLE IF NOT EXISTS files(
    beatmap_id INTEGER PRIMARY KEY,
    osu_path TEXT, audio_path TEXT, audio_sha256 TEXT, downloaded_at TEXT
);
"""

ORDERS = {
    "favourites": "favourite_count DESC, set_id",
    "plays": "play_count DESC, set_id",
    "newest": "ranked_date DESC, set_id",
    "oldest": "ranked_date ASC, set_id",
    "random": "RANDOM()",
}

# fields that are user data or not needed for osu!standard training
DROP_SET_KEYS = ("recent_favourites", "related_users", "user", "converts",
                 "description", "current_nominations", "covers")
DROP_MAP_KEYS = ("current_user_playcount", "current_user_tag_ids")


# --------------------------------------------------------------------------- helpers
def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def meta_get(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def meta_set(db, key, value):
    with db:
        db.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                   "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


# --------------------------------------------------------------------------- API client
class ApiError(Exception):
    pass


class NotFound(ApiError):
    pass


class AuthError(Exception):
    """Fatal: bad client id/secret. Not an ApiError so the crawl loop stops."""


class Api:
    def __init__(self, client_id, client_secret, min_interval=1.05):
        import requests  # imported lazily so build/stats work without it
        self.requests = requests
        self.session = requests.Session()
        self.client_id, self.client_secret = client_id, client_secret
        self.min_interval = max(MIN_INTERVAL_FLOOR, min_interval)
        self.token, self.token_exp, self.last = None, 0.0, 0.0

    def _auth(self):
        try:
            r = self.session.post(TOKEN_URL, data={
                "client_id": self.client_id, "client_secret": self.client_secret,
                "grant_type": "client_credentials", "scope": "public"}, timeout=30)
        except self.requests.RequestException as e:
            raise ApiError(f"token request failed: {e}")
        if r.status_code != 200:
            raise AuthError(f"token request returned {r.status_code}: {r.text[:200]}")
        j = r.json()
        self.token = j["access_token"]
        self.token_exp = time.time() + int(j.get("expires_in", 3600))

    def _throttle(self):
        wait = self.min_interval - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.monotonic()

    def get(self, path, params=None):
        for attempt in range(6):
            if not self.token or time.time() > self.token_exp - 300:
                self._auth()
            self._throttle()
            try:
                r = self.session.get(
                    API_BASE + path, params=params, timeout=30,
                    headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"})
            except self.requests.RequestException as e:
                time.sleep(min(60, 2 ** attempt * 2))
                err = str(e)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 401:
                self.token = None
                err = "401"
                continue
            if r.status_code == 404:
                raise NotFound(path)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", "60") or 60)
                print(f"  rate limited (429), sleeping {wait}s", flush=True)
                time.sleep(wait)
                err = "429"
                continue
            if 500 <= r.status_code < 600:
                time.sleep(min(60, 2 ** attempt * 2))
                err = str(r.status_code)
                continue
            raise ApiError(f"{path} -> HTTP {r.status_code}: {r.text[:200]}")
        raise ApiError(f"{path}: retries exhausted ({err})")


# --------------------------------------------------------------------------- discover
def cmd_discover(db, api, a):
    for status in [s.strip() for s in a.statuses.split(",") if s.strip()]:
        ckey, dkey = f"cursor:{status}", f"discover_done:{status}"
        if a.rescan:
            meta_set(db, ckey, "")
            meta_set(db, dkey, "")
        if meta_get(db, dkey):
            print(f"[{status}] already discovered (use --rescan to walk again)")
            continue
        cursor = meta_get(db, ckey) or None
        pages = 0
        while True:
            params = {"m": 0, "s": status, "sort": "ranked_asc", "nsfw": "true"}
            if cursor:
                params["cursor_string"] = cursor
            data = api.get("/beatmapsets/search", params)
            if "beatmapsets" not in data:
                raise SystemExit(f"unexpected search response, keys: {list(data)[:10]}")
            sets = data["beatmapsets"]
            with db:
                for s in sets:
                    db.execute(
                        "INSERT INTO crawl_queue(set_id,status,favourite_count,play_count,ranked_date,discovered_at)"
                        " VALUES(?,?,?,?,?,?) ON CONFLICT(set_id) DO UPDATE SET"
                        " status=excluded.status, favourite_count=excluded.favourite_count,"
                        " play_count=excluded.play_count, ranked_date=excluded.ranked_date",
                        (s["id"], s.get("status"), s.get("favourite_count"),
                         s.get("play_count"), s.get("ranked_date"), utcnow()))
            cursor = data.get("cursor_string")
            meta_set(db, ckey, cursor or "")
            pages += 1
            total = db.execute("SELECT COUNT(*) FROM crawl_queue").fetchone()[0]
            print(f"[{status}] page {pages}: +{len(sets)} sets, queue={total}", flush=True)
            if not sets or not cursor:
                meta_set(db, dkey, utcnow())
                break
            if a.max_pages and pages >= a.max_pages:
                print(f"[{status}] stopped at --max-pages; run again to continue")
                break


# --------------------------------------------------------------------------- ingest
def strip_raw(s):
    r = {k: v for k, v in s.items() if k not in DROP_SET_KEYS}
    r["beatmaps"] = [{k: v for k, v in b.items() if k not in DROP_MAP_KEYS}
                     for b in s.get("beatmaps") or []]
    return r


def _insert(db, table, row):
    cols = list(row)
    db.execute(f"INSERT OR REPLACE INTO {table}({','.join(cols)}) "
               f"VALUES({','.join(':' + c for c in cols)})", row)


def ingest_set(db, s, keep_raw=True):
    """Store one beatmapset response. Only native osu!standard, non-convert diffs are kept."""
    now = utcnow()
    sid = int(s["id"])
    ratings = s.get("ratings") or []
    raw = None
    if keep_raw:
        raw = zlib.compress(json.dumps(strip_raw(s), ensure_ascii=False,
                                       separators=(",", ":")).encode("utf-8"), 6)
    set_row = {
        "id": sid, "status": s.get("status"),
        "artist": s.get("artist"), "artist_unicode": s.get("artist_unicode"),
        "title": s.get("title"), "title_unicode": s.get("title_unicode"),
        "creator": s.get("creator"), "creator_id": s.get("user_id"),
        "source": s.get("source"), "tags_text": s.get("tags"),
        "favourite_count": s.get("favourite_count"), "play_count": s.get("play_count"),
        "rating": s.get("rating"), "rating_votes": sum(ratings),
        "rating_sum": sum(i * c for i, c in enumerate(ratings)),
        "ratings_json": json.dumps(ratings),
        "genre_id": s.get("genre_id"), "genre": (s.get("genre") or {}).get("name"),
        "language_id": s.get("language_id"), "language": (s.get("language") or {}).get("name"),
        "nsfw": int(bool(s.get("nsfw"))), "video": int(bool(s.get("video"))),
        "storyboard": int(bool(s.get("storyboard"))), "bpm": s.get("bpm"),
        "ranked_date": s.get("ranked_date"), "submitted_date": s.get("submitted_date"),
        "last_updated": s.get("last_updated"), "fetched_at": now, "raw_zlib": raw,
    }
    diffs = tagged = unknown_tags = 0
    with db:
        _insert(db, "sets", set_row)
        for t in s.get("related_tags") or []:
            name = t.get("name") or ""
            db.execute(
                "INSERT INTO tags(id,name,category,ruleset_id,description) VALUES(?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET name=excluded.name, category=excluded.category,"
                " ruleset_id=excluded.ruleset_id, description=excluded.description",
                (t["id"], name, name.split("/")[0], t.get("ruleset_id"), t.get("description")))
        db.execute("DELETE FROM beatmap_tags WHERE beatmap_id IN "
                   "(SELECT id FROM beatmaps WHERE set_id=?)", (sid,))
        db.execute("DELETE FROM beatmaps WHERE set_id=?", (sid,))
        known = {r[0] for r in db.execute("SELECT id FROM tags")}
        for b in s.get("beatmaps") or []:
            if b.get("mode") != "osu" or b.get("convert") or b.get("deleted_at"):
                continue
            owners = b.get("owners") or []
            ft = b.get("failtimes") or {}
            row = {
                "id": b["id"], "set_id": sid, "version": b.get("version"), "mode": b.get("mode"),
                "stars": b.get("difficulty_rating"), "ar": b.get("ar"), "cs": b.get("cs"),
                "od": b.get("accuracy"), "hp": b.get("drain"), "bpm": b.get("bpm"),
                "total_length": b.get("total_length"), "hit_length": b.get("hit_length"),
                "count_circles": b.get("count_circles"), "count_sliders": b.get("count_sliders"),
                "count_spinners": b.get("count_spinners"), "max_combo": b.get("max_combo"),
                "playcount": b.get("playcount"), "passcount": b.get("passcount"),
                "status": b.get("status"), "checksum": b.get("checksum"),
                "lazer_only": int(bool(b.get("lazer_only"))), "user_id": b.get("user_id"),
                "mapper_ids": json.dumps([o.get("id") for o in owners]),
                "mapper_names": json.dumps([o.get("username") for o in owners], ensure_ascii=False),
                "fail_json": json.dumps(ft.get("fail") or []),
                "exit_json": json.dumps(ft.get("exit") or []),
                "last_updated": b.get("last_updated"), "fetched_at": now,
            }
            _insert(db, "beatmaps", row)
            diffs += 1
            votes = b.get("top_tag_ids") or []
            for v in votes:
                db.execute("INSERT OR REPLACE INTO beatmap_tags(beatmap_id,tag_id,votes) VALUES(?,?,?)",
                           (b["id"], v["tag_id"], v["count"]))
                if v["tag_id"] not in known:
                    unknown_tags += 1
            tagged += 1 if votes else 0
    return {"diffs": diffs, "tagged": tagged, "unknown_tags": unknown_tags}


# --------------------------------------------------------------------------- fetch
def cmd_fetch(db, api, a):
    order = ORDERS[a.order]
    sql = (f"SELECT set_id, favourite_count FROM crawl_queue "
           f"WHERE (state='pending' OR (state='error' AND attempts<?)) "
           f"AND COALESCE(favourite_count,0)>=? "
           f"ORDER BY (state='error'), {order} LIMIT 1")
    count_sql = ("SELECT COUNT(*) FROM crawl_queue WHERE (state='pending' OR (state='error' AND attempts<?)) "
                 "AND COALESCE(favourite_count,0)>=?")
    done = consecutive_errors = 0
    t0 = time.monotonic()
    try:
        while True:
            if a.limit and done >= a.limit:
                break
            if a.max_minutes and (time.monotonic() - t0) / 60 >= a.max_minutes:
                print("time budget reached")
                break
            row = db.execute(sql, (a.max_attempts, a.min_favourites)).fetchone()
            if not row:
                print("queue empty")
                break
            sid = row["set_id"]
            info, state, err = None, "done", None
            try:
                info = ingest_set(db, api.get(f"/beatmapsets/{sid}"), keep_raw=not a.no_raw)
            except NotFound:
                state, err = "gone", "404"
            except (ApiError, KeyError, TypeError, ValueError) as e:
                state, err = "error", f"{type(e).__name__}: {e}"[:300]
            with db:
                db.execute("UPDATE crawl_queue SET state=?, last_error=?, attempts=attempts+?, "
                           "fetched_at=? WHERE set_id=?",
                           (state, err, 1 if state == "error" else 0, utcnow(), sid))
            done += 1
            consecutive_errors = consecutive_errors + 1 if state == "error" else 0
            if consecutive_errors >= 10:
                raise SystemExit("10 consecutive errors, stopping (check network / credentials)")
            if state == "done":
                msg = f"diffs={info['diffs']} tagged={info['tagged']}"
                if info["unknown_tags"]:
                    msg += f" unknown_tag_ids={info['unknown_tags']}"
            else:
                msg = f"{state}: {err}"
            print(f"[{done}] set {sid} fav={row['favourite_count']} {msg}", flush=True)
            if done % 25 == 0:
                left = db.execute(count_sql, (a.max_attempts, a.min_favourites)).fetchone()[0]
                print(f"    remaining ~{left} sets, ETA ~{left * api.min_interval / 3600:.1f} h", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted - progress is saved, run fetch again to resume")


# --------------------------------------------------------------------------- build
def song_key(artist, title):
    def norm(x, strip_brackets=True):
        x = (x or "").lower()
        if strip_brackets:
            x = re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", x)
        return re.sub(r"[\W_]+", "", x)
    t = norm(title) or norm(title, False)
    return norm(artist) + "|" + t


def split_for(key, val_frac, test_frac):
    u = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


def group_percentiles(items, key, value):
    """Percentile rank (0..1, ties averaged) of value(item) within groups defined by key(item)."""
    groups = {}
    for idx, it in enumerate(items):
        groups.setdefault(key(it), []).append((value(it), idx))
    out = [0.5] * len(items)
    for g in groups.values():
        g.sort()
        n, i = len(g), 0
        if n == 1:
            continue
        while i < n:
            j = i
            while j + 1 < n and g[j + 1][0] == g[i][0]:
                j += 1
            for k in range(i, j + 1):
                out[g[k][1]] = ((i + j) / 2) / (n - 1)
            i = j + 1
    return out


DATASET_DDL = """
DROP VIEW IF EXISTS v_train; DROP VIEW IF EXISTS v_val; DROP VIEW IF EXISTS v_test;
DROP TABLE IF EXISTS dataset;
CREATE TABLE dataset(
    beatmap_id INTEGER PRIMARY KEY, set_id INTEGER, split TEXT, song_key TEXT,
    artist TEXT, title TEXT, version TEXT,
    stars REAL, ar REAL, cs REAL, od REAL, hp REAL, bpm REAL, length_s INTEGER,
    n_circles INTEGER, n_sliders INTEGER, n_spinners INTEGER, max_combo INTEGER,
    playcount INTEGER, passcount INTEGER, pass_rate REAL,
    mappers TEXT, set_status TEXT, genre TEXT, language TEXT, ranked_date TEXT, nsfw INTEGER,
    rating_bayes REAL, fav_rate REAL, quality REAL, sample_weight REAL,
    tags_json TEXT, tag_names TEXT,
    osu_path TEXT, audio_path TEXT, ready INTEGER
);
CREATE INDEX ix_dataset_split ON dataset(split);
CREATE VIEW v_train AS SELECT * FROM dataset WHERE split='train';
CREATE VIEW v_val   AS SELECT * FROM dataset WHERE split='val';
CREATE VIEW v_test  AS SELECT * FROM dataset WHERE split='test';
"""


def cmd_build(db, a):
    statuses = [s.strip() for s in a.statuses.split(",") if s.strip()]
    marks = ",".join("?" * len(statuses))
    rows = [dict(r) for r in db.execute(f"""
        SELECT b.id AS beatmap_id, b.set_id, b.version, b.stars, b.ar, b.cs, b.od, b.hp, b.bpm,
               b.hit_length, b.count_circles, b.count_sliders, b.count_spinners, b.max_combo,
               COALESCE(b.playcount,0) AS playcount, COALESCE(b.passcount,0) AS passcount,
               b.mapper_names, s.status AS set_status, s.artist, s.title,
               COALESCE(s.favourite_count,0) AS favourite_count, COALESCE(s.play_count,0) AS set_plays,
               s.rating_votes, s.rating_sum, s.genre, s.language, s.ranked_date, s.nsfw,
               f.osu_path, f.audio_path
        FROM beatmaps b JOIN sets s ON s.id=b.set_id
        LEFT JOIN files f ON f.beatmap_id=b.id
        WHERE b.lazer_only=0 AND s.status IN ({marks})
          AND b.stars BETWEEN ? AND ?
          AND COALESCE(b.playcount,0)>=?
          AND b.hit_length BETWEEN ? AND ?
        """, (*statuses, a.min_stars, a.max_stars, a.min_playcount, a.min_length, a.max_length))]
    if not rows:
        raise SystemExit("no rows match the filters (did fetch run? try lower --min-playcount)")

    prior = db.execute("SELECT AVG(rating_sum*1.0/rating_votes) FROM sets WHERE rating_votes>0").fetchone()[0] or 7.5
    C = a.rating_prior_votes
    for r in rows:
        n, tot = r["rating_votes"] or 0, r["rating_sum"] or 0
        r["rating_bayes"] = (C * prior + tot) / (C + n)
        r["fav_rate"] = r["favourite_count"] / max(r["set_plays"], 1)
        r["pass_rate"] = r["passcount"] / r["playcount"] if r["playcount"] else None
        r["song_key"] = song_key(r["artist"], r["title"])
        r["split"] = split_for(r["song_key"], a.val_frac, a.test_frac)

    bucket = lambda r: min(int(r["stars"]), 10)  # percentiles inside 1-star buckets (removes star confound)
    p_rating = group_percentiles(rows, bucket, lambda r: r["rating_bayes"])
    p_fav = group_percentiles(rows, bucket, lambda r: r["fav_rate"])
    p_plays = group_percentiles(rows, bucket, lambda r: math.log1p(r["playcount"]))
    status_factor = {"loved": a.loved_factor}

    tags = {}
    for t in db.execute("""SELECT bt.beatmap_id, bt.tag_id, COALESCE(t.name,'unknown/'||bt.tag_id) AS name, bt.votes
                           FROM beatmap_tags bt LEFT JOIN tags t ON t.id=bt.tag_id
                           ORDER BY bt.beatmap_id, bt.votes DESC"""):
        tags.setdefault(t["beatmap_id"], []).append((t["name"], t["votes"]))

    out = []
    for i, r in enumerate(rows):
        q = 0.5 * p_rating[i] + 0.3 * p_fav[i] + 0.2 * p_plays[i]
        weight = (0.2 + 0.8 * q) * status_factor.get(r["set_status"], 1.0)
        tl = tags.get(r["beatmap_id"], [])
        top = tl[0][1] if tl else 1
        strong = [{"tag": n, "votes": v, "rel": round(v / top, 3)} for n, v in tl if v >= a.min_tag_votes]
        mappers = ", ".join(json.loads(r["mapper_names"] or "[]"))
        ready = int(bool(r["osu_path"] and r["audio_path"]))
        out.append((
            r["beatmap_id"], r["set_id"], r["split"], r["song_key"], r["artist"], r["title"], r["version"],
            r["stars"], r["ar"], r["cs"], r["od"], r["hp"], r["bpm"], r["hit_length"],
            r["count_circles"], r["count_sliders"], r["count_spinners"], r["max_combo"],
            r["playcount"], r["passcount"], r["pass_rate"], mappers, r["set_status"], r["genre"],
            r["language"], r["ranked_date"], r["nsfw"], r["rating_bayes"], r["fav_rate"],
            q, weight, json.dumps(strong, ensure_ascii=False), " ".join(s["tag"] for s in strong),
            r["osu_path"], r["audio_path"], ready))
    if a.only_ready:
        out = [o for o in out if o[-1]]
    db.executescript(DATASET_DDL)
    with db:
        if out:
            db.executemany(f"INSERT INTO dataset VALUES({','.join('?' * len(out[0]))})", out)
    meta_set(db, "last_build", json.dumps({"at": utcnow(), "rows": len(out), "args": vars(a)}, default=str))

    print(f"dataset: {len(out)} beatmaps")
    for split, n, sets, avg in db.execute(
            "SELECT split, COUNT(*), COUNT(DISTINCT set_id), ROUND(AVG(stars),2) FROM dataset GROUP BY split"):
        print(f"  {split:5s} {n:7d} diffs  {sets:6d} sets  avg stars {avg}")
    with_tags = db.execute("SELECT COUNT(*) FROM dataset WHERE tags_json!='[]'").fetchone()[0]
    print(f"  diffs with >=1 tag ({a.min_tag_votes}+ votes): {with_tags} ({100 * with_tags / max(len(out), 1):.0f}%)")
    print("  views: v_train, v_val, v_test  (add 'WHERE ready=1' once files are downloaded)")


# --------------------------------------------------------------------------- stats
def cmd_stats(db, a):
    q = lambda sql: db.execute(sql).fetchall()
    print("queue:", {r[0]: r[1] for r in q("SELECT state, COUNT(*) FROM crawl_queue GROUP BY state")})
    print("sets:", q("SELECT COUNT(*) FROM sets")[0][0], "| beatmaps:", q("SELECT COUNT(*) FROM beatmaps")[0][0],
          "| tags:", q("SELECT COUNT(*) FROM tags")[0][0], "| tag votes:", q("SELECT COUNT(*) FROM beatmap_tags")[0][0])
    print("sets by status:", {r[0]: r[1] for r in q("SELECT status, COUNT(*) FROM sets GROUP BY status")})
    print("stars histogram:", {int(r[0]): r[1] for r in q(
        "SELECT CAST(stars AS INT), COUNT(*) FROM beatmaps GROUP BY 1 ORDER BY 1")})
    print("top tags:", [(r[0], r[1]) for r in q(
        "SELECT t.name, SUM(bt.votes) v FROM beatmap_tags bt JOIN tags t ON t.id=bt.tag_id "
        "GROUP BY t.name ORDER BY v DESC LIMIT 12")])
    lb = meta_get(db, "last_build")
    if lb:
        print("last build:", json.loads(lb)["at"], "rows", json.loads(lb)["rows"])


# --------------------------------------------------------------------------- main
def make_api(a):
    cid, secret = os.environ.get("OSU_CLIENT_ID"), os.environ.get("OSU_CLIENT_SECRET")
    if not cid or not secret:
        raise SystemExit("set OSU_CLIENT_ID and OSU_CLIENT_SECRET first")
    return Api(cid, secret, a.min_interval)


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="osu_dataset.sqlite")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="fill the crawl queue from beatmapsets/search")
    d.add_argument("--statuses", default="ranked,loved")
    d.add_argument("--rescan", action="store_true", help="walk the list again (finds new sets)")
    d.add_argument("--max-pages", type=int, default=0)
    d.add_argument("--min-interval", type=float, default=1.05)

    f = sub.add_parser("fetch", help="download set details in priority order")
    f.add_argument("--order", choices=list(ORDERS), default="favourites")
    f.add_argument("--min-favourites", type=int, default=0)
    f.add_argument("--limit", type=int, default=0, help="max sets this run (0 = all)")
    f.add_argument("--max-minutes", type=float, default=0)
    f.add_argument("--max-attempts", type=int, default=3)
    f.add_argument("--no-raw", action="store_true", help="do not keep the compressed raw JSON")
    f.add_argument("--min-interval", type=float, default=1.05)

    b = sub.add_parser("build", help="filter, score and split into the `dataset` table")
    b.add_argument("--statuses", default="ranked,approved,loved")
    b.add_argument("--min-stars", type=float, default=0.0)
    b.add_argument("--max-stars", type=float, default=15.0)
    b.add_argument("--min-playcount", type=int, default=50, help="per difficulty")
    b.add_argument("--min-length", type=int, default=15, help="hit_length seconds")
    b.add_argument("--max-length", type=int, default=600)
    b.add_argument("--min-tag-votes", type=int, default=5)
    b.add_argument("--rating-prior-votes", type=float, default=10.0)
    b.add_argument("--loved-factor", type=float, default=0.85)
    b.add_argument("--val-frac", type=float, default=0.05)
    b.add_argument("--test-frac", type=float, default=0.05)
    b.add_argument("--only-ready", action="store_true", help="keep only rows that have files downloaded")

    sub.add_parser("stats")
    return ap


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = parser().parse_args()
    db = connect(a.db)
    try:
        if a.cmd == "discover":
            cmd_discover(db, make_api(a), a)
        elif a.cmd == "fetch":
            cmd_fetch(db, make_api(a), a)
        elif a.cmd == "build":
            cmd_build(db, a)
        else:
            cmd_stats(db, a)
    except AuthError as e:
        raise SystemExit(f"authentication failed: {e}")
    except ApiError as e:
        raise SystemExit(f"API error: {e} (progress is saved, run again to resume)")
    except KeyboardInterrupt:
        raise SystemExit("\ninterrupted - progress is saved, run again to resume")


if __name__ == "__main__":
    main()
