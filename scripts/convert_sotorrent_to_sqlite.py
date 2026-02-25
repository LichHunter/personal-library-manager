#!/usr/bin/env python3
"""Convert SOTorrent MySQL dumps to SQLite for benchmark pipeline.

Parses the Posts.sql and PostVersionUrl.sql MySQL INSERT statements
and loads them into SQLite.
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

POSTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS Posts (
    Id INTEGER PRIMARY KEY,
    PostTypeId INTEGER,
    ParentId INTEGER,
    AcceptedAnswerId INTEGER,
    CreationDate TEXT,
    Score INTEGER,
    Body TEXT,
    Title TEXT,
    Tags TEXT
);
CREATE INDEX IF NOT EXISTS idx_posts_parentid ON Posts(ParentId);
CREATE INDEX IF NOT EXISTS idx_posts_posttypeid ON Posts(PostTypeId);
CREATE INDEX IF NOT EXISTS idx_posts_tags ON Posts(Tags);
"""

POST_VERSION_URL_SCHEMA = """
CREATE TABLE IF NOT EXISTS PostVersionUrl (
    Id INTEGER PRIMARY KEY,
    PostId INTEGER,
    PostHistoryId INTEGER,
    Url TEXT
);
CREATE INDEX IF NOT EXISTS idx_pvurl_postid ON PostVersionUrl(PostId);
CREATE INDEX IF NOT EXISTS idx_pvurl_url ON PostVersionUrl(Url);
"""

INSERT_PATTERN = re.compile(r"INSERT INTO `?(\w+)`? VALUES\s*", re.IGNORECASE)
VALUE_PATTERN = re.compile(r"\(([^)]+)\)")


def extract_7z(archive_path: Path, output_dir: Path) -> Path:
    log.info(f"Extracting {archive_path}...")
    subprocess.run(
        ["7z", "x", "-y", f"-o{output_dir}", str(archive_path)],
        check=True,
        capture_output=True,
    )
    sql_file = output_dir / archive_path.stem
    if not sql_file.exists():
        sql_files = list(output_dir.glob("*.sql"))
        if sql_files:
            sql_file = sql_files[0]
    log.info(f"Extracted to {sql_file}")
    return sql_file


def parse_mysql_value(val: str) -> str | int | None:
    val = val.strip()
    if val.upper() == "NULL":
        return None
    if val.startswith("'") and val.endswith("'"):
        inner = val[1:-1]
        inner = inner.replace("\\'", "'")
        inner = inner.replace("\\\\", "\\")
        inner = inner.replace("\\n", "\n")
        inner = inner.replace("\\r", "\r")
        return inner
    try:
        return int(val)
    except ValueError:
        return val


def parse_values_row(row_str: str) -> list:
    values = []
    current = []
    in_string = False
    escape_next = False
    
    for char in row_str:
        if escape_next:
            current.append(char)
            escape_next = False
            continue
        if char == '\\':
            current.append(char)
            escape_next = True
            continue
        if char == "'" and not escape_next:
            in_string = not in_string
            current.append(char)
            continue
        if char == ',' and not in_string:
            values.append(parse_mysql_value(''.join(current)))
            current = []
            continue
        current.append(char)
    
    if current:
        values.append(parse_mysql_value(''.join(current)))
    
    return values


def load_posts(sql_file: Path, conn: sqlite3.Connection, limit: int | None = None):
    log.info(f"Loading Posts from {sql_file}...")
    cursor = conn.cursor()
    cursor.executescript(POSTS_SCHEMA)
    
    row_count = 0
    batch = []
    batch_size = 10000
    
    with open(sql_file, 'r', encoding='utf-8', errors='replace') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip().startswith("("):
                if "INSERT INTO" in line.upper():
                    continue
                continue
            
            line = line.strip().rstrip(",;")
            if line.startswith("(") and line.endswith(")"):
                line = line[1:-1]
            
            try:
                values = parse_values_row(line)
                if len(values) >= 9:
                    post_id, post_type_id, parent_id, accepted_answer_id, creation_date, score, body, title, tags = values[:9]
                    batch.append((post_id, post_type_id, parent_id, accepted_answer_id, creation_date, score, body, title, tags))
                    row_count += 1
            except Exception as e:
                if row_count < 10:
                    log.warning(f"Error parsing line {line_no}: {e}")
                continue
            
            if len(batch) >= batch_size:
                cursor.executemany(
                    "INSERT OR IGNORE INTO Posts (Id, PostTypeId, ParentId, AcceptedAnswerId, CreationDate, Score, Body, Title, Tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    batch
                )
                conn.commit()
                log.info(f"  Inserted {row_count} posts...")
                batch = []
            
            if limit and row_count >= limit:
                break
    
    if batch:
        cursor.executemany(
            "INSERT OR IGNORE INTO Posts (Id, PostTypeId, ParentId, AcceptedAnswerId, CreationDate, Score, Body, Title, Tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch
        )
        conn.commit()
    
    log.info(f"Loaded {row_count} posts total")


def load_post_version_url(sql_file: Path, conn: sqlite3.Connection, limit: int | None = None):
    log.info(f"Loading PostVersionUrl from {sql_file}...")
    cursor = conn.cursor()
    cursor.executescript(POST_VERSION_URL_SCHEMA)
    
    row_count = 0
    batch = []
    batch_size = 50000
    
    with open(sql_file, 'r', encoding='utf-8', errors='replace') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip().startswith("("):
                continue
            
            line = line.strip().rstrip(",;")
            if line.startswith("(") and line.endswith(")"):
                line = line[1:-1]
            
            try:
                values = parse_values_row(line)
                if len(values) >= 4:
                    url_id, post_id, post_history_id, url = values[:4]
                    batch.append((url_id, post_id, post_history_id, url))
                    row_count += 1
            except Exception as e:
                continue
            
            if len(batch) >= batch_size:
                cursor.executemany(
                    "INSERT OR IGNORE INTO PostVersionUrl (Id, PostId, PostHistoryId, Url) VALUES (?, ?, ?, ?)",
                    batch
                )
                conn.commit()
                log.info(f"  Inserted {row_count} URLs...")
                batch = []
            
            if limit and row_count >= limit:
                break
    
    if batch:
        cursor.executemany(
            "INSERT OR IGNORE INTO PostVersionUrl (Id, PostId, PostHistoryId, Url) VALUES (?, ?, ?, ?)",
            batch
        )
        conn.commit()
    
    log.info(f"Loaded {row_count} URLs total")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert SOTorrent MySQL dumps to SQLite")
    parser.add_argument("--sotorrent-dir", required=True, type=Path, help="Directory with .7z files")
    parser.add_argument("--output", required=True, type=Path, help="Output SQLite database")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows (for testing)")
    
    args = parser.parse_args(argv)
    
    posts_archive = args.sotorrent_dir / "Posts.sql.7z"
    pvurl_archive = args.sotorrent_dir / "PostVersionUrl.sql.7z"
    
    if not posts_archive.exists():
        log.error(f"Posts.sql.7z not found in {args.sotorrent_dir}")
        return 1
    if not pvurl_archive.exists():
        log.error(f"PostVersionUrl.sql.7z not found in {args.sotorrent_dir}")
        return 1
    
    log.info("=" * 60)
    log.info("SOTorrent MySQL to SQLite Conversion")
    log.info("=" * 60)
    
    extract_dir = args.sotorrent_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    
    posts_sql = extract_7z(posts_archive, extract_dir)
    pvurl_sql = extract_7z(pvurl_archive, extract_dir)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    
    conn = sqlite3.connect(args.output)
    
    try:
        load_posts(posts_sql, conn, args.limit)
        load_post_version_url(pvurl_sql, conn, args.limit)
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Posts")
        post_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM PostVersionUrl")
        url_count = cursor.fetchone()[0]
        
        log.info("-" * 60)
        log.info(f"Database: {args.output}")
        log.info(f"Posts: {post_count}")
        log.info(f"PostVersionUrl: {url_count}")
        log.info("=" * 60)
        
    finally:
        conn.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
