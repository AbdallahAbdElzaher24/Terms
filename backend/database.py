"""
SQLite data layer.
Kept as plain sqlite3 (no ORM) on purpose — this is a local single-machine
app, so the extra ceremony of SQLAlchemy isn't buying anything yet. Swap
this module out later if the project grows a real multi-user deployment.
"""
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    email TEXT UNIQUE,
    password_hash TEXT,           -- NULL for guest and Google accounts
    google_id TEXT UNIQUE,        -- Google's 'sub' claim, NULL for password/guest accounts
    is_guest INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL,           -- 'user' | 'bot'
    content TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(chat_id) REFERENCES chats(id)
);

CREATE TABLE IF NOT EXISTS uploaded_files (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    chat_id TEXT,
    kind TEXT NOT NULL,           -- 'image' | 'audio' | 'contract'
    original_name TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    extracted_text TEXT,          -- filled in once OCR / Whisper is wired up
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,          -- same id used as document_id in the embeddings store (services/rag/vector_store.py)
    user_id TEXT NOT NULL,
    file_id TEXT,                 -- FK -> uploaded_files.id, nullable (e.g. pasted text has no file)
    title TEXT NOT NULL,
    source_kind TEXT NOT NULL,    -- 'pdf' | 'docx' | 'image' | 'audio' | 'pasted_text' | 'url'
    num_chunks INTEGER DEFAULT 0,
    compliance_gaps TEXT,         -- JSON array of gap titles
    processing_warnings TEXT,     -- JSON array — which pipeline stages degraded (missing models)
    created_at REAL NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- Every query above filters or joins on these columns (sessions.token is
-- already indexed for free via PRIMARY KEY). Without these, each lookup is a
-- full table scan once chat/message history grows past a trivial size.
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_chat_created ON messages(chat_id, created_at);
CREATE INDEX IF NOT EXISTS idx_uploaded_files_user ON uploaded_files(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
"""


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # Lightweight migration: CREATE TABLE IF NOT EXISTS above won't add
        # new columns to a users table that already existed on disk from
        # before google_id was introduced — add it if it's missing.
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
        if "google_id" not in existing_cols:
            conn.execute("ALTER TABLE users ADD COLUMN google_id TEXT")
        # WAL mode lets reads and writes proceed concurrently instead of
        # blocking each other (default journal mode takes an exclusive lock
        # for the whole duration of any write) — this is a one-time,
        # persistent setting stored in the DB file itself.
        conn.execute("PRAGMA journal_mode = WAL")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # If two requests do briefly collide on a write, retry for up to 5s
    # instead of failing immediately with "database is locked".
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> float:
    return time.time()
