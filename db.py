"""
db.py — SQLite data layer for PopcornAI
==========================================================
Local email+password auth (no OAuth setup needed), plus:
1. users        — email (PK), password_hash, salt
2. watchlist    — per-user saved movies, keyed by email
3. notifications — app-wide update feed (movies/app updates)
4. notif_clear  — per-user "cleared before this timestamp" marker,
                  so hitting Clear only affects that user's inbox,
                  and it stays cleared across logout/login (it's in
                  the DB, not session_state, which would reset).

DB file `popcornai.db` is created automatically on first run.
"""

import sqlite3
import hashlib
import re
import secrets
from datetime import datetime, timedelta

DB_FILE = "popcornai.db"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ==========================================================================
# CONNECTION + SETUP
# ==========================================================================
def get_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _column_exists(c, table, column):
    c.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in c.fetchall())


def _table_exists(c, table):
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return c.fetchone() is not None


def _migrate_old_username_schema(c):
    """Earlier versions of this app used a `username` column instead of
    `email`. If an existing popcornai.db still has that old schema, this
    migrates it in place — including correctly looking up each old user's
    real email address (not just reusing their username) — instead of
    crashing with 'no such column: email'. Safe to call every run — it's
    a no-op once the DB is already on the new schema."""
    try:
        # Capture username -> real email BEFORE anything gets renamed/dropped,
        # so watchlist rows can be correctly re-keyed by actual email.
        username_to_email = {}
        if _table_exists(c, "users") and _column_exists(c, "users", "username") \
                and _column_exists(c, "users", "email"):
            c.execute("SELECT username, email FROM users")
            for uname, uemail in c.fetchall():
                if uemail:
                    username_to_email[uname] = uemail

        # --- users table: rebuild keyed by email if it's still keyed by username ---
        if _table_exists(c, "users"):
            c.execute("PRAGMA table_info(users)")
            pk_col = next((row[1] for row in c.fetchall() if row[5] == 1), None)
            if pk_col == "username":
                c.execute("ALTER TABLE users RENAME TO users_old")
                c.execute("""
                    CREATE TABLE users (
                        email           TEXT PRIMARY KEY,
                        password_hash   TEXT NOT NULL,
                        salt            TEXT NOT NULL
                    )
                """)
                c.execute("""
                    INSERT OR IGNORE INTO users (email, password_hash, salt)
                    SELECT COALESCE(NULLIF(email, ''), username), password_hash, salt
                    FROM users_old
                """)
                c.execute("DROP TABLE users_old")

        # --- watchlist table: rebuild keyed by email, using the captured mapping ---
        if _table_exists(c, "watchlist") and not _column_exists(c, "watchlist", "email"):
            if _column_exists(c, "watchlist", "username"):
                c.execute("ALTER TABLE watchlist RENAME TO watchlist_old")
                c.execute("""
                    CREATE TABLE watchlist (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        email       TEXT NOT NULL,
                        title       TEXT NOT NULL,
                        poster      TEXT,
                        movie_id    INTEGER,
                        UNIQUE(email, title),
                        FOREIGN KEY (email) REFERENCES users(email) ON DELETE CASCADE
                    )
                """)
                c.execute("SELECT username, title, poster, movie_id FROM watchlist_old")
                for uname, title, poster, movie_id in c.fetchall():
                    real_email = username_to_email.get(uname, uname)  # fallback if no mapping found
                    c.execute(
                        "INSERT OR IGNORE INTO watchlist (email, title, poster, movie_id) VALUES (?, ?, ?, ?)",
                        (real_email, title, poster, movie_id),
                    )
                c.execute("DROP TABLE watchlist_old")
    except sqlite3.OperationalError:
        # If anything about this migration goes sideways, don't crash the
        # app over it — worst case old rows just stay under their old name.
        pass


def init_db():
    """Call once at app startup. Safe to call every run."""
    conn = get_connection()
    c = conn.cursor()

    _migrate_old_username_schema(c)

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email           TEXT PRIMARY KEY,
            password_hash   TEXT NOT NULL,
            salt            TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            email       TEXT NOT NULL,
            title       TEXT NOT NULL,
            poster      TEXT,
            movie_id    INTEGER,
            UNIQUE(email, title),
            FOREIGN KEY (email) REFERENCES users(email) ON DELETE CASCADE
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            icon        TEXT,
            title       TEXT NOT NULL,
            description TEXT,
            created_at  TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS notif_clear (
            email       TEXT PRIMARY KEY,
            cleared_at  TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS tmdb_cache (
            movie_id      INTEGER PRIMARY KEY,
            poster_path   TEXT,
            overview      TEXT,
            genres        TEXT,
            rating        REAL,
            runtime       TEXT,
            release_date  TEXT,
            popularity    TEXT,
            cached_at     TEXT
        )
    """)

    # Seed a few starter notifications only once (table starts empty)
    c.execute("SELECT COUNT(*) FROM notifications")
    if c.fetchone()[0] == 0:
        now = datetime.utcnow()
        seed = [
            ("🎬", "New movie trending", "The Trending list was just refreshed from TMDB.", now - timedelta(hours=2)),
            ("🚀", "PopcornAI update", "You can now save a persistent watchlist by logging in.", now - timedelta(days=1)),
            ("🏆", "Top Rated refresh", "The Top Rated list was updated this week.", now - timedelta(days=3)),
        ]
        for icon, title, desc, ts in seed:
            c.execute(
                "INSERT INTO notifications (icon, title, description, created_at) VALUES (?, ?, ?, ?)",
                (icon, title, desc, ts.isoformat()),
            )

    conn.commit()
    conn.close()


def add_notification(icon, title, description):
    """Call this whenever something notification-worthy happens (new
    movie added, app updated, etc.) to push a fresh entry into the feed."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO notifications (icon, title, description, created_at) VALUES (?, ?, ?, ?)",
        (icon, title, description, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


# ==========================================================================
# PASSWORD HASHING (simple, dependency-free — fine for a college project)
# ==========================================================================
def _hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    pwd_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return pwd_hash, salt


def _verify_password(password, salt, stored_hash):
    test_hash, _ = _hash_password(password, salt)
    return test_hash == stored_hash


def is_valid_email(email):
    return bool(_EMAIL_RE.match((email or "").strip()))


# ==========================================================================
# USER AUTH — email is the identifier, no separate username
# ==========================================================================
def signup_user(email, password, confirm_password):
    email = (email or "").strip().lower()

    if not email or not password:
        return False, "Gmail/Email and password are both required."
    if not is_valid_email(email):
        return False, "Please enter a valid Gmail/Email address (e.g. name@gmail.com)."
    if len(password) < 4:
        return False, "Password must be at least 4 characters long."
    if password != confirm_password:
        return False, "Password and Confirm Password do not match."

    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT email FROM users WHERE email = ?", (email,))
    if c.fetchone():
        conn.close()
        return False, "An account already exists with this email — try the Login tab."

    pwd_hash, salt = _hash_password(password)
    c.execute(
        "INSERT INTO users (email, password_hash, salt) VALUES (?, ?, ?)",
        (email, pwd_hash, salt),
    )
    conn.commit()
    conn.close()
    return True, "Account created! Now sign in from the Login tab."


def login_user(email, password):
    email = (email or "").strip().lower()
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT password_hash, salt FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    conn.close()

    if not row:
        return False, "No account found with this email."
    stored_hash, salt = row
    if _verify_password(password, salt, stored_hash):
        return True, "Login successful!"
    return False, "Incorrect password."


# ==========================================================================
# WATCHLIST (per-user via email, persisted in DB)
# ==========================================================================
def get_watchlist(email):
    if not email:
        return {}
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT title, poster, movie_id FROM watchlist WHERE email = ?", (email,))
    rows = c.fetchall()
    conn.close()
    return {title: {"poster": poster, "movie_id": movie_id} for title, poster, movie_id in rows}


def add_to_watchlist(email, title, poster, movie_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO watchlist (email, title, poster, movie_id) VALUES (?, ?, ?, ?)",
        (email, title, poster, movie_id),
    )
    conn.commit()
    conn.close()


def remove_from_watchlist(email, title):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM watchlist WHERE email = ? AND title = ?", (email, title))
    conn.commit()
    conn.close()


# ==========================================================================
# NOTIFICATIONS — global feed, per-user "cleared" marker so Clear doesn't
# wipe them for everyone, and survives logout -> login for that user
# (it's a DB row keyed by email, not session_state which would reset).
# ==========================================================================
def get_notifications(email=None):
    conn = get_connection()
    c = conn.cursor()
    cleared_at = None
    if email:
        c.execute("SELECT cleared_at FROM notif_clear WHERE email = ?", (email,))
        row = c.fetchone()
        if row:
            cleared_at = row[0]

    if cleared_at:
        c.execute(
            "SELECT icon, title, description, created_at FROM notifications "
            "WHERE created_at > ? ORDER BY created_at DESC",
            (cleared_at,),
        )
    else:
        c.execute(
            "SELECT icon, title, description, created_at FROM notifications ORDER BY created_at DESC"
        )
    rows = c.fetchall()
    conn.close()
    return [
        {"icon": icon, "title": title, "desc": desc, "created_at": created_at}
        for icon, title, desc, created_at in rows
    ]


def clear_notifications(email):
    """Marks everything up to now as cleared for this user. Stored in the
    DB (not session_state), so logging out and back in does NOT bring
    cleared notifications back."""
    if not email:
        return
    conn = get_connection()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute(
        "INSERT INTO notif_clear (email, cleared_at) VALUES (?, ?) "
        "ON CONFLICT(email) DO UPDATE SET cleared_at = excluded.cleared_at",
        (email, now),
    )
    conn.commit()
    conn.close()


# ==========================================================================
# TMDB RESPONSE CACHE — persisted in SQLite so posters/details fetched once
# stay fetched even after the app restarts (st.cache_data alone resets
# every time the Streamlit process restarts; this survives that).
# ==========================================================================
def get_cached_movie(movie_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT poster_path, overview, genres, rating, runtime, release_date, popularity "
        "FROM tmdb_cache WHERE movie_id = ?",
        (movie_id,),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    poster_path, overview, genres, rating, runtime, release_date, popularity = row
    return {
        "poster_path": poster_path or "",
        "overview": overview or "",
        "genres": genres or "",
        "rating": rating if rating is not None else 0,
        "runtime": runtime or "",
        "release_date": release_date or "",
        "popularity": popularity or "",
    }


def cache_movie(movie_id, data):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO tmdb_cache
            (movie_id, poster_path, overview, genres, rating, runtime, release_date, popularity, cached_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(movie_id) DO UPDATE SET
            poster_path=excluded.poster_path, overview=excluded.overview, genres=excluded.genres,
            rating=excluded.rating, runtime=excluded.runtime, release_date=excluded.release_date,
            popularity=excluded.popularity, cached_at=excluded.cached_at
        """,
        (
            movie_id,
            data.get("poster_path", ""),
            data.get("overview", ""),
            data.get("genres", ""),
            data.get("rating", 0),
            data.get("runtime", ""),
            data.get("release_date", ""),
            data.get("popularity", ""),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
