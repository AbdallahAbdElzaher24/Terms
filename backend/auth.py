"""
Minimal local auth: salted-hash passwords + opaque session tokens stored in
SQLite, plus optional Google Sign-In. No JWT — this app runs on localhost
for one person, so this is intentionally simple. Swap for a real auth
provider before this ever leaves your machine.
"""
import hashlib
import hmac
import os
import secrets

from fastapi import Header, HTTPException

from database import get_conn, new_id, now

PEPPER = os.environ.get("AUTH_PEPPER", "local-dev-pepper-change-me")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.sha256((salt + PEPPER + password).encode()).hexdigest()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest = stored.split("$")
    except ValueError:
        return False
    check = hashlib.sha256((salt + PEPPER + password).encode()).hexdigest()
    return hmac.compare_digest(check, digest)


def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, now()),
        )
    return token


def revoke_session(token: str) -> None:
    """Used by POST /api/auth/logout. Deleting a row that doesn't exist is a
    harmless no-op, so this is safe to call with an already-expired token."""
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def google_login(credential: str) -> tuple[str, str]:
    """Verifies a Google Identity Services ID token server-side (never trust
    a client-sent email/name — the whole point of verifying is that only
    Google's signature can be trusted) and returns (user_id, username),
    creating the user on first sign-in.

    Setup: create an OAuth 2.0 Client ID (Web application) in Google Cloud
    Console, set GOOGLE_CLIENT_ID in .env, and use the same client ID in the
    frontend's Google Identity Services <script> config.
    """
    if not GOOGLE_CLIENT_ID:
        raise ValueError("GOOGLE_CLIENT_ID is not set on the backend — add it to .env")

    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token
    except ImportError as e:
        raise ValueError(
            f"google-auth isn't installed ({e}). Run: pip install google-auth"
        )

    try:
        claims = id_token.verify_oauth2_token(
            credential, google_requests.Request(), GOOGLE_CLIENT_ID,
            # A few seconds of local-clock drift (NTP sync isn't instant, and
            # Windows/etc. only resync periodically) shouldn't fail sign-in —
            # google-auth defaults this to 0, i.e. zero tolerance.
            clock_skew_in_seconds=10,
        )
    except Exception as e:
        raise ValueError(f"Invalid Google token: {e}")

    google_id = claims["sub"]
    email = claims.get("email")
    username = claims.get("name") or (email.split("@")[0] if email else "Google user")

    with get_conn() as conn:
        row = conn.execute("SELECT id, username FROM users WHERE google_id = ?", (google_id,)).fetchone()
        if row:
            return row["id"], row["username"]

        # First time this Google account signs in. If an account with the
        # same email already exists (e.g. they registered with a password
        # first), link the Google id to it instead of creating a duplicate.
        if email:
            existing = conn.execute("SELECT id, username FROM users WHERE email = ?", (email,)).fetchone()
            if existing:
                conn.execute("UPDATE users SET google_id = ? WHERE id = ?", (google_id, existing["id"]))
                return existing["id"], existing["username"]

        user_id = new_id()
        conn.execute(
            """INSERT INTO users (id, username, email, password_hash, google_id, is_guest, created_at)
               VALUES (?,?,?,NULL,?,0,?)""",
            (user_id, username, email, google_id, now()),
        )
        return user_id, username


def get_current_user(authorization: str = Header(default=None)):
    """FastAPI dependency: expects `Authorization: Bearer <token>`."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    with get_conn() as conn:
        row = conn.execute(
            """SELECT users.* FROM sessions
               JOIN users ON users.id = sessions.user_id
               WHERE sessions.token = ?""",
            (token,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Session expired or invalid — please sign in again")
    return dict(row)
