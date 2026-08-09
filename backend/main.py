import asyncio
import json
import logging
import mimetypes
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ — do this before any service module reads GROQ_*/etc.

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from auth import create_session, get_current_user, google_login, hash_password, revoke_session, verify_password
from database import get_conn, init_db, new_id, now
from orchestrator import build_answer_context, process_document
from services.legal_ai.compliance_rules import check_compliance
from services.legal_ai.obligations import extract_obligations
from services.parsing.docx_parser import extract_docx_full_text
from services.parsing.pdf_parser import extract_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# ------------------------------------------------------------- LLM provider --
# Groq (hosted) is the only LLM provider — needs GROQ_API_KEY, see
# services/llm/groq_service.py. No local model download/GPU needed.
from services.llm.groq_service import generate_chat_title, stream_chat

logger.info("LLM provider: groq")

app = FastAPI(title="Legal AI Assistant (local)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev only — tighten this before deploying anywhere
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE = Path(__file__).parent / "storage"
for sub in ("contracts", "images", "audio", "temp"):
    (STORAGE / sub).mkdir(parents=True, exist_ok=True)

init_db()


@app.on_event("startup")
def _warm_up_models() -> None:
    """Loads every ML model once at process startup instead of on whichever
    request happens to need it first — models are already cached after
    that (@lru_cache in each loader), this just moves the multi-second/
    multi-GB-download cold-load cost out of a user-facing request. Runs in
    a background thread so the API can start serving immediately rather
    than blocking startup on it; each loader is isolated so one missing/
    not-yet-trained model can't block the others from warming up."""
    import threading

    def _try(label: str, fn) -> None:
        try:
            fn()
            logger.info(f"[warm-up] {label}: ready")
        except Exception as e:  # noqa: BLE001 — a cold-load failure here just means the first real request pays for it and reports the same warning it always did
            logger.info(f"[warm-up] {label}: skipped ({e})")

    def _run() -> None:
        from services.legal_ai.legalbert_classifier import _load as _load_clause
        from services.rag.embeddings import _get_model as _load_embeddings
        from services.rag.reranker import _get_model as _load_reranker

        _try("Embeddings", _load_embeddings)
        _try("Reranker", _load_reranker)
        _try("Clause classifier (Legal-BERT ONNX)", lambda: _load_clause("clause"))

    threading.Thread(target=_run, name="model-warmup", daemon=True).start()


# ---------------------------------------------------------------- schemas --
class RegisterBody(BaseModel):
    username: str
    email: str
    password: str


class LoginBody(BaseModel):
    email: str
    password: str


class ChatBody(BaseModel):
    message: str
    chat_id: str | None = None
    document_id: str | None = None  # ties the message to a processed upload for RAG context
    language: str = "en"  # 'ar' or 'en' — drives which language the LLM is asked to answer in
    temporary: bool = False
    # Temporary chats aren't persisted, so there's no chat_id to reload history
    # from — the client keeps its own in-memory transcript and resends it here
    # each turn instead. Ignored for non-temporary chats (history there always
    # comes from the DB, which the client can't tamper with).
    history: list[dict] | None = None


class PasteTextBody(BaseModel):
    text: str
    title: str = "Pasted text"
    language: str = "en"


# -------------------------------------------------------------------- auth --
@app.post("/api/auth/register")
def register(body: RegisterBody):
    with get_conn() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (body.email,)).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="An account with this email already exists")
        user_id = new_id()
        conn.execute(
            "INSERT INTO users (id, username, email, password_hash, is_guest, created_at) VALUES (?,?,?,?,0,?)",
            (user_id, body.username, body.email, hash_password(body.password), now()),
        )
    token = create_session(user_id)
    return {"token": token, "username": body.username, "user_id": user_id}


@app.post("/api/auth/login")
def login(body: LoginBody):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (body.email,)).fetchone()
    if not row or not row["password_hash"] or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_session(row["id"])
    return {"token": token, "username": row["username"], "user_id": row["id"]}


@app.post("/api/auth/guest")
def guest():
    user_id = new_id()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (id, username, email, password_hash, is_guest, created_at) VALUES (?,?,?,?,1,?)",
            (user_id, "Guest", None, None, now()),
        )
    token = create_session(user_id)
    return {"token": token, "username": "Guest", "user_id": user_id}


class GoogleAuthBody(BaseModel):
    credential: str  # the ID token returned by Google Identity Services on the frontend


@app.post("/api/auth/google")
def google_auth(body: GoogleAuthBody):
    """Verifies the Google ID token server-side (never trust a client-sent
    email/name directly) and creates the user on first sign-in."""
    try:
        user_id, username = google_login(body.credential)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    token = create_session(user_id)
    return {"token": token, "username": username, "user_id": user_id}


@app.post("/api/auth/logout")
def logout(authorization: str = Header(default=None)):
    """Invalidates the current session token server-side. Frontend should
    also clear its own localStorage — this just makes the old token unusable
    even if it leaked (e.g. stayed in browser history)."""
    if authorization and authorization.startswith("Bearer "):
        revoke_session(authorization.removeprefix("Bearer ").strip())
    return {"ok": True}


# ------------------------------------------------------------------ chats --
def _time_group(ts: float) -> str:
    d = datetime.fromtimestamp(ts, tz=timezone.utc).date()
    today = datetime.now(tz=timezone.utc).date()
    delta = (today - d).days
    if delta <= 0:
        return "Today"
    if delta <= 7:
        return "Previous 7 days"
    return "Older"


@app.get("/api/chats")
def list_chats(user=Depends(get_current_user)):
    # One query instead of 1 + N (a separate "last message" lookup per chat) —
    # the correlated subquery lets SQLite do this in a single pass using the
    # idx_messages_chat_created index instead of a Python-side loop of
    # round-trips.
    with get_conn() as conn:
        chats = conn.execute(
            """SELECT c.id, c.title, c.updated_at,
                      (SELECT content FROM messages m
                       WHERE m.chat_id = c.id
                       ORDER BY m.created_at DESC LIMIT 1) AS preview
               FROM chats c
               WHERE c.user_id = ?
               ORDER BY c.updated_at DESC""",
            (user["id"],),
        ).fetchall()
    return [
        {
            "id": c["id"],
            "title": c["title"],
            "preview": (c["preview"][:80] if c["preview"] else ""),
            "group": _time_group(c["updated_at"]),
        }
        for c in chats
    ]


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str, user=Depends(get_current_user)):
    with get_conn() as conn:
        chat = conn.execute(
            "SELECT * FROM chats WHERE id = ? AND user_id = ?", (chat_id, user["id"])
        ).fetchone()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        msgs = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY created_at ASC", (chat_id,)
        ).fetchall()
    return {"id": chat["id"], "title": chat["title"], "messages": [dict(m) for m in msgs]}


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, user=Depends(get_current_user)):
    with get_conn() as conn:
        chat = conn.execute(
            "SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, user["id"])
        ).fetchone()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    return {"ok": True}


_background_tasks: set[asyncio.Task] = set()

_ATTACHMENT_NOTE_RE = re.compile(
    r"\[Attached (?:image|audio|contract): (.+?) — processed into \d+ chunk\(s\)[^\]]*\]"
)
_PIPELINE_NOTE_RE = re.compile(r"\[Note:.*?\]")


def _derive_title_seed(message: str) -> str:
    """Best-effort short seed for naming a new chat. Prefers whatever text
    the user actually typed; if they only attached a file with no message
    (or the typed text is blank), falls back to that file's name instead of
    leaving the raw "[Attached ...]" bracket note as the title."""
    text_only = _PIPELINE_NOTE_RE.sub("", _ATTACHMENT_NOTE_RE.sub("", message)).strip()
    if text_only:
        return text_only
    match = _ATTACHMENT_NOTE_RE.search(message)
    if match:
        filename = match.group(1).strip()
        return filename.rsplit(".", 1)[0] or filename
    return "New conversation"


async def _name_new_chat(chat_id: str, first_message: str) -> None:
    """Names a brand-new chat exactly once, from its very first message, with
    a short title written by the LLM. Runs as a background task so the first
    reply isn't delayed by a second model round-trip; later turns never
    touch the title again. Falls back to the message prefix if the model
    call fails."""
    first_message = first_message.strip()
    if not first_message:
        return  # nothing to name from (e.g. attachment-only send) — keep the placeholder
    try:
        title = await generate_chat_title(first_message)
    except Exception as e:
        logger.warning(f"chat title generation failed for {chat_id}: {e}")
        return
    if not title:
        return
    try:
        with get_conn() as conn:
            conn.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                (title, now(), chat_id),
            )
    except Exception as e:
        logger.warning(f"couldn't persist generated chat title for {chat_id}: {e}")


# -------------------------------------------------------------- chat send --
@app.post("/api/chat")
async def chat(body: ChatBody, user=Depends(get_current_user)):
    """
    Streams plain-text chunks back to the client as Groq generates them.
    The very first chunk sent is a JSON line: {"chat_id": "...", "pipeline_warnings": [...], "citations": [...]}
    so the frontend knows which chat this belongs to (new chats get an id
    here), which pipeline stages degraded (missing models), and which
    retrieved chunks the answer is grounded in (id/excerpt/score — same
    ranking the LLM's system prompt was built from, so the frontend can
    show "based on clause X" instead of an unverifiable LLM claim).
    Every following chunk is raw assistant text.

    If `document_id` is set, the AI Orchestrator runs retrieval + NER +
    clause/risk classification + compliance checks on that document first
    and builds a grounded system prompt (see orchestrator.py). Otherwise
    it's a plain chat turn with just conversation history.
    """
    chat_id = body.chat_id
    is_new_chat = False
    history: list[dict] = []

    with get_conn() as conn:
        if chat_id and not body.temporary:
            chat_row = conn.execute(
                "SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, user["id"])
            ).fetchone()
            if not chat_row:
                raise HTTPException(status_code=404, detail="Chat not found")
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
                (chat_id,),
            ).fetchall()
            history = [
                {"role": "assistant" if r["role"] == "bot" else "user", "content": r["content"]}
                for r in rows
            ]
        elif not body.temporary:
            is_new_chat = True
            chat_id = new_id()
            # Placeholder shown until the background LLM title lands. Uses
            # the same seed-derivation as below so an attachment-only send
            # gets the filename instead of the raw bracket note.
            title = _derive_title_seed(body.message)[:60] or "New conversation"
            conn.execute(
                "INSERT INTO chats (id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
                (chat_id, user["id"], title, now(), now()),
            )
        if not body.temporary:
            conn.execute(
                "INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?,?,?,?,?)",
                (new_id(), chat_id, "user", body.message, now()),
            )

    if is_new_chat:
        # Same seed used for the placeholder title above — for an
        # attachment-only send this is the filename, not the raw note, so
        # the LLM has something sensible to build a title from.
        title_source = _derive_title_seed(body.message)
        task = asyncio.create_task(_name_new_chat(chat_id, title_source))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    if body.temporary:
        # Trust this only for temporary chats — it's never written to the DB,
        # just replayed straight back into this one LLM call.
        history = [
            {"role": h.get("role"), "content": h.get("content", "")}
            for h in (body.history or [])
            if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content")
        ]

    # build_answer_context is synchronous and, when document_id is set, does
    # real CPU/model work (embedding, retrieval, rerank, NER, classification,
    # PII, compliance) — potentially seconds of it. Calling it directly here
    # would block the whole asyncio event loop for that entire duration (no
    # other request, including a concurrent /api/health, could be served),
    # and the response can't start streaming until it returns either way.
    # Running it in a worker thread frees the event loop for other requests
    # in the meantime. (When there's no document_id it returns almost
    # instantly regardless — this mainly matters for document-grounded chats.)
    answer_ctx = await asyncio.to_thread(
        build_answer_context,
        body.document_id,
        body.message,
        language=body.language,
    )

    async def generator():
        yield json.dumps(
            {
                "chat_id": chat_id,
                "pipeline_warnings": answer_ctx.warnings,
                "citations": answer_ctx.citations,
                "stage_timings_ms": answer_ctx.stage_timings_ms,
                "total_latency_ms": answer_ctx.total_latency_ms,
                "quality_score": answer_ctx.quality_score,
            }
        ) + "\n"
        full_reply = []
        try:
            async for piece in stream_chat(history, body.message, system_prompt=answer_ctx.system_prompt):
                full_reply.append(piece)
                yield piece
        finally:
            # Runs even if the client stopped the generation partway through
            # (StreamingResponse cancels this generator on disconnect) — save
            # whatever was produced so far instead of losing it silently.
            if not body.temporary:
                with get_conn() as conn:
                    conn.execute(
                        "INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?,?,?,?,?)",
                        (new_id(), chat_id, "bot", "".join(full_reply), now()),
                    )
                    conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now(), chat_id))

    return StreamingResponse(generator(), media_type="text/plain")


# --------------------------------------------------------- document intake --
def _extract_text_pages(dest: Path, content_type: str, filename: str) -> tuple[list[str], str]:
    """Returns (pages_of_text, source_kind). Raises a friendly HTTP error for
    formats that need a model that isn't set up yet, instead of a raw
    stack trace bubbling up to the client."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf" or content_type == "application/pdf":
        pages = extract_pdf(str(dest))
        if all(p.is_likely_scanned for p in pages):
            try:
                import fitz

                from services.parsing.ocr import ocr_pdf_page

                texts = []
                with fitz.open(str(dest)) as doc:
                    for i, page in enumerate(doc):
                        img_path = str(dest) + f".page{i}.png"
                        page.get_pixmap(dpi=200).save(img_path)
                        texts.append(ocr_pdf_page(img_path))
                return texts, "pdf_scanned"
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"This looks like a scanned PDF with no text layer, and OCR isn't working ({e}). "
                    "Install rapidocr-onnxruntime (see services/parsing/ocr.py) or set GROQ_API_KEY for the "
                    "hosted vision fallback — or upload a text-based PDF.",
                )
        return [p.text for p in pages], "pdf"

    if suffix == ".docx" or content_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ):
        return [extract_docx_full_text(str(dest))], "docx"

    if content_type.startswith("image/"):
        try:
            from services.parsing.ocr import ocr_image_full_text

            return [ocr_image_full_text(str(dest))], "image"
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"OCR isn't working ({e}). Install rapidocr-onnxruntime — see services/parsing/ocr.py — "
                "or set GROQ_API_KEY for the hosted vision fallback.",
            )

    if content_type.startswith("audio/"):
        try:
            from services.parsing.speech_to_text import transcribe

            text, _segments = transcribe(str(dest))
            return [text], "audio"
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Speech-to-text isn't set up yet ({e}). Install faster-whisper — see services/parsing/speech_to_text.py.",
            )

    raise HTTPException(status_code=415, detail=f"Unsupported file type: {content_type or suffix}")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), user=Depends(get_current_user)):
    """
    Saves the file, extracts text with the matching parser (PyMuPDF / python-docx
    / rapidocr-onnxruntime / faster-whisper), then runs the full orchestrator pipeline
    (chunk -> embed -> index -> classify -> compliance check)
    and returns a `document_id` you can pass to POST /api/chat.
    """
    content_type = (file.content_type or "").lower()
    if content_type.startswith("image/"):
        kind, folder = "image", "images"
    elif content_type.startswith("audio/"):
        kind, folder = "audio", "audio"
    else:
        kind, folder = "contract", "contracts"

    file_id = new_id()
    dest = STORAGE / folder / f"{file_id}_{file.filename}"
    dest.write_bytes(await file.read())

    with get_conn() as conn:
        conn.execute(
            """INSERT INTO uploaded_files
               (id, user_id, chat_id, kind, original_name, stored_path, extracted_text, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (file_id, user["id"], None, kind, file.filename, str(dest), None, now()),
        )

    pages, source_kind = _extract_text_pages(dest, content_type, file.filename)

    document_id = new_id()
    result = process_document(document_id, pages)

    with get_conn() as conn:
        conn.execute(
            """INSERT INTO documents
               (id, user_id, file_id, title, source_kind,
                num_chunks, compliance_gaps, processing_warnings, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                document_id,
                user["id"],
                file_id,
                file.filename,
                source_kind,
                result.num_chunks,
                json.dumps(result.compliance_gap_titles),
                json.dumps(result.warnings),
                now(),
            ),
        )

    return {
        "document_id": document_id,
        "file_id": file_id,
        "source_kind": source_kind,
        "num_chunks": result.num_chunks,
        "compliance_gaps": result.compliance_gap_titles,
        "pipeline_warnings": result.warnings,
        "stage_timings_ms": result.stage_timings_ms,
        "total_latency_ms": result.total_latency_ms,
    }


@app.post("/api/paste-text")
def paste_text(body: PasteTextBody, user=Depends(get_current_user)):
    """Same pipeline as /api/upload, for the 'Paste Text (Contract/T&C)'
    entry point in the architecture diagram — no file involved."""
    document_id = new_id()
    result = process_document(document_id, [body.text])

    with get_conn() as conn:
        conn.execute(
            """INSERT INTO documents
               (id, user_id, file_id, title, source_kind,
                num_chunks, compliance_gaps, processing_warnings, created_at)
               VALUES (?,?,NULL,?,?,?,?,?,?)""",
            (
                document_id,
                user["id"],
                body.title,
                "pasted_text",
                result.num_chunks,
                json.dumps(result.compliance_gap_titles),
                json.dumps(result.warnings),
                now(),
            ),
        )

    return {
        "document_id": document_id,
        "num_chunks": result.num_chunks,
        "compliance_gaps": result.compliance_gap_titles,
        "pipeline_warnings": result.warnings,
        "stage_timings_ms": result.stage_timings_ms,
        "total_latency_ms": result.total_latency_ms,
    }


@app.get("/api/documents")
def list_documents(user=Depends(get_current_user)):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, source_kind, num_chunks, compliance_gaps, created_at "
            "FROM documents WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],),
        ).fetchall()
    return [{**dict(r), "compliance_gaps": json.loads(r["compliance_gaps"] or "[]")} for r in rows]


@app.get("/api/documents/{document_id}/report")
def compliance_report(document_id: str, user=Depends(get_current_user)):
    """A fresh, full-document GDPR/CCPA/general compliance report — not
    limited to whatever chunks a chat question happened to retrieve."""
    with get_conn() as conn:
        doc = conn.execute(
            "SELECT * FROM documents WHERE id = ? AND user_id = ?", (document_id, user["id"])
        ).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        file_row = None
        if doc["file_id"]:
            file_row = conn.execute("SELECT * FROM uploaded_files WHERE id = ?", (doc["file_id"],)).fetchone()

    # Re-extract text for the report rather than storing full text twice —
    # simplest correct option for a local single-user app. _extract_text_pages
    # branches on content_type (not just file extension) for images/audio, so
    # reconstruct a stand-in content_type from the stored `kind` — otherwise
    # every image/audio-sourced document 415s here even though the original
    # upload succeeded via OCR/speech-to-text.
    if file_row:
        content_type_hint = {"image": "image/*", "audio": "audio/*"}.get(file_row["kind"], "")
        pages, _ = _extract_text_pages(Path(file_row["stored_path"]), content_type_hint, file_row["original_name"])
        full_text = "\n\n".join(pages)
    else:
        full_text = ""  # pasted-text documents: extend to store the raw text if you want reports for these too

    findings = check_compliance(full_text)
    return {
        "document_id": document_id,
        "title": doc["title"],
        "findings": [f.__dict__ for f in findings],
    }


@app.get("/api/documents/{document_id}/obligations")
def document_obligations(document_id: str, user=Depends(get_current_user)):
    """Key dates & deadlines in the document — renewal notice windows,
    termination notice periods, payment due dates, expiration — surfaced as
    a structured, sortable list instead of something the user has to hunt
    for by reading the whole contract. See services/legal_ai/obligations.py."""
    with get_conn() as conn:
        doc = conn.execute(
            "SELECT * FROM documents WHERE id = ? AND user_id = ?", (document_id, user["id"])
        ).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        file_row = None
        if doc["file_id"]:
            file_row = conn.execute("SELECT * FROM uploaded_files WHERE id = ?", (doc["file_id"],)).fetchone()

    if file_row:
        content_type_hint = {"image": "image/*", "audio": "audio/*"}.get(file_row["kind"], "")
        pages, _ = _extract_text_pages(Path(file_row["stored_path"]), content_type_hint, file_row["original_name"])
        full_text = "\n\n".join(pages)
    else:
        full_text = ""  # pasted-text documents: extend to store the raw text if you want this for these too

    obligations = extract_obligations(full_text)

    return {
        "document_id": document_id,
        "title": doc["title"],
        "obligations": [
            {
                "type": o.obligation_type,
                "label": o.label,
                "date_text": o.raw_text,
                "parsed_date": o.parsed_date,
                "context": o.context,
                "confidence": o.confidence,
            }
            for o in obligations
        ],
    }


@app.get("/api/documents/{document_id}/download")
def download_document(document_id: str, user=Depends(get_current_user)):
    """Serves back the original uploaded file (PDF/DOCX/image/audio) for a
    document. Without this, the frontend could only offer a "download" of
    an attachment for as long as the in-browser blob URL from the original
    upload was still alive — i.e. it broke the moment the tab was closed,
    the page reloaded, or a different chat was opened. This lets a file
    sent in any past conversation be fetched back at any time, as long as
    it's still on disk in storage/."""
    with get_conn() as conn:
        doc = conn.execute(
            "SELECT * FROM documents WHERE id = ? AND user_id = ?", (document_id, user["id"])
        ).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        file_row = None
        if doc["file_id"]:
            file_row = conn.execute("SELECT * FROM uploaded_files WHERE id = ?", (doc["file_id"],)).fetchone()

    if not file_row:
        # Pasted-text documents have no underlying file to hand back.
        raise HTTPException(status_code=404, detail="This document has no original file (it was pasted text)")

    path = Path(file_row["stored_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="The original file is no longer on disk")

    media_type = mimetypes.guess_type(file_row["original_name"])[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=file_row["original_name"])


@app.get("/api/health")
def health():
    return {"ok": True}
