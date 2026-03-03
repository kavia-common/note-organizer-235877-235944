from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# --- App setup ---

openapi_tags = [
    {"name": "Health", "description": "Service health and diagnostics."},
    {"name": "Notes", "description": "CRUD operations for notes."},
    {"name": "Tags", "description": "CRUD operations for tags and tag listing."},
    {"name": "Search", "description": "Search and filtering across notes (text + tags)."},
]

app = FastAPI(
    title="Notes Backend API",
    description=(
        "REST API for a notes app.\n\n"
        "Backed by SQLite with normalized tags via a many-to-many join table (note_tags)."
    ),
    version="0.1.0",
    openapi_tags=openapi_tags,
)

# CORS: allow React frontend (default permissive; can be restricted via env)
# NOTE: For production, set CORS_ALLOW_ORIGINS to a comma-separated list of allowed origins.
allow_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*")
allow_origins = (
    ["*"]
    if allow_origins_env.strip() == "*"
    else [o.strip() for o in allow_origins_env.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database helpers ---

ENV_SQLITE_DB = "SQLITE_DB"


def _get_db_path() -> str:
    """
    Return absolute path to the SQLite database.

    The database container exposes its DB path via SQLITE_DB environment variable.
    """
    db_path = os.getenv(ENV_SQLITE_DB)
    if not db_path:
        raise RuntimeError(
            f"Missing required environment variable {ENV_SQLITE_DB}. "
            "Ask orchestrator to provide it from the database container."
        )
    return db_path


def _connect() -> sqlite3.Connection:
    """Create a SQLite connection with project defaults."""
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a sqlite Row to a plain dict."""
    return {k: row[k] for k in row.keys()}


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse SQLite timestamp values that are stored as text-ish into datetime."""
    if value is None:
        return None
    # SQLite CURRENT_TIMESTAMP returns "YYYY-MM-DD HH:MM:SS"
    # Keep this tolerant in case of ISO strings.
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        # Best-effort fallback: return None rather than failing response rendering.
        return None


def _ensure_schema_exists(conn: sqlite3.Connection) -> None:
    """
    Ensure required tables exist.

    This does not create schema; it just validates presence so we can return a clear error.
    """
    required = {"notes", "tags", "note_tags"}
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    tables = {r["name"] for r in cur.fetchall()}
    missing = sorted(required - tables)
    if missing:
        raise RuntimeError(
            "Database schema is missing required tables: "
            + ", ".join(missing)
            + ". Run the database init script in the database container."
        )


# --- Pydantic models ---

class TagBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, description="Tag name (unique).")

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("Tag name cannot be blank.")
        return v2


class TagCreate(TagBase):
    """Request body for creating a tag."""


class TagUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, description="New tag name.")

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("Tag name cannot be blank.")
        return v2


class TagOut(BaseModel):
    id: int = Field(..., description="Tag id.")
    name: str = Field(..., description="Tag name.")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp.")

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TagOut":
        d = _row_to_dict(row)
        return cls(
            id=int(d["id"]),
            name=str(d["name"]),
            created_at=_parse_iso_datetime(d.get("created_at")),
        )


class NoteBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Note title.")
    content: str = Field("", max_length=20000, description="Note content.")
    is_archived: bool = Field(False, description="Whether the note is archived.")
    tags: List[str] = Field(
        default_factory=list,
        description="List of tag names to attach to the note (created if missing).",
    )

    @field_validator("title")
    @classmethod
    def normalize_title(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("Title cannot be blank.")
        return v2

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, v: List[str]) -> List[str]:
        cleaned: List[str] = []
        for t in v:
            if t is None:
                continue
            t2 = str(t).strip()
            if not t2:
                continue
            cleaned.append(t2)
        # de-dupe preserving order (case-sensitive; frontend can control casing)
        seen = set()
        out: List[str] = []
        for t in cleaned:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out


class NoteCreate(NoteBase):
    """Request body for creating a note."""


class NoteUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200, description="Note title.")
    content: Optional[str] = Field(None, max_length=20000, description="Note content.")
    is_archived: Optional[bool] = Field(None, description="Whether the note is archived.")
    tags: Optional[List[str]] = Field(
        None,
        description="Replace tags with this list of tag names (created if missing).",
    )

    @field_validator("title")
    @classmethod
    def normalize_title(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v2 = v.strip()
        if not v2:
            raise ValueError("Title cannot be blank.")
        return v2

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        cleaned: List[str] = []
        for t in v:
            if t is None:
                continue
            t2 = str(t).strip()
            if not t2:
                continue
            cleaned.append(t2)
        seen = set()
        out: List[str] = []
        for t in cleaned:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out


class NoteOut(BaseModel):
    id: int = Field(..., description="Note id.")
    title: str = Field(..., description="Note title.")
    content: str = Field(..., description="Note content.")
    is_archived: bool = Field(..., description="Archived flag.")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp.")
    updated_at: Optional[datetime] = Field(None, description="Update timestamp.")
    tags: List[TagOut] = Field(default_factory=list, description="Tags attached to this note.")


class NoteListItem(BaseModel):
    """Smaller note shape for list endpoints."""
    id: int
    title: str
    content: str
    is_archived: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list, description="Tag names.")


class NotesListResponse(BaseModel):
    items: List[NoteListItem]
    total: int


# --- Dependency ---

def get_db() -> sqlite3.Connection:
    """FastAPI dependency that yields a DB connection."""
    conn = _connect()
    try:
        _ensure_schema_exists(conn)
        yield conn
    finally:
        conn.close()


# --- Internal tag/note helpers ---

def _get_or_create_tag_ids(conn: sqlite3.Connection, tag_names: Sequence[str]) -> List[int]:
    """Resolve tag names to ids, creating tags that do not exist."""
    ids: List[int] = []
    for name in tag_names:
        row = conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()
        if row:
            ids.append(int(row["id"]))
            continue
        try:
            cur = conn.execute("INSERT INTO tags (name) VALUES (?)", (name,))
            ids.append(int(cur.lastrowid))
        except sqlite3.IntegrityError:
            # Someone else created it; re-select.
            row2 = conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()
            if not row2:
                raise
            ids.append(int(row2["id"]))
    return ids


def _set_note_tags(conn: sqlite3.Connection, note_id: int, tag_names: Sequence[str]) -> None:
    """Replace tags for a note with the given tag names."""
    tag_ids = _get_or_create_tag_ids(conn, tag_names)
    conn.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
    for tid in tag_ids:
        conn.execute(
            "INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)",
            (note_id, tid),
        )


def _fetch_tags_for_note(conn: sqlite3.Connection, note_id: int) -> List[TagOut]:
    cur = conn.execute(
        """
        SELECT t.id, t.name, t.created_at
        FROM tags t
        JOIN note_tags nt ON nt.tag_id = t.id
        WHERE nt.note_id = ?
        ORDER BY t.name ASC
        """,
        (note_id,),
    )
    return [TagOut.from_row(r) for r in cur.fetchall()]


def _note_exists(conn: sqlite3.Connection, note_id: int) -> bool:
    row = conn.execute("SELECT 1 FROM notes WHERE id = ?", (note_id,)).fetchone()
    return row is not None


def _tag_exists(conn: sqlite3.Connection, tag_id: int) -> bool:
    row = conn.execute("SELECT 1 FROM tags WHERE id = ?", (tag_id,)).fetchone()
    return row is not None


def _fetch_note(conn: sqlite3.Connection, note_id: int) -> NoteOut:
    row = conn.execute(
        "SELECT id, title, content, is_archived, created_at, updated_at FROM notes WHERE id = ?",
        (note_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Note not found.")
    d = _row_to_dict(row)
    return NoteOut(
        id=int(d["id"]),
        title=str(d["title"]),
        content=str(d["content"]),
        is_archived=bool(d["is_archived"]),
        created_at=_parse_iso_datetime(d.get("created_at")),
        updated_at=_parse_iso_datetime(d.get("updated_at")),
        tags=_fetch_tags_for_note(conn, note_id),
    )


# --- Routes ---

@app.get(
    "/",
    tags=["Health"],
    summary="Health check",
    description="Simple health check endpoint.",
    operation_id="health_check",
)
# PUBLIC_INTERFACE
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"message": "Healthy"}


@app.get(
    "/docs/help",
    tags=["Health"],
    summary="API usage help",
    description="Quick notes about configuration and how to use the API from the frontend.",
    operation_id="docs_help",
)
# PUBLIC_INTERFACE
def docs_help() -> Dict[str, Any]:
    """Return short usage notes for the API."""
    return {
        "database_env": ENV_SQLITE_DB,
        "cors_env": "CORS_ALLOW_ORIGINS (comma-separated) or '*'",
        "notes": {
            "list": "GET /notes?query=&tags=tag1&tags=tag2&archived=false&limit=50&offset=0",
            "create": "POST /notes",
            "get": "GET /notes/{id}",
            "update": "PATCH /notes/{id}",
            "delete": "DELETE /notes/{id}",
        },
        "tags": {
            "list": "GET /tags",
            "create": "POST /tags",
            "get": "GET /tags/{id}",
            "update": "PATCH /tags/{id}",
            "delete": "DELETE /tags/{id} (fails if used by notes)",
        },
    }


@app.get(
    "/tags",
    response_model=List[TagOut],
    tags=["Tags"],
    summary="List tags",
    description="List all tags, ordered by name.",
    operation_id="list_tags",
)
# PUBLIC_INTERFACE
def list_tags(db: sqlite3.Connection = Depends(get_db)) -> List[TagOut]:
    """List all tags."""
    cur = db.execute("SELECT id, name, created_at FROM tags ORDER BY name ASC")
    return [TagOut.from_row(r) for r in cur.fetchall()]


@app.post(
    "/tags",
    response_model=TagOut,
    status_code=status.HTTP_201_CREATED,
    tags=["Tags"],
    summary="Create tag",
    description="Create a new tag. Tag names must be unique.",
    operation_id="create_tag",
)
# PUBLIC_INTERFACE
def create_tag(payload: TagCreate, db: sqlite3.Connection = Depends(get_db)) -> TagOut:
    """Create a tag."""
    try:
        cur = db.execute("INSERT INTO tags (name) VALUES (?)", (payload.name,))
        db.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Tag name already exists.")
    row = db.execute(
        "SELECT id, name, created_at FROM tags WHERE id = ?", (int(cur.lastrowid),)
    ).fetchone()
    return TagOut.from_row(row)


@app.get(
    "/tags/{tag_id}",
    response_model=TagOut,
    tags=["Tags"],
    summary="Get tag",
    description="Fetch a tag by id.",
    operation_id="get_tag",
)
# PUBLIC_INTERFACE
def get_tag(tag_id: int, db: sqlite3.Connection = Depends(get_db)) -> TagOut:
    """Get a tag by id."""
    row = db.execute(
        "SELECT id, name, created_at FROM tags WHERE id = ?", (tag_id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Tag not found.")
    return TagOut.from_row(row)


@app.patch(
    "/tags/{tag_id}",
    response_model=TagOut,
    tags=["Tags"],
    summary="Update tag",
    description="Update a tag name (must remain unique).",
    operation_id="update_tag",
)
# PUBLIC_INTERFACE
def update_tag(tag_id: int, payload: TagUpdate, db: sqlite3.Connection = Depends(get_db)) -> TagOut:
    """Update a tag."""
    if not _tag_exists(db, tag_id):
        raise HTTPException(status_code=404, detail="Tag not found.")
    try:
        db.execute("UPDATE tags SET name = ? WHERE id = ?", (payload.name, tag_id))
        db.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Tag name already exists.")
    return get_tag(tag_id, db)


@app.delete(
    "/tags/{tag_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Tags"],
    summary="Delete tag",
    description=(
        "Delete a tag by id. Fails with 409 if the tag is still used by any note."
    ),
    operation_id="delete_tag",
)
# PUBLIC_INTERFACE
def delete_tag(tag_id: int, db: sqlite3.Connection = Depends(get_db)) -> Response:
    """Delete a tag, disallowing deletion if referenced by notes."""
    if not _tag_exists(db, tag_id):
        raise HTTPException(status_code=404, detail="Tag not found.")
    used = db.execute(
        "SELECT 1 FROM note_tags WHERE tag_id = ? LIMIT 1", (tag_id,)
    ).fetchone()
    if used:
        raise HTTPException(
            status_code=409, detail="Tag is in use by one or more notes."
        )
    db.execute("DELETE FROM tags WHERE id = ?", (tag_id,))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get(
    "/notes",
    response_model=NotesListResponse,
    tags=["Notes", "Search"],
    summary="List/search notes",
    description=(
        "List notes with optional filtering by text query and/or tags.\n\n"
        "- `query`: searches in title and content (case-insensitive LIKE).\n"
        "- `tags`: one or more tag names; note must match ALL provided tags.\n"
        "- `archived`: if provided, filters by archived state.\n"
        "- `limit/offset`: pagination.\n"
        "- `sort`: 'updated_desc' (default), 'created_desc', 'title_asc'."
    ),
    operation_id="list_notes",
)
# PUBLIC_INTERFACE
def list_notes(
    query: Optional[str] = Query(None, description="Search text for title/content."),
    tags: List[str] = Query(default_factory=list, description="Filter by tag names (AND)."),
    archived: Optional[bool] = Query(None, description="Filter by archived status."),
    limit: int = Query(50, ge=1, le=200, description="Max number of notes to return."),
    offset: int = Query(0, ge=0, description="Pagination offset."),
    sort: str = Query(
        "updated_desc",
        pattern="^(updated_desc|created_desc|title_asc)$",
        description="Sort order.",
    ),
    db: sqlite3.Connection = Depends(get_db),
) -> NotesListResponse:
    """List/search notes with optional query + tag filtering."""
    q = (query or "").strip()
    tag_names = [t.strip() for t in tags if t.strip()]

    where: List[str] = []
    params: List[Any] = []

    if archived is not None:
        where.append("n.is_archived = ?")
        params.append(1 if archived else 0)

    if q:
        where.append("(n.title LIKE ? OR n.content LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like])

    # Tag filtering: require ALL tags. Implement via join + GROUP BY + HAVING count distinct = N
    joins = ""
    having = ""
    if tag_names:
        joins = """
            JOIN note_tags nt ON nt.note_id = n.id
            JOIN tags t ON t.id = nt.tag_id
        """
        where.append("t.name IN (" + ",".join(["?"] * len(tag_names)) + ")")
        params.extend(tag_names)
        having = "HAVING COUNT(DISTINCT t.name) = ?"
        params.append(len(tag_names))

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    order_by = {
        "updated_desc": "n.updated_at DESC, n.id DESC",
        "created_desc": "n.created_at DESC, n.id DESC",
        "title_asc": "n.title ASC, n.id ASC",
    }[sort]

    # Total count (distinct notes)
    if tag_names:
        count_sql = f"""
            SELECT COUNT(*) AS c FROM (
                SELECT n.id
                FROM notes n
                {joins}
                {where_sql}
                GROUP BY n.id
                {having}
            ) sub
        """
        count_params = params[:]  # includes having parameter already
    else:
        count_sql = f"SELECT COUNT(*) AS c FROM notes n {where_sql}"
        count_params = params[:]

    total_row = db.execute(count_sql, tuple(count_params)).fetchone()
    total = int(total_row["c"]) if total_row else 0

    # Fetch ids (paged), then load each note summary + tags in a second query
    if tag_names:
        list_sql = f"""
            SELECT n.id, n.title, n.content, n.is_archived, n.created_at, n.updated_at
            FROM notes n
            {joins}
            {where_sql}
            GROUP BY n.id
            {having}
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
        """
        list_params = params[:] + [limit, offset]
    else:
        list_sql = f"""
            SELECT n.id, n.title, n.content, n.is_archived, n.created_at, n.updated_at
            FROM notes n
            {where_sql}
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
        """
        list_params = params[:] + [limit, offset]

    rows = db.execute(list_sql, tuple(list_params)).fetchall()

    note_ids = [int(r["id"]) for r in rows]
    tags_map: Dict[int, List[str]] = {nid: [] for nid in note_ids}
    if note_ids:
        placeholders = ",".join(["?"] * len(note_ids))
        tag_rows = db.execute(
            f"""
            SELECT nt.note_id AS note_id, t.name AS name
            FROM note_tags nt
            JOIN tags t ON t.id = nt.tag_id
            WHERE nt.note_id IN ({placeholders})
            ORDER BY t.name ASC
            """,
            tuple(note_ids),
        ).fetchall()
        for tr in tag_rows:
            tags_map[int(tr["note_id"])].append(str(tr["name"]))

    items: List[NoteListItem] = []
    for r in rows:
        d = _row_to_dict(r)
        nid = int(d["id"])
        items.append(
            NoteListItem(
                id=nid,
                title=str(d["title"]),
                content=str(d["content"]),
                is_archived=bool(d["is_archived"]),
                created_at=_parse_iso_datetime(d.get("created_at")),
                updated_at=_parse_iso_datetime(d.get("updated_at")),
                tags=tags_map.get(nid, []),
            )
        )

    return NotesListResponse(items=items, total=total)


@app.post(
    "/notes",
    response_model=NoteOut,
    status_code=status.HTTP_201_CREATED,
    tags=["Notes"],
    summary="Create note",
    description="Create a new note, optionally with tags (tags are created if missing).",
    operation_id="create_note",
)
# PUBLIC_INTERFACE
def create_note(payload: NoteCreate, db: sqlite3.Connection = Depends(get_db)) -> NoteOut:
    """Create a note and set its tags."""
    cur = db.execute(
        "INSERT INTO notes (title, content, is_archived) VALUES (?, ?, ?)",
        (payload.title, payload.content, 1 if payload.is_archived else 0),
    )
    note_id = int(cur.lastrowid)
    _set_note_tags(db, note_id, payload.tags)
    db.commit()
    return _fetch_note(db, note_id)


@app.get(
    "/notes/{note_id}",
    response_model=NoteOut,
    tags=["Notes"],
    summary="Get note",
    description="Fetch a note by id, including its tags.",
    operation_id="get_note",
)
# PUBLIC_INTERFACE
def get_note(note_id: int, db: sqlite3.Connection = Depends(get_db)) -> NoteOut:
    """Get a single note by id."""
    return _fetch_note(db, note_id)


@app.patch(
    "/notes/{note_id}",
    response_model=NoteOut,
    tags=["Notes"],
    summary="Update note",
    description="Update title/content/is_archived and/or replace tags list.",
    operation_id="update_note",
)
# PUBLIC_INTERFACE
def update_note(note_id: int, payload: NoteUpdate, db: sqlite3.Connection = Depends(get_db)) -> NoteOut:
    """Update an existing note and optionally replace its tags."""
    if not _note_exists(db, note_id):
        raise HTTPException(status_code=404, detail="Note not found.")

    fields: List[str] = []
    params: List[Any] = []

    if payload.title is not None:
        fields.append("title = ?")
        params.append(payload.title)
    if payload.content is not None:
        fields.append("content = ?")
        params.append(payload.content)
    if payload.is_archived is not None:
        fields.append("is_archived = ?")
        params.append(1 if payload.is_archived else 0)

    if fields:
        params.append(note_id)
        db.execute(f"UPDATE notes SET {', '.join(fields)} WHERE id = ?", tuple(params))

    if payload.tags is not None:
        _set_note_tags(db, note_id, payload.tags)

    db.commit()
    return _fetch_note(db, note_id)


@app.delete(
    "/notes/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Notes"],
    summary="Delete note",
    description="Delete a note by id (also deletes note_tags via FK cascade).",
    operation_id="delete_note",
)
# PUBLIC_INTERFACE
def delete_note(note_id: int, db: sqlite3.Connection = Depends(get_db)) -> Response:
    """Delete a note by id."""
    if not _note_exists(db, note_id):
        raise HTTPException(status_code=404, detail="Note not found.")
    db.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
