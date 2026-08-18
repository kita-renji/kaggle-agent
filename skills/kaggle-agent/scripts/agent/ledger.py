# SPDX-License-Identifier: MIT
"""Workspace paths, cross-process locking, and append-only JSONL ledgers.

Every durable fact the loop relies on is an append-only JSON line. Nothing is
ever mutated in place, so a tick that dies halfway cannot corrupt state and two
concurrent competition loops cannot lose each other's writes.

Paths hang off ``runtime.find_project_root()``, which honours ``PROJECT_ROOT``.
That single indirection is what lets the whole test suite run against a
``tmp_path`` workspace with no monkeypatching.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from runtime import find_project_root

logger = logging.getLogger(__name__)

LOCK_TIMEOUT_SECONDS = 10.0
LOCK_STALE_SECONDS = 60.0
_LOCK_POLL_INITIAL = 0.01
_LOCK_POLL_MAX = 0.25


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def project_root() -> Path:
    return find_project_root()


def competitions_root() -> Path:
    return project_root() / "competitions"


def competition_dir(slug: str) -> Path:
    return competitions_root() / slug


def mission_path(slug: str) -> Path:
    return competition_dir(slug) / "MISSION.md"


def state_md_path(slug: str) -> Path:
    return competition_dir(slug) / "STATE.md"


def backlog_path(slug: str) -> Path:
    return competition_dir(slug) / "backlog.md"


def journal_path(slug: str) -> Path:
    return competition_dir(slug) / "journal.md"


def runs_ledger_path(slug: str) -> Path:
    return competition_dir(slug) / "runs.jsonl"


def budget_ledger_path(slug: str) -> Path:
    return competition_dir(slug) / "budget.jsonl"


def halt_path(slug: str) -> Path:
    return competition_dir(slug) / "HALT"


def research_dir(slug: str) -> Path:
    return competition_dir(slug) / "research"


def kernels_dir(slug: str) -> Path:
    return competition_dir(slug) / "kernels"


def agent_data_dir() -> Path:
    return project_root() / "data" / "agent"


def gpu_ledger_path() -> Path:
    """Global, shared by every competition loop on this Kaggle account."""
    return agent_data_dir() / "gpu_usage.jsonl"


def lock_dir() -> Path:
    return agent_data_dir() / "locks"


def tick_state_path(slug: str) -> Path:
    return agent_data_dir() / "ticks" / f"{slug}.json"


def run_output_dir(run_id: str) -> Path:
    return agent_data_dir() / "runs" / run_id


def kernel_lock_name(kernel_slug: str) -> str:
    """Lock name for a Kaggle kernel ref (``owner/slug``)."""
    return "kernel--" + kernel_slug.replace("/", "--")


class SlugError(RuntimeError):
    """The competition slug could not be determined from context."""


def resolve_slug(explicit: str | None = None, *, cwd: Path | None = None) -> str:
    """Figure out which competition we are working on.

    An explicit argument (slug or Kaggle URL) always wins. Otherwise the
    working directory names it: inside ``competitions/<slug>/`` that is the
    slug, and anywhere else the directory's own name is used — which is what
    makes a per-competition directory just work without repeating yourself.
    """
    if explicit:
        from runtime import competition_slug

        return competition_slug(explicit)

    here = Path(cwd or Path.cwd()).resolve()
    competitions = competitions_root()
    try:
        relative = here.relative_to(competitions)
    except ValueError:
        pass
    else:
        if relative.parts:
            return relative.parts[0]
        raise SlugError(
            f"{here} is the competitions/ directory itself, not a competition. "
            "Name one, or run from inside its folder."
        )

    if here == project_root():
        raise SlugError(
            f"No competition given, and {here} is the project root. "
            "Name one (e.g. `titanic`) or run from competitions/<slug>/."
        )
    return here.name


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp(moment: datetime | None = None) -> str:
    return (moment or utc_now()).isoformat(timespec="seconds")


def new_run_id(moment: datetime | None = None) -> str:
    return f"r-{(moment or utc_now()):%Y%m%dT%H%MZ}-{secrets.token_hex(2)}"


def session_id() -> str:
    """Stable per-process id, overridable so a /loop session can name itself."""
    explicit = os.environ.get("KAGGLE_AGENT_SESSION")
    if explicit:
        return explicit
    return f"pid{os.getpid()}-{uuid.uuid4().hex[:6]}"


# --------------------------------------------------------------------------
# Locking
# --------------------------------------------------------------------------

class LockTimeout(RuntimeError):
    """Raised when a lock could not be acquired within the timeout."""


def _lock_path(name: str) -> Path:
    return lock_dir() / f"{name}.lock"


def _lock_payload(**extra) -> bytes:
    payload = {"pid": os.getpid(), "session": session_id(), "created_at": utc_stamp(), **extra}
    return (json.dumps(payload) + "\n").encode("utf-8")


def read_lock(name: str) -> dict | None:
    """Return the lock holder's metadata, or None when the lock is free."""
    path = _lock_path(name)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _reclaim_if_stale(path: Path, stale_after: float) -> bool:
    """Delete an abandoned lock. Returns True when one was reclaimed."""
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    if age < stale_after:
        return False
    logger.warning("reclaiming stale lock %s (age %.0fs)", path.name, age)
    try:
        path.unlink()
    except OSError:
        return False
    return True


def acquire_lock(
    name: str,
    *,
    timeout: float = LOCK_TIMEOUT_SECONDS,
    stale_after: float = LOCK_STALE_SECONDS,
    **payload,
) -> Path:
    """Take a named lock, spinning with backoff. Raises LockTimeout on failure.

    Used directly (rather than via ``file_lock``) for the kernel lock, which is
    deliberately held across ticks — from the push that starts a run until the
    tick that collects its output — so no second process can push the same
    kernel while it is executing on Kaggle.
    """
    path = _lock_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    delay = _LOCK_POLL_INITIAL
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _reclaim_if_stale(path, stale_after):
                continue
            if time.monotonic() >= deadline:
                holder = read_lock(name) or {}
                raise LockTimeout(
                    f"lock {name!r} held by session {holder.get('session', '?')} "
                    f"(pid {holder.get('pid', '?')}, since {holder.get('created_at', '?')})"
                )
            time.sleep(delay)
            delay = min(delay * 2, _LOCK_POLL_MAX)
            continue
        try:
            os.write(fd, _lock_payload(**payload))
        finally:
            os.close(fd)
        return path


def release_lock(name: str) -> bool:
    """Drop a named lock. Returns False when it was already gone."""
    try:
        _lock_path(name).unlink()
        return True
    except OSError:
        return False


@contextmanager
def file_lock(
    name: str,
    *,
    timeout: float = LOCK_TIMEOUT_SECONDS,
    stale_after: float = LOCK_STALE_SECONDS,
    **payload,
):
    """Scoped lock for a short critical section, released on the way out."""
    acquire_lock(name, timeout=timeout, stale_after=stale_after, **payload)
    try:
        yield
    finally:
        release_lock(name)


# --------------------------------------------------------------------------
# JSONL ledgers
# --------------------------------------------------------------------------

def append_jsonl(path: Path, record: dict, *, strict: bool = False) -> bool:
    """Append one record under a lock. Returns True when it reached disk.

    Best-effort by default, matching ``submission_log.append_record``: an audit
    write must never abort an in-flight Kaggle operation. Pass ``strict=True``
    for records that a later decision depends on being durable — chiefly the
    submission reservation, where a silently dropped write would let the agent
    spend a slot it has already spent.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            **record,
            # Stamped last so the ledger's own clock is authoritative and a
            # caller cannot backdate a record.
            "logged_at": utc_stamp(),
        }
        line = json.dumps(entry, default=str) + "\n"
        with file_lock(f"jsonl--{path.name}"):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        return True
    except Exception as exc:  # noqa: BLE001 — see docstring
        if strict:
            raise
        logger.warning("could not append to %s: %s", path, exc)
        return False


def read_jsonl(path: Path) -> list[dict]:
    """Read records in file order, skipping malformed lines with a warning."""
    if not path.exists():
        return []
    records: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("skipping malformed line %d in %s", lineno, path)
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload) -> None:
    """Write JSON atomically, so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_text(path: Path, text: str) -> None:
    """Append to a markdown file (journal), creating it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text if text.endswith("\n") else text + "\n")
