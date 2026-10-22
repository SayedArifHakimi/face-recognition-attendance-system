"""Transactional SQLite database layer for attendance events and audit logging."""
import csv
import io
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Generator

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "attendance.db"


@contextmanager
def db_session(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite connection with automatic close and transaction support."""
    target_path = db_path if db_path is not None else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Create tables and indexes if they do not exist."""
    with db_session(db_path) as conn:
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS attendance_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    person_name TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK(event_type IN ('CHECK_IN', 'CHECK_OUT')),
                    event_time TEXT NOT NULL,
                    device_id TEXT NOT NULL DEFAULT 'KIOSK-01',
                    similarity_score REAL,
                    status TEXT NOT NULL DEFAULT 'RECORDED' CHECK(status IN ('RECORDED', 'CORRECTED', 'VOID')),
                    source TEXT NOT NULL DEFAULT 'FACE_ID' CHECK(source IN ('FACE_ID', 'MANUAL')),
                    notes TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_person_time 
                    ON attendance_events(person_name, event_time);
                CREATE INDEX IF NOT EXISTS idx_events_time 
                    ON attendance_events(event_time);
                CREATE INDEX IF NOT EXISTS idx_events_status 
                    ON attendance_events(status);

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('CREATED', 'CORRECTED', 'VOIDED', 'MANUAL_ENTRY')),
                    actor TEXT NOT NULL DEFAULT 'SYSTEM',
                    reason TEXT NOT NULL,
                    performed_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES attendance_events(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_audit_event 
                    ON audit_log(event_id);
            """)


def record_event(
    person_name: str,
    event_type: str,
    similarity_score: Optional[float] = None,
    device_id: str = "KIOSK-01",
    source: str = "FACE_ID",
    notes: Optional[str] = None,
    event_time: Optional[str] = None,
    event_id: Optional[str] = None,
    actor: str = "SYSTEM",
    cooldown_seconds: int = 60,
    db_path: Optional[Path] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Record an attendance event with idempotency, duplicate cooldown prevention, and audit trail.
    Returns (success, message, event_dict).
    """
    clean_name = person_name.strip()
    if not clean_name or clean_name.casefold() == "unknown":
        return False, "Cannot record attendance for unknown or empty identity", None

    clean_type = event_type.strip().upper()
    if clean_type not in ("CHECK_IN", "CHECK_OUT"):
        return False, f"Invalid event type: {event_type}. Must be CHECK_IN or CHECK_OUT", None

    now_iso = datetime.now(timezone.utc).astimezone().isoformat()
    record_time = event_time or now_iso
    record_event_id = event_id or str(uuid.uuid4())

    with db_session(db_path) as conn:
        with conn:
            cursor = conn.cursor()

            # Idempotency check: if event_id already exists, return existing record
            cursor.execute("SELECT * FROM attendance_events WHERE event_id = ?", (record_event_id,))
            existing = cursor.fetchone()
            if existing:
                return True, "Event already recorded (idempotent)", dict(existing)

            # Duplicate cooldown check: prevent rapid repeat events of same type for the same person
            if cooldown_seconds > 0:
                cursor.execute("""
                    SELECT event_time, status FROM attendance_events
                    WHERE person_name = ? AND event_type = ? AND status = 'RECORDED'
                    ORDER BY event_time DESC LIMIT 1
                """, (clean_name, clean_type))
                last = cursor.fetchone()
                if last:
                    try:
                        last_time = datetime.fromisoformat(last["event_time"])
                        current_time = datetime.fromisoformat(record_time)
                        elapsed = abs((current_time - last_time).total_seconds())
                        if elapsed < cooldown_seconds:
                            return False, (
                                f"Cooldown active: {clean_name} already logged {clean_type} "
                                f"{int(elapsed)}s ago (minimum interval: {cooldown_seconds}s)"
                            ), None
                    except ValueError:
                        pass

            # Insert event record
            cursor.execute("""
                INSERT INTO attendance_events (
                    event_id, person_name, event_type, event_time,
                    device_id, similarity_score, status, source, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'RECORDED', ?, ?, ?)
            """, (
                record_event_id, clean_name, clean_type, record_time,
                device_id, similarity_score, source, notes, now_iso
            ))
            row_id = cursor.lastrowid

            # Insert audit trail
            action_type = "MANUAL_ENTRY" if source == "MANUAL" else "CREATED"
            audit_reason = notes or (f"Explicit {clean_type} logged at {device_id}")
            cursor.execute("""
                INSERT INTO audit_log (event_id, action, actor, reason, performed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (row_id, action_type, actor, audit_reason, now_iso))

            cursor.execute("SELECT * FROM attendance_events WHERE id = ?", (row_id,))
            saved = cursor.fetchone()
            return True, f"Recorded {clean_type} for {clean_name}", dict(saved)


def get_events(
    date_str: Optional[str] = None,
    person_name: Optional[str] = None,
    event_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Retrieve filtered attendance events with ordering by event_time DESC."""
    with db_session(db_path) as conn:
        query = ["SELECT * FROM attendance_events WHERE 1=1"]
        params: List[Any] = []

        if date_str:
            query.append("AND substr(event_time, 1, 10) = ?")
            params.append(date_str)
        if person_name:
            query.append("AND person_name LIKE ?")
            params.append(f"%{person_name.strip()}%")
        if event_type and event_type != "ALL":
            query.append("AND event_type = ?")
            params.append(event_type.strip().upper())
        if status and status != "ALL":
            query.append("AND status = ?")
            params.append(status.strip().upper())

        query.append("ORDER BY event_time DESC, id DESC LIMIT ? OFFSET ?")
        params.extend([max(1, min(limit, 500)), max(0, offset)])

        cursor = conn.cursor()
        cursor.execute(" ".join(query), params)
        return [dict(row) for row in cursor.fetchall()]


def get_daily_summary(
    date_str: Optional[str] = None,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Get aggregate statistics for a given date (YYYY-MM-DD), default today."""
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_events,
                COUNT(DISTINCT CASE WHEN status = 'RECORDED' THEN person_name END) as unique_people,
                SUM(CASE WHEN event_type = 'CHECK_IN' AND status = 'RECORDED' THEN 1 ELSE 0 END) as total_check_ins,
                SUM(CASE WHEN event_type = 'CHECK_OUT' AND status = 'RECORDED' THEN 1 ELSE 0 END) as total_check_outs,
                SUM(CASE WHEN status = 'VOID' THEN 1 ELSE 0 END) as total_voided
            FROM attendance_events
            WHERE substr(event_time, 1, 10) = ?
        """, (date_str,))
        row = cursor.fetchone()
        return {
            "date": date_str,
            "total_events": row["total_events"] or 0,
            "unique_people": row["unique_people"] or 0,
            "total_check_ins": row["total_check_ins"] or 0,
            "total_check_outs": row["total_check_outs"] or 0,
            "total_voided": row["total_voided"] or 0,
        }


def get_daily_reconciliation(
    date_str: Optional[str] = None,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Reconcile daily attendance by individual:
    Identifies first check-in, last check-out, and flags missing check-outs.
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                person_name,
                MIN(CASE WHEN event_type = 'CHECK_IN' THEN event_time END) as first_in,
                MAX(CASE WHEN event_type = 'CHECK_OUT' THEN event_time END) as last_out,
                SUM(CASE WHEN event_type = 'CHECK_IN' THEN 1 ELSE 0 END) as in_count,
                SUM(CASE WHEN event_type = 'CHECK_OUT' THEN 1 ELSE 0 END) as out_count,
                COUNT(*) as total_actions
            FROM attendance_events
            WHERE substr(event_time, 1, 10) = ? AND status = 'RECORDED'
            GROUP BY person_name
            ORDER BY person_name ASC
        """, (date_str,))

        results = []
        for row in cursor.fetchall():
            first_in = row["first_in"]
            last_out = row["last_out"]
            in_count = row["in_count"]
            out_count = row["out_count"]

            if in_count > 0 and out_count > 0:
                reconciliation_status = "COMPLETED"
            elif in_count > 0 and out_count == 0:
                reconciliation_status = "MISSING_CHECK_OUT"
            elif in_count == 0 and out_count > 0:
                reconciliation_status = "CHECK_OUT_ONLY"
            else:
                reconciliation_status = "UNKNOWN"

            results.append({
                "person_name": row["person_name"],
                "date": date_str,
                "first_in": first_in,
                "last_out": last_out,
                "in_count": in_count,
                "out_count": out_count,
                "total_actions": row["total_actions"],
                "reconciliation_status": reconciliation_status
            })
        return results


def correct_event(
    event_id: int,
    new_event_type: Optional[str] = None,
    new_status: Optional[str] = None,
    actor: str = "Admin",
    reason: str = "Administrative correction",
    db_path: Optional[Path] = None
) -> Tuple[bool, str]:
    """
    Audited correction or status change for an existing attendance event.
    Preserves original event and appends to audit_log.
    """
    if not reason.strip():
        return False, "Audit reason is required for correction"

    with db_session(db_path) as conn:
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM attendance_events WHERE id = ?", (event_id,))
            event = cursor.fetchone()
            if not event:
                return False, f"Event #{event_id} not found"

            updates = []
            params = []
            action = "CORRECTED"

            if new_event_type and new_event_type != event["event_type"]:
                if new_event_type not in ("CHECK_IN", "CHECK_OUT"):
                    return False, f"Invalid event type {new_event_type}"
                updates.append("event_type = ?")
                params.append(new_event_type)

            if new_status and new_status != event["status"]:
                if new_status not in ("RECORDED", "CORRECTED", "VOID"):
                    return False, f"Invalid status {new_status}"
                updates.append("status = ?")
                params.append(new_status)
                if new_status == "VOID":
                    action = "VOIDED"

            if not updates:
                return False, "No changes specified"

            now_iso = datetime.now(timezone.utc).astimezone().isoformat()
            params.append(event_id)
            cursor.execute(f"UPDATE attendance_events SET {', '.join(updates)} WHERE id = ?", params)

            cursor.execute("""
                INSERT INTO audit_log (event_id, action, actor, reason, performed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (event_id, action, actor, reason, now_iso))

            return True, f"Event #{event_id} updated ({action}) by {actor}"


def get_audit_trail(event_id: Optional[int] = None, limit: int = 100, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve audit history records."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        if event_id:
            cursor.execute("""
                SELECT a.*, e.person_name, e.event_type, e.event_time
                FROM audit_log a
                JOIN attendance_events e ON a.event_id = e.id
                WHERE a.event_id = ?
                ORDER BY a.performed_at DESC LIMIT ?
            """, (event_id, limit))
        else:
            cursor.execute("""
                SELECT a.*, e.person_name, e.event_type, e.event_time
                FROM audit_log a
                JOIN attendance_events e ON a.event_id = e.id
                ORDER BY a.performed_at DESC LIMIT ?
            """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


def export_csv(
    date_str: Optional[str] = None,
    person_name: Optional[str] = None,
    db_path: Optional[Path] = None
) -> str:
    """Generate RFC 4180 compliant CSV export for attendance records."""
    events = get_events(date_str=date_str, person_name=person_name, limit=5000, db_path=db_path)
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([
        "Record ID", "Event ID", "Person Name", "Event Type",
        "Event Time", "Device ID", "Similarity Score", "Status",
        "Source", "Notes", "Created At"
    ])
    for event in events:
        writer.writerow([
            event["id"],
            event["event_id"],
            event["person_name"],
            event["event_type"],
            event["event_time"],
            event["device_id"],
            f"{event['similarity_score']:.4f}" if event["similarity_score"] is not None else "",
            event["status"],
            event["source"],
            event["notes"] or "",
            event["created_at"],
        ])
    return output.getvalue()


def backup_db(backup_dir: Optional[Path] = None, db_path: Optional[Path] = None) -> Path:
    """Create a live, transactionally consistent snapshot of the SQLite database."""
    target_path = db_path if db_path is not None else DEFAULT_DB_PATH
    target_dir = backup_dir if backup_dir is not None else target_path.parent / "backups"
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = target_dir / f"attendance_backup_{timestamp}.db"

    with db_session(target_path) as source_conn:
        dest_conn = sqlite3.connect(str(backup_file))
        try:
            source_conn.backup(dest_conn)
        finally:
            dest_conn.close()

    return backup_file
