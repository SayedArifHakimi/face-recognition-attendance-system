"""Unit tests for attendance database transactions, cooldowns, idempotency, audit logs and backups."""
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from attendance_db import (
    init_db,
    record_event,
    get_events,
    get_daily_summary,
    get_daily_reconciliation,
    correct_event,
    get_audit_trail,
    export_csv,
    backup_db,
    list_backups,
    restore_db,
)


class TestAttendanceDB(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_attendance.db"
        init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_record_event_basic(self):
        ok, msg, evt = record_event("Alice", "CHECK_IN", similarity_score=0.85, db_path=self.db_path)
        self.assertTrue(ok)
        self.assertEqual(evt["person_name"], "Alice")
        self.assertEqual(evt["event_type"], "CHECK_IN")
        self.assertEqual(evt["status"], "RECORDED")
        self.assertAlmostEqual(evt["similarity_score"], 0.85)

        # Check audit trail automatically created
        trails = get_audit_trail(db_path=self.db_path)
        self.assertEqual(len(trails), 1)
        self.assertEqual(trails[0]["action"], "CREATED")
        self.assertEqual(trails[0]["actor"], "SYSTEM")

    def test_cooldown_duplicate_prevention(self):
        # First check-in
        ok, _, _ = record_event("Bob", "CHECK_IN", cooldown_seconds=60, db_path=self.db_path)
        self.assertTrue(ok)

        # Immediate second check-in of same type -> must be rejected by cooldown
        ok2, msg2, evt2 = record_event("Bob", "CHECK_IN", cooldown_seconds=60, db_path=self.db_path)
        self.assertFalse(ok2)
        self.assertIn("Cooldown active", msg2)
        self.assertIsNone(evt2)

        # Different event type (CHECK_OUT) -> allowed
        ok3, _, evt3 = record_event("Bob", "CHECK_OUT", cooldown_seconds=60, db_path=self.db_path)
        self.assertTrue(ok3)
        self.assertEqual(evt3["event_type"], "CHECK_OUT")

    def test_idempotency_via_event_id(self):
        eid = "uuid-12345"
        ok1, _, evt1 = record_event("Charlie", "CHECK_IN", event_id=eid, db_path=self.db_path)
        self.assertTrue(ok1)

        # Re-sending same event_id
        ok2, msg2, evt2 = record_event("Charlie", "CHECK_IN", event_id=eid, db_path=self.db_path)
        self.assertTrue(ok2)
        self.assertIn("already recorded", msg2)
        self.assertEqual(evt1["id"], evt2["id"])

    def test_reject_unknown_and_invalid(self):
        ok, msg, _ = record_event("Unknown", "CHECK_IN", db_path=self.db_path)
        self.assertFalse(ok)
        self.assertIn("unknown", msg.lower())

        ok, msg, _ = record_event("", "CHECK_IN", db_path=self.db_path)
        self.assertFalse(ok)

        ok, msg, _ = record_event("David", "INVALID_ACTION", db_path=self.db_path)
        self.assertFalse(ok)

    def test_correction_and_audit(self):
        ok, _, evt = record_event("Eve", "CHECK_IN", db_path=self.db_path)
        self.assertTrue(ok)
        event_id = evt["id"]

        # Void the record with audit reason
        ok, msg = correct_event(event_id, new_status="VOID", actor="AdminUser", reason="Accidental badge", db_path=self.db_path)
        self.assertTrue(ok)

        events = get_events(status="VOID", db_path=self.db_path)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "VOID")

        trails = get_audit_trail(event_id=event_id, db_path=self.db_path)
        self.assertEqual(len(trails), 2)
        self.assertEqual(trails[0]["action"], "VOIDED")
        self.assertEqual(trails[0]["actor"], "AdminUser")
        self.assertEqual(trails[0]["reason"], "Accidental badge")

    def test_daily_summary_and_csv(self):
        record_event("UserA", "CHECK_IN", db_path=self.db_path)
        record_event("UserB", "CHECK_IN", db_path=self.db_path)
        record_event("UserA", "CHECK_OUT", db_path=self.db_path)

        summary = get_daily_summary(db_path=self.db_path)
        self.assertEqual(summary["total_events"], 3)
        self.assertEqual(summary["unique_people"], 2)
        self.assertEqual(summary["total_check_ins"], 2)
        self.assertEqual(summary["total_check_outs"], 1)

        csv_text = export_csv(db_path=self.db_path)
        lines = csv_text.strip().split("\n")
        self.assertEqual(len(lines), 4)  # Header + 3 rows
        self.assertTrue("UserA" in csv_text)
        self.assertTrue("UserB" in csv_text)

    def test_reconciliation_and_backup(self):
        today = datetime.now().strftime("%Y-%m-%d")
        record_event("Alice", "CHECK_IN", db_path=self.db_path)
        record_event("Alice", "CHECK_OUT", db_path=self.db_path)
        record_event("Bob", "CHECK_IN", db_path=self.db_path)

        reconcile = get_daily_reconciliation(today, db_path=self.db_path)
        self.assertEqual(len(reconcile), 2)
        alice_rec = next(r for r in reconcile if r["person_name"] == "Alice")
        bob_rec = next(r for r in reconcile if r["person_name"] == "Bob")

        self.assertEqual(alice_rec["reconciliation_status"], "COMPLETED")
        self.assertEqual(bob_rec["reconciliation_status"], "MISSING_CHECK_OUT")

        # Test backup
        backup_folder = Path(self.temp_dir.name) / "backups"
        backup_file = backup_db(backup_dir=backup_folder, db_path=self.db_path)
        self.assertTrue(backup_file.exists())
        self.assertGreater(backup_file.stat().st_size, 0)

    def test_restore_drill(self):
        backup_folder = Path(self.temp_dir.name) / "backups"

        # Populate initial events
        record_event("Alice", "CHECK_IN", db_path=self.db_path)
        ok, msg, evt = record_event("Bob", "CHECK_IN", db_path=self.db_path)
        correct_event(evt["id"], new_event_type="CHECK_OUT", actor="Manager", reason="User tapped wrong button", db_path=self.db_path)

        # Snapshot before catastrophic event
        backup_file = backup_db(backup_dir=backup_folder, db_path=self.db_path)
        self.assertTrue(backup_file.exists())
        backups = list_backups(backup_dir=backup_folder)
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0]["filename"], backup_file.name)

        # Simulate unwanted drift or corruption
        record_event("Intruder", "CHECK_IN", db_path=self.db_path)
        events_before_restore = get_events(db_path=self.db_path)
        self.assertEqual(len(events_before_restore), 3)

        # Restore from snapshot
        ok, msg = restore_db(backup_file, db_path=self.db_path)
        self.assertTrue(ok)
        self.assertIn("Successfully restored", msg)

        # Confirm state reverted exactly to snapshot (Alice and Bob present, Intruder absent)
        restored_events = get_events(db_path=self.db_path)
        self.assertEqual(len(restored_events), 2)
        names = [e["person_name"] for e in restored_events]
        self.assertIn("Alice", names)
        self.assertIn("Bob", names)
        self.assertNotIn("Intruder", names)

        # Confirm audit log integrity is preserved
        trails = get_audit_trail(event_id=evt["id"], db_path=self.db_path)
        self.assertEqual(len(trails), 2)
        self.assertEqual(trails[0]["actor"], "Manager")

    def test_restore_validation_failures(self):
        # 1. Non-existent file
        bogus_file = Path(self.temp_dir.name) / "non_existent.db"
        ok, msg = restore_db(bogus_file, db_path=self.db_path)
        self.assertFalse(ok)
        self.assertIn("not found", msg.lower())

        # 2. Corrupt / non-sqlite file
        corrupt_file = Path(self.temp_dir.name) / "corrupt.db"
        corrupt_file.write_bytes(b"This is not a sqlite database file at all!")
        ok, msg = restore_db(corrupt_file, db_path=self.db_path)
        self.assertFalse(ok)
        self.assertTrue("failed" in msg.lower() or "error" in msg.lower())

        # 3. Valid SQLite file but missing attendance schema
        empty_sqlite = Path(self.temp_dir.name) / "empty.db"
        import sqlite3
        conn = sqlite3.connect(str(empty_sqlite))
        try:
            conn.execute("CREATE TABLE dummy (id INTEGER PRIMARY KEY);")
        finally:
            conn.close()
        ok, msg = restore_db(empty_sqlite, db_path=self.db_path)
        self.assertFalse(ok)
        self.assertIn("missing required attendance tables", msg.lower())


if __name__ == "__main__":
    unittest.main()
