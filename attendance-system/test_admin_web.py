"""Unit tests for the admin web portal HTTP endpoints."""
import json
import tempfile
import threading
import time
import urllib.request
import unittest
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch

from attendance_db import init_db, record_event
from admin_web import AttendanceHTTPHandler


class TestAdminWeb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.temp_dir.name) / "test_admin.db"
        init_db(cls.db_path)

        # Prepopulate sample records
        record_event("Sayed", "CHECK_IN", similarity_score=0.92, db_path=cls.db_path)
        record_event("Alice", "CHECK_IN", similarity_score=0.88, db_path=cls.db_path)

        # Patch DEFAULT_DB_PATH in admin_web and attendance_db
        cls.patcher = patch("admin_web.DEFAULT_DB_PATH", cls.db_path)
        cls.patcher_db = patch("attendance_db.DEFAULT_DB_PATH", cls.db_path)
        cls.gallery_path = Path(cls.temp_dir.name) / "test_gallery.json"
        with open(cls.gallery_path, "w", encoding="utf-8") as f:
            json.dump({"Alice": [[1.0] * 128], "Bob": [[0.5] * 128]}, f)
        cls.patcher_gal = patch("admin_web.GALLERY_FILE", cls.gallery_path)
        cls.patcher.start()
        cls.patcher_db.start()
        cls.patcher_gal.start()

        # Start background HTTP server on port 0 (OS picks free port)
        cls.server = HTTPServer(("127.0.0.1", 0), AttendanceHTTPHandler)
        cls.port = cls.server.server_port
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.patcher.stop()
        cls.patcher_db.stop()
        cls.patcher_gal.stop()
        cls.temp_dir.cleanup()

    def test_get_user_kiosk_page(self):
        for route in ["/", "/kiosk", "/user"]:
            req = urllib.request.urlopen(f"{self.base_url}{route}")
            self.assertEqual(req.status, 200)
            content = req.read().decode("utf-8")
            self.assertIn("Face Attendance Kiosk", content)
            self.assertIn("Start Camera Stream", content)
            self.assertIn("Admin Portal", content)
            # Verify admin actions are NOT exposed on user kiosk
            self.assertNotIn("Purge Retention", content)
            self.assertNotIn("Database Restore", content)

    def test_get_admin_portal_page(self):
        req = urllib.request.urlopen(f"{self.base_url}/admin")
        self.assertEqual(req.status, 200)
        content = req.read().decode("utf-8")
        self.assertIn("Attendance Admin Portal", content)
        self.assertIn("Daily Reconciliation", content)
        self.assertIn("Enrolled Persons in Gallery", content)
        self.assertIn("Data Retention Policy &amp; Event Purge", content)
        self.assertIn("Kiosk View", content)

    def test_get_events_and_summary(self):
        req = urllib.request.urlopen(f"{self.base_url}/api/events")
        self.assertEqual(req.status, 200)
        events = json.loads(req.read().decode("utf-8"))
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["person_name"], "Alice")

        req_sum = urllib.request.urlopen(f"{self.base_url}/api/summary")
        summary = json.loads(req_sum.read().decode("utf-8"))
        self.assertEqual(summary["total_check_ins"], 2)
        self.assertEqual(summary["unique_people"], 2)

    def test_manual_entry_and_audit(self):
        payload = json.dumps({
            "person_name": "Bob",
            "event_type": "CHECK_IN",
            "actor": "LeadAdmin",
            "reason": "Forgot camera check"
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/manual",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            event_id = data["event"]["id"]

        # Now test correction endpoint
        correct_payload = json.dumps({
            "event_id": event_id,
            "new_status": "VOID",
            "actor": "LeadAdmin",
            "reason": "Duplicate entry"
        }).encode("utf-8")
        correct_req = urllib.request.Request(
            f"{self.base_url}/api/correct",
            data=correct_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(correct_req) as resp:
            self.assertEqual(resp.status, 200)
            cdata = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(cdata["success"])

        # Check audit trail endpoint
        audit_req = urllib.request.urlopen(f"{self.base_url}/api/audit")
        audit_logs = json.loads(audit_req.read().decode("utf-8"))
        self.assertTrue(any(a["action"] == "VOIDED" and a["actor"] == "LeadAdmin" for a in audit_logs))

    def test_csv_export(self):
        req = urllib.request.urlopen(f"{self.base_url}/export/csv")
        self.assertEqual(req.status, 200)
        self.assertEqual(req.headers.get("Content-Type"), "text/csv; charset=utf-8")
        content = req.read().decode("utf-8")
        self.assertTrue(content.startswith("Record ID,Event ID"))
        self.assertIn("Sayed", content)

    def test_reconciliation_and_backup(self):
        req = urllib.request.urlopen(f"{self.base_url}/api/reconciliation")
        self.assertEqual(req.status, 200)
        reconcile = json.loads(req.read().decode("utf-8"))
        self.assertGreaterEqual(len(reconcile), 2)

        backup_req = urllib.request.Request(
            f"{self.base_url}/api/backup",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(backup_req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertTrue(data["backup_path"].startswith("attendance_backup_"))

    def test_backup_and_restore_endpoints(self):
        # 1. Trigger backup
        backup_req = urllib.request.Request(
            f"{self.base_url}/api/backup",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(backup_req) as resp:
            self.assertEqual(resp.status, 200)
            bdata = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(bdata["success"])
            filename = bdata["backup_path"]

        # 2. Query /api/backups
        req_list = urllib.request.urlopen(f"{self.base_url}/api/backups")
        self.assertEqual(req_list.status, 200)
        backups = json.loads(req_list.read().decode("utf-8"))
        self.assertGreaterEqual(len(backups), 1)
        self.assertTrue(any(b["filename"] == filename for b in backups))

        # 3. Post /api/restore
        restore_payload = json.dumps({"filename": filename}).encode("utf-8")
        restore_req = urllib.request.Request(
            f"{self.base_url}/api/restore",
            data=restore_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(restore_req) as resp:
            self.assertEqual(resp.status, 200)
            rdata = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(rdata["success"])
            self.assertIn("Successfully restored", rdata["message"])

    def test_favicon_endpoints(self):
        req_svg = urllib.request.urlopen(f"{self.base_url}/favicon.svg")
        self.assertEqual(req_svg.status, 200)
        self.assertEqual(req_svg.headers.get("Content-Type"), "image/svg+xml")
        svg_content = req_svg.read().decode("utf-8")
        self.assertIn("<svg", svg_content)

        req_ico = urllib.request.urlopen(f"{self.base_url}/favicon.ico")
        self.assertEqual(req_ico.status, 200)
        self.assertEqual(req_ico.headers.get("Content-Type"), "image/x-icon")
        self.assertGreater(len(req_ico.read()), 0)

    def test_purge_endpoint(self):
        purge_payload = json.dumps({
            "older_than": "2026-06-01",
            "actor": "SecurityLead",
            "reason": "Purge test events"
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/purge",
            data=purge_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertIn("purged_count", data)

    def test_gallery_delete_endpoint(self):
        del_payload = json.dumps({
            "person_name": "Alice",
            "actor": "Admin",
            "reason": "Test departure"
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/gallery/delete",
            data=del_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertEqual(data["count"], 1)

    def test_maintenance_endpoint(self):
        # Trigger an action first to ensure at least one maintenance record exists
        purge_payload = json.dumps({
            "older_than": "2026-01-01",
            "actor": "AuditBot",
            "reason": "Initial test purge"
        }).encode("utf-8")
        req_p = urllib.request.Request(
            f"{self.base_url}/api/purge",
            data=purge_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        urllib.request.urlopen(req_p).read()

        req = urllib.request.urlopen(f"{self.base_url}/api/maintenance")
        self.assertEqual(req.status, 200)
        logs = json.loads(req.read().decode("utf-8"))
        self.assertIsInstance(logs, list)
        self.assertGreaterEqual(len(logs), 1)


if __name__ == "__main__":
    unittest.main()
