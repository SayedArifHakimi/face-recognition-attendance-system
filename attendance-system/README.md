# Local Face Identity & Attendance System

Supervised face recognition and attendance kiosk prototype, modernizing the February 2023 UiTM Shah Alam degree FYP. Built on lightweight open-source edge models (OpenCV YuNet + SFace ONNX) with transactional SQLite attendance logging, duplicate prevention, and a zero-dependency admin web portal.

---

## 1. Quick Start

Ensure commands are executed in PowerShell from this folder (`attendance-system`):

```powershell
# 1. Run mandatory HCE verification suite (25 tests + diagnostics + calibration)
.\.venv\Scripts\python.exe hce_check.py

# 2. Run empirical gallery calibration & LOOCV
.\.venv\Scripts\python.exe evaluate_gallery.py --sweep

# 3. Start the interactive Kiosk & Face ID monitor
.\.venv\Scripts\python.exe face_id.py

# 4. Start the Web Server (serves both User Kiosk and Admin Portal)
.\.venv\Scripts\python.exe admin_web.py --port 8080
```

- **User / Kiosk Interface**: **http://127.0.0.1:8080/** (also `/kiosk`, `/user`)
- **Admin Management Portal**: **http://127.0.0.1:8080/admin**

---

## 2. Kiosk Attendance Operations (`face_id.py`)

- **Live Face ID**: Checks detected faces in real-time. Authorized individuals show green bounding boxes with their name; unrecognized visitors show orange boxes with **Unknown**.
- **Explicit Clocking**:
  - When exactly one recognized face is visible, the **Attendance Action** panel unlocks.
  - Click **Clock In: [Name]** or press `I`.
  - Click **Clock Out: [Name]** or press `O`.
- **Anti-Duplicate Cooldown**:
  - Enforces a 60-second cooldown per person for identical actions to prevent multiple accidental records from standing in front of the lens.
- **Guided Multi-Phase Enrollment**:
  - Click **Capture Samples**, enter an ASCII name, center face, and press `SPACE` to start.
  - Guided 35-second capture across straight, left, right, up, down, natural blink, and varying expression/lighting (up to 350 samples).
  - Embeddings are stored locally in `data/gallery.json`. No raw webcam video or photos are saved to disk.
- **Admin / Settings Page**:
  - Real-time adjustment of match threshold (default `0.50`), ambiguity margin (default `0.08`), minimum face pixels (default `80`), and sharpness (default `50`).
  - View enrolled gallery and securely remove references with `DELETE` confirmation.

---

## 3. Web Architecture: User Kiosk & Admin Portal (`admin_web.py`)

A zero-dependency Python HTTP server strictly separating self-service employee attendance and administrative controls:

### A. User / Kiosk Web Interface (`/`, `/kiosk`, `/user`)
- **Self-Service Attendance**: Live camera stream with visual bounding boxes, real-time facial recognition status, and single-click **Clock In** (`I`) / **Clock Out** (`O`) controls.
- **Privacy & Simplicity**: Read-only display of today's verified check-in/out events. Zero administrative controls, database purge tools, or sensitive audit logs are exposed on the user page.

### B. Admin Management Portal (`/admin`)
- **Metrics & Reconciliation**: Live tally of attendance actions, daily check-in vs check-out reconciliations, and automatic flagging of missing check-outs.
- **Filterable Logs & Audit Trail**: Query attendance records by date, employee name, event type, and status (`RECORDED`, `CORRECTED`, `VOID`). Every correction requires an admin name and audited reason.
- **Manual Fallback & CSV Export**: Record manual entries when badge/camera scans are unavailable, and download RFC 4180 CSV logs with a single click.
- **Enrolled Identity Management**: View enrolled persons, prune centroid outlier embeddings, or delete enrolled identities with automatic pre-deletion backups (`data/gallery_backups/`).
- **Data Retention & Snapshots**: Automated pre-action database backups, point-in-time snapshot restores, and retention-based event purging.
- **Biometric Favicon**: Native SVG and standard 32×32 ICO icons served at `/favicon.svg` and `/favicon.ico` for clean browser bookmarking.

---

## 4. Database & Storage Architecture (`attendance_db.py`)

All attendance data is stored locally in `data/attendance.db` (SQLite) with foreign key constraints and ACID transaction guarantees:
- `attendance_events`: records event UUID, person name, event type, timestamp, similarity score, status, source, device ID, and creation date.
- `audit_log`: records event ID, action (`CREATED`, `CORRECTED`, `VOIDED`, `MANUAL_ENTRY`), actor, reason, and execution timestamp.
- **Idempotency**: Submitting an existing `event_id` returns the existing record without duplicate insertion.
- **Snapshot Backups**: Built using SQLite's native `conn.backup()` API.

---

## 5. Setup on a Fresh PC

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_models.py
.\.venv\Scripts\python.exe test_face_id.py
.\.venv\Scripts\python.exe test_attendance.py
.\.venv\Scripts\python.exe test_admin_web.py
.\.venv\Scripts\python.exe face_id.py --check-camera
.\.venv\Scripts\python.exe face_id.py
```

---

## 6. Architecture & Security Boundaries

- **Supervised Prototype**: This system is designed as a manned kiosk. It does not contain 3D depth or infrared anti-spoofing; presentation attacks with printed photos or digital replays require human supervision. See `docs/presentation-attack-analysis.md` for full PAD analysis.
- **Privacy & Biometric Data**: Biometric embeddings in `data/gallery.json` and attendance records in `data/attendance.db` are private. Both files are excluded from Git in `.gitignore`.
- **Model Hashes**: OpenCV Zoo pretrained YuNet (`yunet.onnx`) and SFace (`sface.onnx`) hashes are pinned in `models/manifest.json`.

See `docs/restart-plan-2026-10-08.md` and `docs/milestones-2026-10-22.md` for the technical roadmap and delivery records.
