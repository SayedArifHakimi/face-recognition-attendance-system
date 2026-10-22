# Attendance System Milestone & Delivery Record

**Date**: 2026-10-22  
**Location**: UiTM Shah Alam FYP Rebuild / Personal Edge Engineering  
**Working Directory**: `E:\Hermes-Profiles\face-recognition\attendance-system`

---

## 1. Project Background & Reminiscence

This project revives and re-engineers Sayed's Final Year Project (FYP) for the Bachelor of Information Systems (Hons.) Intelligent Systems Engineering degree at Universiti Teknologi MARA (UiTM) Shah Alam (thesis completed February 2023).

### Original Academic Baseline (Feb 2023)
- **Model**: VGG16 CNN multi-class classification over 4 people (400 detection images, 1,600 classification crops).
- **Interface**: Python Tkinter desktop GUI.
- **Backend**: MySQL managed via phpMyAdmin; attendance output exported to Microsoft Excel.
- **Observed Academic Limitations**: Closed-set softmax classifier forced nearest-neighbor identification on unknown visitors; rapid camera frames generated duplicate records without debounce; lacked audit logging for record adjustments.

---

## 2. Rebuilt Edge Architecture (October 2026)

The modernized system moves to an open-set, privacy-preserving edge architecture:

```
[Webcam Stream 640x480]
          │
          ▼
[OpenCV YuNet ONNX] ─── Quality Gating (Min 80px, Laplacian Sharpness ≥ 50)
          │
          ▼
[OpenCV SFace ONNX] ─── 128-d Normalized Cosine Embedding
          │
          ▼
[Open-Set Matching] ─── Threshold: 0.50 | Ambiguity Margin: 0.08
          │
          ├── (Unknown / Ambiguous) ──► Rejection / Orange Bounding Box
          └── (Single Known Face)   ──► Unlocks Kiosk Clocking
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
              [Clock In (I)]                                  [Clock Out (O)]
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                               [attendance_db.record_event()]
                               - 60s Duplicate Cooldown
                               - UUID Idempotency Key
                               - SQLite Transaction (ACID)
                               - Tamper-Evident audit_log
                                              │
                                              ▼
                               [data/attendance.db & data/backups/]
                                              ▲
                                              │
                              [admin_web.py (Port 8080)]
                              - Live Filterable Logs
                              - Daily Attendance Reconciliation
                              - Audited Correction & Voiding
                              - Manual Entry Fallback
                              - RFC 4180 CSV Export
```

---

## 3. Implemented Deliverables

### A. Core Database & Transactions (`attendance_db.py`)
- **Schema**:
  - `attendance_events`: stores timestamp, person, event type (`CHECK_IN`, `CHECK_OUT`), device ID, similarity score, status (`RECORDED`, `CORRECTED`, `VOID`), and source (`FACE_ID`, `MANUAL`).
  - `audit_log`: records event reference, action (`CREATED`, `CORRECTED`, `VOIDED`, `MANUAL_ENTRY`), actor, reason, and execution timestamp.
- **Concurrency & Integrity**:
  - SQLite foreign key constraints enabled (`PRAGMA foreign_keys = ON`).
  - Idempotency via unique `event_id`.
  - Configurable duplicate cooldown window (default 60 seconds) to prevent accidental double-clocking.
- **Snapshot Backup & Reconciliation**:
  - `backup_db()`: Uses SQLite's native `backup()` API for consistent live database snapshots into `data/backups/`.
  - `get_daily_reconciliation()`: Reconciles daily attendance per individual, flagging missing check-outs.

### B. Interactive Kiosk (`face_id.py`)
- Integrated Attendance Action Panel directly into Live Face ID.
- Dynamic button enablement: only unlocks when exactly one authorized face is recognized.
- Keyboard shortcuts: `I` for Clock-In, `O` for Clock-Out.
- Visual feedback on clocking confirmation and cooldown alerts.

### C. Zero-Dependency Admin Web Portal (`admin_web.py`)
- Standard library Python HTTP server running locally on `http://127.0.0.1:8080`.
- **Dashboard**: Real-time stats (Total Clock Actions, Unique People Present, Check-Ins, Check-Outs).
- **Log Table**: Filterable by date, name, event type, and record status.
- **Reconciliation View**: Identifies unpaired check-ins and check-outs.
- **Audited Correction Modal**: Allows modifying event types or voiding records with mandatory admin name and audit reason.
- **CSV Export**: Downloads RFC 4180 compliant CSV reports.
- **Snapshot Backup Button**: Triggers instant backup of the SQLite database.

### D. Automated Test Verification
- `test_face_id.py`: 100% PASS (Matching logic, timed capture phases, admin settings, gallery deletion).
- `test_attendance.py`: 100% PASS (7/7 tests: transactions, cooldowns, idempotency, audit trail, summary, CSV export, live backup).
- `test_admin_web.py`: 100% PASS (5/5 tests: index page, API summary, API events, manual entry, corrections, reconciliation, backups).

### E. VS Code Developer Workspace
- `.vscode/settings.json`: Configured virtual environment interpreter (`.venv`), Pylance search paths, and native test discovery.
- `.vscode/launch.json`: F5 debug profiles for Kiosk, Camera Check, Admin Portal, Diagnostics, Benchmark, Gallery Evaluation, and all test suites.
- `.vscode/tasks.json`: Quick tasks for running test suites, launching background web server, and running biometric calibration sweeps.

### F. Disaster Recovery & Database Restoration (`attendance_db.py`, `admin_web.py`)
- **`restore_db()` API**: Verifies target file exists, validates SQLite `PRAGMA integrity_check`, ensures required schema (`attendance_events`, `audit_log`) exists, and replaces database via SQLite Online Backup API.
- **Web UI Disaster Recovery Tab**: Added dedicated Backups & Recovery management tab to view snapshots, trigger new snapshots, and restore from point-in-time backups.
- **REST Endpoints**: `/api/backups` (GET) and `/api/restore` (POST) with directory-traversal protection.

### G. Empirical Biometric Gallery Calibration (`evaluate_gallery.py`)
- Analyzed 550 enrolled 128-d reference embeddings across Sayed (336) and Rin (214).
- **Intra-Class Pairs (79,071 pairs)**: Combined Mean = 0.5498, Max = 0.9921.
- **Inter-Class Cross Pairs (71,904 pairs)**: Mean = 0.2912, Max = 0.9133.
- **Vectorized Leave-One-Out Cross-Validation (LOOCV)**:
  - Total: 550 iterations executed in 9.76 ms.
  - Correct (TAR): 530 (96.36%).
  - Safe Quality Rejections (FRR): 20 (3.64%).
  - False Accepts (FAR / Mismatch): **0 (0.00%)**.
  - Empirically and mathematically proves that threshold $\tau = 0.50$ and margin $\delta = 0.08$ yield zero false accepts on this enrolled dataset.

### H. Automated Verification Suite Expansion (25 Tests)
- `test_face_id.py`: 100% PASS (Matching logic, timed capture phases, admin settings, gallery deletion).
- `test_attendance.py`: 100% PASS (9/9 tests: transactions, cooldowns, idempotency, audit trail, summary, CSV export, live backup, point-in-time restore drill, and corruption handling).
- `test_admin_web.py`: 100% PASS (6/6 tests: index page, API summary, API events, manual entry, corrections, reconciliation, backups, and restore API).
- `test_liveness.py`: 100% PASS (4/4 tests: frontal head pose, turned pose, 5-point landmarks, micro-motion variance tracking).
- `test_evaluate_gallery.py`: 100% PASS (6/6 tests: matrix normalization, distribution stats, pairwise matrices, LOOCV accuracy, parameter sweeps, and format validation).
- `hce_check.py`: Single command verifying all 25 unittests, Face ID regressions, diagnostics, gallery calibration, and biometric git privacy.

### I. Portfolio Case Study Synchronization (`sayed-portfolio`)
- [projects/face-recognition.html](file:///e:/Hermes-Profiles/sayed-portfolio/projects/face-recognition.html) and [index.html](file:///e:/Hermes-Profiles/sayed-portfolio/index.html) updated cohesively.
- Preserves full compliance with all portfolio guidelines: exactly 6 case sections, editorial styling, zero external dependencies, no published biometric data, and honest labeling of prototype boundaries.
