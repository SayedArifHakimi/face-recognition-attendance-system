# Face Recognition Attendance System

An edge computer vision and transactional attendance kiosk, reminiscing and re-engineering my degree Final Year Project (FYP) in Intelligent Systems Engineering at Universiti Teknologi MARA (UiTM) Shah Alam (thesis completed February 2023).

---

## 1. Project Background & Evolution

| Milestone | Original Academic Baseline (Feb 2023) | Rebuilt Edge System (Oct 2026) |
| :--- | :--- | :--- |
| **Model** | VGG16 CNN classification over 4 people | OpenCV YuNet (detection) + SFace ONNX (128-d embeddings) |
| **Recognition** | Closed-set softmax (forced nearest match) | Open-set cosine verification with unknown rejection & ambiguity margin |
| **Interface** | Python Tkinter GUI | Desktop Kiosk (`face_id.py`) & Zero-Dependency Web Portal (`admin_web.py`) |
| **Storage** | MySQL / phpMyAdmin | Transactional SQLite with ACID guarantees & duplicate cooldown |
| **Auditability** | Static Excel export, no audit trail | Tamper-evident `audit_log`, manual entries, and audited corrections |
| **Liveness / PAD** | None | 5-point landmark head pose estimation & micro-motion variance heuristics |

---

## 2. Architecture Overview

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

## 3. Quick Start

### Prerequisites
- Python 3.10+ on Windows / Linux / macOS
- Webcam (Camera 0)

### Setup & Run
```powershell
# Navigate to attendance-system
cd attendance-system

# Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Download model weights (verified via manifest SHA-256)
.\.venv\Scripts\python.exe setup_models.py

# Run full HCE (Health Check Engineer) verification suite (25 tests + diagnostics + calibration)
.\.venv\Scripts\python.exe hce_check.py

# Run empirical biometric gallery calibration & LOOCV evaluation
.\.venv\Scripts\python.exe evaluate_gallery.py --sweep

# Launch interactive Kiosk & Face ID monitor
.\.venv\Scripts\python.exe face_id.py

# Launch Admin & Audit Web Portal (http://127.0.0.1:8080)
.\.venv\Scripts\python.exe admin_web.py --port 8080
```

---

## 4. Empirical Hardware Latency (Windows PC)

Measured via `benchmark.py`:
- **Face Detection (YuNet 640x480)**: Mean = 9.00 ms (111.1 FPS)
- **Feature Extraction (SFace 112x112)**: Mean = 6.91 ms
- **End-to-End Recognition Pipeline**: Mean = 15.92 ms (**62.8 FPS** real-time processing)
- **Gallery Search**: ~6.6 ms for 2 users (200 references); ~32 ms for 10 users (1,000 references).

---

## 5. Security & Privacy Boundaries

- **Local-First Storage**: Biometric templates and attendance records remain strictly local on the kiosk and are excluded from Git version control.
- **Supervised Operation**: Designed for manned reception/kiosks. Presentation attack analysis and hardware requirements (NIR/3D ToF) are documented in `attendance-system/docs/presentation-attack-analysis.md`.
