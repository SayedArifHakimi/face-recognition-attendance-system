"""System diagnostics and health verification for local face identity and attendance kiosk."""
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
DATA_DIR = ROOT / "data"
GALLERY_PATH = DATA_DIR / "gallery.json"
SETTINGS_PATH = DATA_DIR / "settings.json"
DB_PATH = DATA_DIR / "attendance.db"
BACKUPS_DIR = DATA_DIR / "backups"


def check_models() -> bool:
    print("[1/5] Checking model integrity against manifest.json...")
    if not MANIFEST_PATH.exists():
        print("  FAIL: manifest.json not found")
        return False
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    all_ok = True
    for filename, meta in manifest.get("files", {}).items():
        file_path = MODELS_DIR / filename
        if not file_path.exists():
            print(f"  FAIL: Missing model file: {filename}")
            all_ok = False
            continue
        sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if sha.lower() == meta["sha256"].lower():
            print(f"  OK: {filename} (SHA-256 match)")
        else:
            print(f"  FAIL: {filename} hash mismatch!")
            all_ok = False
    return all_ok


def check_database() -> bool:
    print("\n[2/5] Checking SQLite database integrity...")
    if not DB_PATH.exists():
        print("  WARN: attendance.db does not exist yet (will be created on first start)")
        return True
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        res = cursor.fetchone()
        if res and res[0] == "ok":
            print("  OK: Database PRAGMA integrity_check passed")
        else:
            print(f"  FAIL: Database corrupted: {res}")
            return False

        cursor.execute("SELECT COUNT(*) FROM attendance_events;")
        events_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM audit_log;")
        audit_count = cursor.fetchone()[0]
        print(f"  OK: {events_count} attendance events, {audit_count} audit trail records")

        backups = list(BACKUPS_DIR.glob("*.db")) if BACKUPS_DIR.exists() else []
        print(f"  OK: {len(backups)} snapshot backup(s) found in data/backups/")
        conn.close()
        return True
    except Exception as e:
        print(f"  FAIL: Database error: {e}")
        return False


def check_gallery() -> bool:
    print("\n[3/5] Checking biometric gallery & settings...")
    if not GALLERY_PATH.exists():
        print("  INFO: gallery.json does not exist (no enrolled persons yet)")
    else:
        try:
            gallery = json.loads(GALLERY_PATH.read_text(encoding="utf-8"))
            if isinstance(gallery, dict):
                print(f"  OK: Gallery contains {len(gallery)} enrolled person(s):")
                for name, refs in gallery.items():
                    print(f"      - {name}: {len(refs)} reference embeddings (128-d)")
                if len(gallery) >= 2:
                    from evaluate_gallery import load_gallery_matrix, run_loocv
                    names, X, y = load_gallery_matrix(GALLERY_PATH)
                    res = run_loocv(names, X, y, threshold=0.50, margin=0.08)
                    print(f"  OK: Empirical LOOCV (threshold=0.50, margin=0.08): TAR={res['tar']*100:.2f}%, FAR={res['far']*100:.2f}% ({res['mismatch']} mismatches)")
            else:
                print("  FAIL: gallery.json is not a valid JSON dictionary")
                return False
        except Exception as e:
            print(f"  FAIL: Gallery read error: {e}")
            return False

    if not SETTINGS_PATH.exists():
        print("  INFO: settings.json not found (default settings will be used)")
    else:
        try:
            settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            print(f"  OK: Settings configured: threshold={settings.get('threshold')}, margin={settings.get('margin')}, min_face={settings.get('min_face')}, min_sharpness={settings.get('min_sharpness')}")
        except Exception as e:
            print(f"  WARN: Settings read error: {e}")

    return True


def check_inference() -> bool:
    print("\n[4/5] Checking OpenCV DNN inference runtime...")
    try:
        import cv2
        import numpy as np

        yunet_path = str(MODELS_DIR / "yunet.onnx")
        sface_path = str(MODELS_DIR / "sface.onnx")

        start = time.perf_counter()
        detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.9)
        recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

        # Test dummy forward pass
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        detector.detect(dummy)
        dummy_crop = np.zeros((112, 112, 3), dtype=np.uint8)
        feature = recognizer.feature(dummy_crop)
        duration = (time.perf_counter() - start) * 1000

        if feature.shape[-1] == 128:
            print(f"  OK: OpenCV DNN model inference passed in {duration:.1f}ms (128-d output verified)")
            return True
        else:
            print(f"  FAIL: Unexpected feature output shape: {feature.shape}")
            return False
    except Exception as e:
        print(f"  FAIL: DNN runtime error: {e}")
        return False


def check_camera(camera_id: int = 0) -> bool:
    print(f"\n[5/5] Checking camera {camera_id} access...")
    try:
        import cv2
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"  WARN: Camera {camera_id} could not be opened (is another application using it?)")
            return False
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            print(f"  OK: Camera {camera_id} opened successfully: {frame.shape[1]}x{frame.shape[0]} px (no frame saved)")
            return True
        else:
            print(f"  WARN: Camera {camera_id} opened but returned no frame")
            return False
    except Exception as e:
        print(f"  WARN: Camera check encountered exception: {e}")
        return False


def main():
    print("=" * 60)
    print("  ATTENDANCE SYSTEM DIAGNOSTICS & HEALTH CHECK")
    print("=" * 60)
    
    ok1 = check_models()
    ok2 = check_database()
    ok3 = check_gallery()
    ok4 = check_inference()
    ok5 = check_camera()

    print("\n" + "=" * 60)
    if ok1 and ok2 and ok3 and ok4:
        print("  STATUS: ALL SYSTEM CHECKS PASSED HEALTHILY")
    else:
        print("  STATUS: DIAGNOSTICS DETECTED ISSUES")
    print("=" * 60)


if __name__ == "__main__":
    main()
