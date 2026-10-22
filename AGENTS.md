# Face Recognition & Attendance System Governance

**Governed by:** HCE (Health Check Engineer)  
**Standard:** Ponytail Lazy Senior Developer Framework

---

## Directives for Working on Face Recognition

1. **Ponytail Discipline:**
   - Keep the kiosk and recognition engine lean. Rely on OpenCV DNN and NumPy.
   - Do not introduce heavy external ML frameworks, unneeded abstractions, or complex UI layers without explicit approval.
   - Verify every change using `/ponytail-review`.

2. **Biometric Security & Privacy (Strict):**
   - **NEVER** commit, share, or publish `data/gallery.json` or any raw facial embeddings.
   - All enrollment data belongs strictly in the local `data/` directory, which is excluded from version control.

3. **Mandatory Test Verification:**
   - All automated unit, evaluation, and kiosk tests must pass with 0 failures before any task is complete:
     ```powershell
     cd attendance-system
     & .\.venv\Scripts\python.exe -m unittest discover -v -p "test_*.py"
     & .\.venv\Scripts\python.exe test_face_id.py
     ```

4. **Model Assets Integrity:**
   - Keep ONNX model assets (`sface.onnx` and `yunet.onnx`) verified against `manifest.json`. Do not modify or replace model binaries without updating checksums and licenses.

---

## 5. GitHub Cloud Storage & Remote Policy

- **Official Remote:** `https://github.com/SayedArifHakimi/face-recognition-attendance-system.git`
- **Visibility:** Public
- **Strict Sync Exclusions:**
  - `data/gallery.json` (biometric vectors)
  - `data/*.db` (local SQLite transaction store)
  - `.venv/` (Python virtual environment)
  - `tmp/` and `__pycache__/`
- **Push Policy:** Only clean, tested kiosk and engine code may be pushed to `origin/main`.
