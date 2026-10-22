# Presentation attack detection (PAD) & liveness analysis

Prepared: 2026-10-22, Asia/Singapore  
Scope: Edge webcam kiosk evaluation for face-recognition attendance prototype.

---

## 1. Context & Motivation

In the original degree Final Year Project (UiTM Shah Alam, February 2023), face recognition focused on VGG16 classification over static crops without liveness verification. In an unattended kiosk or door-access deployment, 2D face recognition is vulnerable to presentation attacks (spoofing):
1. **Printed photo attack**: Holding a color photograph or printed card.
2. **Digital replay attack**: Playing back recorded video of an authorized user on a smartphone or tablet screen.
3. **3D mask / sculpture attack**: Physical mask approximating human facial contours.

---

## 2. Feasibility Analysis with OpenCV YuNet & 2D Webcam

OpenCV's YuNet detector provides bounding boxes and 5 facial landmarks:
- Right eye `(x_re, y_re)`
- Left eye `(x_le, y_le)`
- Nose tip `(x_nt, y_nt)`
- Right mouth corner `(x_rc, y_rc)`
- Left mouth corner `(x_lc, y_lc)`

### 2.1 Landmark-Based Heuristics (Provisional & Supervised Only)

| Heuristic | Mechanism | Strengths | Critical Failure Modes |
| :--- | :--- | :--- | :--- |
| **Blink Detection** | Eye aspect ratio / vertical landmark distance changes. | Low compute; flags static printed photos. | Easily defeated by digital video replay of a blinking user. |
| **Pose / Turn Challenge** | Requesting random head turn (Left, Right, Tilt). | Enforces cooperative interaction; stops static photos. | Friction for regular employees; replay video can be curated. |
| **Micro-Motion & Texture** | Laplacian variance / frequency analysis of screen moiré. | Detects screen refresh artifacts under controlled lighting. | Unreliable under low lighting, screen glare, or distance. |

### 2.2 Why 2D Passive Software Liveness Alone is Insufficient

- **Vulnerability to Video Replay**: Anyone with access to a smartphone recording or social media story of an enrolled person can replay it in front of the lens.
- **Lighting Sensitivity**: Edge artifact detection fails under varying ambient fluorescent or daylight conditions.
- **Academic vs Production Boundary**: While academic papers often report >95% accuracy on benchmark spoof datasets (e.g., CASIA-SURF, SiW), real-world cross-dataset generalization degrades significantly.

---

## 3. Recommended Production Hardware Architecture (TimeTec Reference)

Commercial attendance terminals (such as TimeTec TA / Face ID kiosks) achieve high-assurance anti-spoofing through **hardware-assisted multi-sensor fusion**:

1. **Dual-Lens Optical System**:
   - **RGB Visible Sensor**: Used for color image capture, alignment, and SFace feature extraction.
   - **Near-Infrared (NIR) Sensor + IR Illuminator**: Skin reflects 850nm/940nm infrared light differently than paper or electronic display glass. Phone screens appear black or produce intense specular flash, instantly exposing replays.
2. **Structured Light / Time-of-Flight (ToF) Depth**:
   - 3D point cloud generation verifies that the face is a 3D curved surface with depth relief (nose protruding ~2–3 cm from cheeks), rejecting flat 2D printouts and screens.
3. **Supervised Mode Policy for Phase 5 Baseline**:
   - The current local prototype remains explicitly designated as a **supervised kiosk**: operated in a manned reception or controlled entryway where human presence deters blatant presentation attacks.

---

## 4. Current Prototype Implementation Status

- [x] Phase 2: Open-set YuNet detection + SFace ONNX cosine embeddings.
- [x] Phase 3: Calibration parameters (sharpness threshold $\ge 50$, face size $\ge 80$ px, cosine threshold $0.50$, ambiguity margin $0.08$).
- [x] Phase 5: Transactional SQLite database, idempotency, 60s duplicate cooldown, audit logging, and Admin Web Portal.
- [ ] Phase 6: Active presentation attack dataset benchmark (requires consenting volunteer printouts & replay tests under controlled protocol).
- [ ] Phase 7: Unattended hardware evaluation (NIR / Dual-camera integration).
