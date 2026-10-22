# Attendance system restart plan

Prepared: 2026-10-08; Updated: 2026-10-22, Asia/Singapore.  
Status: Recognition baseline and Phase 5 attendance implementation delivered. SQLite transactional storage, duplicate cooldowns, audit logging, kiosk explicit clocking, and zero-dependency admin web portal verified with automated test suites. Liveness verification and unattended pilot evaluation remain pending under Phase 6/7. See `docs/milestones-2026-10-22.md` and `docs/presentation-attack-analysis.md`.

## Purpose and assumptions

Restart the degree FYP as a usable face-recognition attendance system, taking TimeTec as a product reference. Preserve the original FYP archive. Build in E:\Hermes-Profiles\face-recognition\attendance-system.

Confirmed user scope: personal PC at home, a few consenting users for testing, a working face-to-name prototype first, attendance and expansion later. Current baseline: one Windows webcam window and local embedding gallery. Browser administration and attendance storage are later phases. No claim of production readiness.

## Existing evidence

The February 2023 thesis documents two VGG16 experiments, a Tkinter prototype, student details in a database managed with phpMyAdmin, and attendance output in Excel. It reports 400 detection images and 1,600 classification images covering four people. Live classification includes incorrect identities despite high validation accuracy.

The inspected CODE archive contains detection/training notebooks, a saved facetracker model, a webcam preview script, augmentation experiments, old dlib binaries and an empty VB.NET form. The documented final attendance application was not found in those files. Existing artifacts have not been executed or reproduced.

## First release

An operator registers a person and explicitly enrolls their face. The kiosk captures a face, checks image quality, aligns it, generates an embedding, and compares it with enrolled references. Uncertain or unknown people are rejected; no forced nearest-person assignment. Multiple faces require retry for the first release.

A successful authorized match records an explicit check-in or check-out with person, event time, received time, device and event ID. No automatic toggling on every detection. Database uniqueness and transaction handling prevent duplicate records. Corrections retain the original event and record who changed it and why.

Admin screens cover enrollment, active users, attendance filtering, audited correction and CSV export. Include manual attendance fallback. Local kiosk continues without internet; backups must be restorable.

No payroll, leave, GPS clocking, mobile app, multi-tenant cloud platform or door access in the first release.

## Technology baseline

- Python with OpenCV YuNet detection and SFace embeddings as the first benchmark candidate. Use pretrained inference; enrollment stores references without retraining for each new user.
- Python standard-library SQLite for a single local deployment. Use transactions and constraints for event integrity. Revisit a server database only for shared deployment needs.
- A single Python web application with simple HTML forms for administration; choose its framework after confirming deployment and existing runtime. No separate frontend build required initially.
- Pin dependency versions and model artifact hashes after compatibility testing. Review the exact model licenses before commercial selection. No model is selected solely from published benchmark accuracy.
- InsightFace is an optional comparison candidate. Its code license and pretrained model restrictions are separate; public pretrained models are not an automatic commercial default.

## Delivery phases and evidence gates

1. Scope and environment: confirm users, pilot size, camera, deployment topology and recognition-versus-verification flow. Check Python/runtime and camera access. Hardware query was denied in this session; CPU/GPU/RAM remain unverified. [COMPLETE]
2. Recognition baseline: runnable camera enrollment and matching with quality checks, unknown rejection and adjustable similarity threshold. Save reproducible commands and model manifest. Measure latency on the actual machine before committing a target. [COMPLETE]
3. Data and calibration: collect consented enrollment and independent later-session recordings. Split by capture session before augmentation; avoid frames from the same recording appearing in both calibration and final test. Include non-enrolled volunteers and varied light, pose, glasses and distance. Retain a test set untouched by threshold selection. [IN PROGRESS]
4. Training decision: start with pretrained embeddings and threshold calibration. Fine-tuning is conditional on measured failures and sufficient properly licensed, diverse data. Train a new network only if research scope or evidence justifies it. If fine-tuning occurs, version its data, configuration and weights, and compare it against the unchanged baseline on the same held-out set. [DECISION: PRETRAINED RETAINED]
5. Attendance application: implement enrollment, explicit attendance actions, duplicate prevention, reporting, admin authentication, correction audit and recoverable backups. Recognition tests and event-integrity tests are runnable with synthetic records; camera tests remain separate. [COMPLETE: attendance_db.py, face_id.py kiosk bar, admin_web.py portal, test_attendance.py, test_admin_web.py]
6. Spoof testing: test printed photos and phone replay. Ordinary webcam matching does not prove liveness; blink or head-turn checks alone are insufficient to claim robust anti-spoofing. Choose and evaluate a liveness approach or suitable hardware before unattended deployment. Supervised pilots must identify this limit explicitly. [ANALYSIS DOCUMENTED: presentation-attack-analysis.md]
7. Pilot execution: supervised use, manual fallback, daily reconciliation, restart and restore drills, measured failure review. Expand only after recognition, spoof resistance and record integrity meet agreed criteria. [READY FOR SUPERVISED PILOT]

## Evaluation

Report wrong-person acceptance, unknown-person acceptance, genuine rejection, correct identification, enrollment failures and end-to-end latency. State participant count, attempt count, gallery size and capture conditions. Identification at a kiosk is a one-to-many test; pairwise verification scores alone do not establish its accuracy.

Tune thresholds on calibration data only. Require a minimum top-match score and separation from the next candidate, then assess on untouched sessions. Zero observed false accepts in a small pilot is evidence for that sample only, not proof of zero risk.

Exercise duplicate frame/event processing, wrong clocks, repeat check-ins, missing check-outs, database errors, process restarts, backup restoration and corrections. Define timezone and day-boundary rules once the university/workplace scope is confirmed.

Protect enrollment access and biometric templates; do not commit faces, embeddings, attendance data or secrets to Git. Define retention/deletion, access control and storage protection before collecting a pilot dataset. Disable raw-video retention by default. Biometric embeddings remain sensitive data.

## Immediate continuation

Enroll the first participants interactively, then measure recognition on separate captures. Use calibration sessions to tune similarity/margin and keep an untouched final test. The current local environment uses Python 3.13, OpenCV 5.0.0 and NumPy 2.5.3. Camera 0 returned a 640x480 frame without saving it. Model SHA-256 checks, detector inference on a blank image and finite 128-dimensional SFace output passed. Synthetic matching checks cover known, unknown, ambiguous and invalid references.

## Sources checked

The supplied TimeTec advertising landing page was inaccessible through the browsing tool. The official clocking-options page was read instead:
https://www.timeteccloud.com/attendance/clocking-options

OpenCV recognition documentation and model repositories:
https://docs.opencv.org/5.0/tutorials/dnn/dnn_face/dnn_face.html
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface

InsightFace model-license documentation:
https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md
