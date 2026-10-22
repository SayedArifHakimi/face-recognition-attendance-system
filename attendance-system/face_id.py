"""Local supervised face identification; this prototype does not prove liveness."""
import argparse
import json
import math
import time
from pathlib import Path

from attendance_db import init_db, record_event
from liveness import extract_landmarks, estimate_pose

ROOT = Path(__file__).resolve().parent
GALLERY = ROOT / "data" / "gallery.json"
SETTINGS = ROOT / "data" / "settings.json"
CAPTURE_STEPS = (
    (10.0, "Phase 1: look STRAIGHT at the camera", True),
    (2.0, "PAUSED - turn slightly to YOUR LEFT", False),
    (3.0, "Phase 2: hold slightly LEFT", True),
    (2.0, "PAUSED - turn slightly to YOUR RIGHT", False),
    (3.0, "Phase 2: hold slightly RIGHT", True),
    (2.0, "PAUSED - look slightly UP", False),
    (3.0, "Phase 2: hold slightly UP", True),
    (2.0, "PAUSED - look slightly DOWN", False),
    (3.0, "Phase 2: hold slightly DOWN", True),
    (2.0, "PAUSED - face forward; prepare to blink", False),
    (3.0, "Phase 2: close / open eyes; blink naturally", True),
    (2.0, "PAUSED - prepare to vary expression / position", False),
    (10.0, "Phase 2: smile, move gently; vary light angle", True),
)
CAPTURE_SECONDS = sum(seconds for seconds, _, active in CAPTURE_STEPS if active)
SESSION_SECONDS = sum(seconds for seconds, _, _ in CAPTURE_STEPS)
SAMPLE_RATE = 10


class GuidedCapture:
    def __init__(self, started):
        self.started = started
        self.last_slot = None
        self.samples = []
        self.counts = [0] * len(CAPTURE_STEPS)

    def position(self, now):
        elapsed = max(0, now - self.started)
        if elapsed >= SESSION_SECONDS:
            return None
        boundary = 0
        for step, (seconds, _, _) in enumerate(CAPTURE_STEPS):
            if elapsed < boundary + seconds:
                return step, int((elapsed - boundary) * SAMPLE_RATE)
            boundary += seconds

    def due(self, now):
        position = self.position(now)
        return (position is not None and CAPTURE_STEPS[position[0]][2]
                and position != self.last_slot)

    def update(self, now, embedding):
        position = self.position(now)
        if position is None:
            return None
        step, _ = position
        if embedding is not None and self.due(now):
            self.samples.append(embedding)
            self.counts[step] += 1
            self.last_slot = position
        return step

    def complete(self, early=False):
        return len(self.samples) >= 5 and (early or all(
            count for count, (_, _, active) in zip(self.counts, CAPTURE_STEPS) if active))


def normalize(vector):
    values = [float(v) for v in vector]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Embedding must contain finite numbers")
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        raise ValueError("Embedding cannot be zero")
    return [v / norm for v in values]


def identify(vector, gallery, threshold=0.5, margin=0.08):
    query = normalize(vector)
    scores = []
    # ponytail: linear gallery scan for a home pilot; benchmark vector search when the gallery grows.
    for name, references in gallery.items():
        if not references:
            raise ValueError(f"No references for {name}")
        similarities = []
        for reference in references:
            reference = normalize(reference)
            if len(query) != len(reference):
                raise ValueError("Embedding dimensions do not match")
            similarities.append(sum(a * b for a, b in zip(query, reference)))
        scores.append((max(similarities), name))
    scores.sort(reverse=True)
    if not scores:
        return "Unknown", 0.0
    best, name = scores[0]
    ambiguous = len(scores) > 1 and best - scores[1][0] < margin
    return (name if best >= threshold and not ambiguous else "Unknown"), best


def read_gallery():
    if not GALLERY.exists():
        return {}
    gallery = json.loads(GALLERY.read_text(encoding="utf-8"))
    if not isinstance(gallery, dict):
        raise ValueError("Invalid gallery")
    for name, references in gallery.items():
        if not isinstance(name, str) or not isinstance(references, list) or not references:
            raise ValueError("Invalid enrollment")
        for reference in references:
            if len(normalize(reference)) != 128:
                raise ValueError("Expected 128-dimensional SFace references")
    return gallery


def save_gallery(gallery):
    GALLERY.parent.mkdir(exist_ok=True)
    temporary = GALLERY.with_suffix(".tmp")
    temporary.write_text(json.dumps(gallery), encoding="utf-8")
    temporary.replace(GALLERY)


def validate_settings(values):
    try:
        settings = {"threshold": float(values["threshold"]), "margin": float(values["margin"]),
                    "min_face": int(str(values["min_face"])), "min_sharpness": float(values["min_sharpness"])}
    except (KeyError, ValueError, TypeError, OverflowError):
        raise ValueError("Enter numbers for all settings; minimum face size must be a whole number") from None
    if not (-1 <= settings["threshold"] <= 1 and 0 <= settings["margin"] <= 2):
        raise ValueError("Similarity threshold: -1 to 1; ambiguity margin: 0 to 2")
    if settings["min_face"] < 1 or not math.isfinite(settings["min_sharpness"]) or settings["min_sharpness"] < 0:
        raise ValueError("Minimum face size must be positive; sharpness must be finite and nonnegative")
    return settings


def read_settings():
    defaults = {"threshold": 0.5, "margin": 0.08, "min_face": 80, "min_sharpness": 50}
    if SETTINGS.exists():
        values = json.loads(SETTINGS.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Invalid settings file")
        defaults.update(values)
    return validate_settings(defaults)


def save_settings(settings):
    settings = validate_settings(settings)
    SETTINGS.parent.mkdir(exist_ok=True)
    temporary = SETTINGS.with_suffix(".tmp")
    temporary.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    temporary.replace(SETTINGS)


def enrollment_name(value, gallery, replace=False):
    name = value.strip()
    if not name or name.casefold() == "unknown" or not name.isascii() or len(name) > 60 or not name.isprintable():
        raise ValueError("Enter a printable name of 1..60 ASCII characters, other than Unknown")
    existing = next((key for key in gallery if key.casefold() == name.casefold()), None)
    if existing and not replace:
        raise ValueError("Name already enrolled: tick Replace existing enrollment to recapture")
    return existing or name


class FaceWindow:
    def __init__(self, root, camera, args):
        import cv2
        import tkinter as tk
        from tkinter import ttk
        self.root, self.camera, self.args = root, camera, args
        self.mode, self.capture, self.ready = "check", None, False
        self.active_person, self.active_score = None, None
        self.kiosk_banner = None
        self.pending_delete = None
        self.gallery = read_gallery()
        self.detector = cv2.FaceDetectorYN.create(str(ROOT / "models/yunet.onnx"), "", (320, 320), 0.9)
        self.recognizer = cv2.FaceRecognizerSF.create(str(ROOT / "models/sface.onnx"), "")
        root.title("Face ID - Live Monitor")
        root.minsize(680, 760)
        panel = ttk.Frame(root, padding=12)
        panel.pack(fill="both", expand=True)
        nav = ttk.Frame(panel)
        nav.pack(fill="x", pady=(0, 8))
        self.check_button = ttk.Button(nav, text="Live Face ID", command=self.select_check)
        self.check_button.pack(side="left", padx=(0, 8))
        self.capture_button = ttk.Button(nav, text="Capture Samples", command=self.select_capture)
        self.capture_button.pack(side="left", padx=(0, 8))
        self.admin_button = ttk.Button(nav, text="Admin / Settings", command=self.select_admin)
        self.admin_button.pack(side="left")
        self.heading = tk.StringVar(value="Live Face ID")
        ttk.Label(panel, textvariable=self.heading, font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=5)
        self.video = ttk.Label(panel, anchor="center")
        self.video.pack(fill="both", expand=True)
        self.quality = tk.StringVar(value="Opening camera...")
        self.message = tk.StringVar(value="Live recognition runs automatically")
        self.detail = tk.StringVar(value=f"{len(self.gallery)} enrolled people | Local supervised prototype")
        self.pages = {key: ttk.Frame(panel) for key in ("check", "enroll", "admin")}
        ttk.Label(self.pages["check"], text="Live camera: names appear above face boxes. Orange boxes indicate Unknown.").pack(anchor="w")

        # Attendance Action Panel
        kiosk = ttk.LabelFrame(self.pages["check"], text="Attendance Action", padding=6)
        kiosk.pack(fill="x", pady=4)
        kiosk_row = ttk.Frame(kiosk)
        kiosk_row.pack(fill="x")
        self.clock_in_button = ttk.Button(kiosk_row, text="Clock In (I)", command=lambda: self.clock_attendance("CHECK_IN"))
        self.clock_in_button.pack(side="left", padx=(0, 8))
        self.clock_out_button = ttk.Button(kiosk_row, text="Clock Out (O)", command=lambda: self.clock_attendance("CHECK_OUT"))
        self.clock_out_button.pack(side="left", padx=(0, 8))
        self.kiosk_status = tk.StringVar(value="Looking for recognized face to clock in/out...")
        ttk.Label(kiosk, textvariable=self.kiosk_status).pack(anchor="w", pady=(3, 0))

        fields = ttk.Frame(self.pages["enroll"])
        fields.pack(fill="x", pady=5)
        ttk.Label(fields, text="Name:").pack(side="left")
        self.name = tk.StringVar(value=args.enroll or "")
        self.entry = ttk.Entry(fields, textvariable=self.name, width=30)
        self.entry.pack(side="left", padx=8)
        self.replace = tk.BooleanVar(value=args.replace)
        self.replace_box = ttk.Checkbutton(fields, text="Replace existing enrollment", variable=self.replace)
        self.replace_box.pack(side="left")
        self.progress = ttk.Progressbar(self.pages["enroll"], maximum=SESSION_SECONDS)
        self.progress.pack(fill="x", pady=5)
        controls = ttk.Frame(self.pages["enroll"])
        controls.pack(fill="x", pady=5)
        self.start_button = ttk.Button(controls, text="Start Capture (SPACE)", command=self.start)
        self.start_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(controls, text="Stop and Save (ENTER)", command=self.stop)
        self.stop_button.pack(side="left", padx=(0, 8))
        self.cancel_button = ttk.Button(controls, text="Cancel Capture (ESC)", command=self.cancel)
        self.cancel_button.pack(side="left")
        admin = self.pages["admin"]
        settings_frame = ttk.LabelFrame(admin, text="Recognition settings", padding=6)
        settings_frame.pack(fill="x", pady=4)
        self.setting_vars = {}
        for column, (key, label) in enumerate((("threshold", "Match threshold"), ("margin", "Ambiguity margin"),
                                               ("min_face", "Min face pixels"), ("min_sharpness", "Min sharpness"))):
            ttk.Label(settings_frame, text=label).grid(row=0, column=column, padx=4)
            self.setting_vars[key] = tk.StringVar(value=str(getattr(args, key)))
            ttk.Entry(settings_frame, textvariable=self.setting_vars[key], width=14).grid(row=1, column=column, padx=4)
        ttk.Button(settings_frame, text="Save Settings", command=self.apply_settings).grid(row=2, column=0, columnspan=4, pady=5)
        enrolled = ttk.LabelFrame(admin, text="Enrolled people / stored references", padding=6)
        enrolled.pack(fill="x", pady=4)
        self.people = tk.Listbox(enrolled, height=3, exportselection=False)
        self.people.pack(fill="x")
        delete_actions = ttk.Frame(enrolled)
        delete_actions.pack(fill="x", pady=5)
        ttk.Button(delete_actions, text="Remove Selected", command=self.request_delete).pack(side="left", padx=(0, 8))
        ttk.Button(delete_actions, text="Remove All Enrollments", command=lambda: self.request_delete(all_people=True)).pack(side="left")
        ttk.Label(enrolled, text="Deletion confirmation: type DELETE, then click Confirm Delete").pack(anchor="w")
        confirm = ttk.Frame(enrolled)
        confirm.pack(fill="x")
        self.delete_text = tk.StringVar()
        ttk.Entry(confirm, textvariable=self.delete_text, width=20).pack(side="left", padx=(0, 8))
        ttk.Button(confirm, text="Confirm Delete", command=self.confirm_delete).pack(side="left")
        self.page = "check"
        self.pages["check"].pack(fill="x", pady=5)
        for variable in (self.quality, self.message, self.detail):
            ttk.Label(panel, textvariable=variable, wraplength=650).pack(fill="x", pady=3)
        root.bind("<Return>", lambda event: self.stop())
        root.bind("<Escape>", lambda event: self.cancel())
        root.bind("<space>", lambda event: self.start() if event.widget != self.entry else None)
        root.bind("<i>", lambda event: self.clock_attendance("CHECK_IN") if self.mode == "check" and self.active_person else None)
        root.bind("<I>", lambda event: self.clock_attendance("CHECK_IN") if self.mode == "check" and self.active_person else None)
        root.bind("<o>", lambda event: self.clock_attendance("CHECK_OUT") if self.mode == "check" and self.active_person else None)
        root.bind("<O>", lambda event: self.clock_attendance("CHECK_OUT") if self.mode == "check" and self.active_person else None)
        self.refresh_people()
        if args.enroll:
            self.select_capture()
        self.sync_controls()

    def clock_attendance(self, event_type):
        if self.capture is not None or self.mode != "check" or not self.active_person:
            return
        device = getattr(self.args, "device_id", "KIOSK-01")
        ok, msg, _ = record_event(
            person_name=self.active_person,
            event_type=event_type,
            similarity_score=self.active_score,
            device_id=device
        )
        self.kiosk_status.set(msg)
        if ok:
            action_label = "CHECKED IN" if event_type == "CHECK_IN" else "CHECKED OUT"
            self.kiosk_banner = (f"[OK] {action_label}: {self.active_person}", time.monotonic(), (20, 140, 20))
        else:
            self.kiosk_banner = (f"[ALERT] {msg}", time.monotonic(), (0, 0, 180))

    def show_page(self, page):
        for frame in self.pages.values():
            frame.pack_forget()
        # Insert page controls before status labels while keeping the shared live preview.
        self.pages[page].pack(fill="x", pady=5, after=self.video)
        self.page = page
        self.heading.set({"check": "Live Face ID", "enroll": "Capture Samples", "admin": "Admin / Settings"}[page])
        self.pending_delete = None
        self.delete_text.set("")

    def refresh_people(self):
        self.person_names = sorted(self.gallery)
        self.people.delete(0, "end")
        for name in self.person_names:
            self.people.insert("end", f"{name} - {len(self.gallery[name])} references")

    def select_admin(self):
        if self.capture is not None:
            return
        self.mode = "check"
        self.show_page("admin")
        self.refresh_people()
        self.message.set("Live recognition continues. Settings apply immediately after saving")
        self.detail.set("Deletion removes enrolled references only; type DELETE to confirm")
        self.sync_controls()

    def apply_settings(self):
        if self.capture is not None:
            return
        try:
            settings = validate_settings({key: value.get() for key, value in self.setting_vars.items()})
            save_settings(settings)
        except (ValueError, OSError) as error:
            self.detail.set(f"Settings not saved: {error}")
            return
        for key, value in settings.items():
            setattr(self.args, key, value)
        self.detail.set("Settings saved and applied; they will be loaded next time")

    def request_delete(self, all_people=False):
        if self.capture is not None:
            return
        selection = self.people.curselection()
        names = tuple(self.gallery) if all_people else tuple(self.person_names[index] for index in selection)
        self.pending_delete = names or None
        self.delete_text.set("")
        self.detail.set(f"Remove {', '.join(names)}? Type DELETE and click Confirm Delete" if names else "Select an enrolled person first")

    def confirm_delete(self):
        if self.capture is not None or not self.pending_delete:
            return
        if self.delete_text.get() != "DELETE":
            self.detail.set("Deletion cancelled: type DELETE exactly to confirm")
            return
        updated = {name: references for name, references in self.gallery.items() if name not in self.pending_delete}
        try:
            save_gallery(updated)
        except OSError as error:
            self.detail.set(f"Deletion failed; existing data kept: {error}")
            return
        removed = len(self.gallery) - len(updated)
        self.gallery, self.pending_delete = updated, None
        self.delete_text.set("")
        self.refresh_people()
        self.detail.set(f"Removed {removed} enrollment(s). Models and other project files are untouched")

    def sync_controls(self):
        busy = self.capture is not None
        for widget in (self.entry, self.replace_box, self.capture_button, self.check_button, self.admin_button):
            widget.configure(state="disabled" if busy else "normal")
        self.start_button.configure(state="normal" if self.mode == "enroll" and self.ready and not busy else "disabled")
        self.stop_button.configure(state="normal" if busy else "disabled")
        self.cancel_button.configure(state="normal" if busy else "disabled")
        kiosk_ready = not busy and self.mode == "check" and self.active_person is not None
        self.clock_in_button.configure(
            state="normal" if kiosk_ready else "disabled",
            text=f"Clock In: {self.active_person} (I)" if self.active_person else "Clock In (I)"
        )
        self.clock_out_button.configure(
            state="normal" if kiosk_ready else "disabled",
            text=f"Clock Out: {self.active_person} (O)" if self.active_person else "Clock Out (O)"
        )

    def select_capture(self):
        if self.capture is not None:
            return
        self.mode, self.ready = "enroll", False
        self.show_page("enroll")
        self.message.set("Enter a name, look straight and center your whole face; then click Start Capture")
        self.detail.set(f"{CAPTURE_SECONDS:g}s capture + {SESSION_SECONDS - CAPTURE_SECONDS:g}s guidance; target {SAMPLE_RATE} samples/s")
        self.progress["value"] = 0
        self.sync_controls()

    def select_check(self):
        if self.capture is not None:
            return
        self.mode = "check"
        self.show_page("check")
        self.message.set("Live recognition runs automatically" if self.gallery else "No enrollments yet; detected faces will be Unknown")
        self.detail.set(f"{len(self.gallery)} enrolled people | Similarity is not a confidence percentage")
        self.progress["value"] = 0
        self.sync_controls()

    def start(self):
        if self.capture is not None or self.mode != "enroll" or not self.ready:
            return
        try:
            self.capture_name = enrollment_name(self.name.get(), self.gallery, self.replace.get())
        except ValueError as error:
            self.message.set(str(error))
            return
        self.capture = GuidedCapture(time.monotonic())
        self.sync_controls()

    def stop(self, early=True):
        if self.capture is None:
            return
        if not self.capture.complete(early=early):
            self.capture = None
            self.message.set("Stopped without saving: too few samples" if early else "Missed a capture step: click Start Capture to retry")
            self.detail.set("Existing enrollment is unchanged")
            self.sync_controls()
            return
        updated = {**self.gallery, self.capture_name: self.capture.samples}
        try:
            save_gallery(updated)
        except OSError as error:
            self.message.set(f"Save failed: {error}. Samples retained; try Stop and Save again")
            self.sync_controls()
            return
        count = len(self.capture.samples)
        self.gallery, self.capture = updated, None
        self.refresh_people()
        self.replace.set(False)
        self.message.set(f"Saved {count} references for {self.capture_name}. Click Face ID Check")
        self.detail.set("Finished early; guided coverage may be incomplete" if early else "All guided capture steps completed")
        self.sync_controls()

    def cancel(self):
        if self.capture is None:
            return
        self.capture = None
        self.message.set("Capture cancelled; existing enrollment is unchanged")
        self.detail.set("Click Start Capture to retry")
        self.progress["value"] = 0
        self.sync_controls()

    def tick(self):
        import cv2
        import tkinter as tk
        try:
            ok, frame = self.camera.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames; close and reopen the app")
            frame_time = time.monotonic()
            self.detector.setInputSize((frame.shape[1], frame.shape[0]))
            _, faces = self.detector.detect(frame)
            embedding = None
            self.ready = False
            self.last_labels = []
            quality = "No face detected" if self.mode == "check" else "Show exactly one face"
            labels = []
            recognized_candidates = []
            single_face_landmarks = None

            if faces is not None and len(faces) == 1:
                single_face_landmarks = extract_landmarks(faces[0])

            for face in ([] if faces is None else faces):
                x, y, w, h = (int(v) for v in face[:4])
                identity = "Unknown"
                aligned = self.recognizer.alignCrop(frame, face)
                sharpness = cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                valid = min(w, h) >= self.args.min_face and sharpness >= self.args.min_sharpness
                score = 0.0
                if self.mode == "check":
                    if valid and self.gallery:
                        feature = normalize(self.recognizer.feature(aligned).ravel())
                        identity, score = identify(feature, self.gallery, self.args.threshold, self.args.margin)
                    labels.append(identity)
                    quality = f"{len(faces)} face(s) detected"
                elif len(faces) == 1:
                    quality = "Move closer / improve lighting and sharpness"
                    if valid:
                        self.ready = (x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]
                                      and abs(x + w / 2 - frame.shape[1] / 2) < frame.shape[1] * 0.2
                                      and abs(y + h / 2 - frame.shape[0] / 2) < frame.shape[0] * 0.2)
                        quality = "Clear face" if self.capture else ("Ready" if self.ready else "Center your whole face")
                        if self.capture and self.capture.due(frame_time):
                            embedding = normalize(self.recognizer.feature(aligned).ravel())
                # Draw only after inference for every face, avoiding overlay pixels in sibling crops.
                self.last_labels.append((identity, x, y, w, h))
                recognized_candidates.append((identity, score))

            if self.mode == "check":
                known = [c for c in recognized_candidates if c[0] != "Unknown"]
                if len(known) == 1:
                    self.active_person, self.active_score = known[0]
                    pose_advice = ""
                    if single_face_landmarks:
                        is_frontal, _, _, desc = estimate_pose(single_face_landmarks)
                        if not is_frontal and desc != "INVALID_GEOMETRY":
                            pose_advice = f" (Look forward: {desc.replace('_', ' ').lower()})"
                    self.kiosk_status.set(f"Ready: {self.active_person} (Score: {self.active_score:.2f}){pose_advice}")
                else:
                    self.active_person, self.active_score = None, None
                    if len(known) > 1:
                        self.kiosk_status.set("Multiple recognized faces: only one face allowed to clock")
                    elif faces is not None and len(faces) > 0:
                        self.kiosk_status.set("Unrecognized face: cannot clock attendance")
                    else:
                        self.kiosk_status.set("Looking for recognized face to clock in/out...")
            else:
                self.active_person, self.active_score = None, None

            for identity, x, y, w, h in self.last_labels:
                color = (0, 200, 0) if identity != "Unknown" else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                scale = min(0.6, (frame.shape[1] - 20) / max(1, cv2.getTextSize(identity, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]))
                text_width = cv2.getTextSize(identity, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
                text_x = max(0, min(x, frame.shape[1] - text_width - 5))
                cv2.putText(frame, identity, (text_x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

            if self.kiosk_banner:
                banner_text, banner_time, banner_color = self.kiosk_banner
                if time.monotonic() - banner_time < 3.0:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), banner_color, -1)
                    cv2.putText(frame, banner_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    self.kiosk_banner = None

            if self.mode == "check":
                self.message.set(" | ".join(labels) if labels else "Waiting for faces")
            self.quality.set(quality)
            if self.capture:
                step = self.capture.update(frame_time, embedding)
                if step is None:
                    self.stop(early=False)
                else:
                    elapsed = min(SESSION_SECONDS, time.monotonic() - self.capture.started)
                    self.message.set(CAPTURE_STEPS[step][1])
                    self.detail.set(f"{SESSION_SECONDS - elapsed:.1f}s left | {len(self.capture.samples)} samples | ENTER to finish")
                    self.progress["value"] = elapsed
            ok, ppm = cv2.imencode(".ppm", frame)
            if not ok:
                raise RuntimeError("Cannot display camera frame")
            self.photo = tk.PhotoImage(master=self.root, data=ppm.tobytes(), format="PPM")
            self.video.configure(image=self.photo)
            self.sync_controls()
        except (OSError, ValueError, RuntimeError, cv2.error) as error:
            self.camera.release()
            self.ready = False
            self.start_button.configure(state="disabled")
            self.quality.set(f"Camera/inference error: {error}")
            self.detail.set("Close and reopen the app. Unsaved samples remain in memory until closing")
            return
        self.root.after(10, self.tick)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enroll", help="Prefill the name and select Capture Samples")
    parser.add_argument("--replace", action="store_true", help="Allow replacing existing enrollment after successful capture")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device-id", default="KIOSK-01", help="Device identifier for attendance records")
    defaults = read_settings()
    parser.add_argument("--threshold", type=float, default=defaults["threshold"])
    parser.add_argument("--margin", type=float, default=defaults["margin"])
    parser.add_argument("--min-face", type=int, default=defaults["min_face"])
    parser.add_argument("--min-sharpness", type=float, default=defaults["min_sharpness"])
    parser.add_argument("--check-camera", action="store_true")
    args = parser.parse_args()
    if args.replace and not args.enroll:
        parser.error("--replace requires --enroll")
    if not (-1 <= args.threshold <= 1 and 0 <= args.margin <= 2):
        parser.error("Threshold must be -1..1 and margin 0..2")
    if args.min_face < 1 or not math.isfinite(args.min_sharpness) or args.min_sharpness < 0:
        parser.error("Quality limits must be positive/finite")

    init_db()
    import cv2
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    try:
        if args.check_camera:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera unavailable; close other camera apps or try --camera 1")
            print(f"Camera {args.camera}: {frame.shape[1]}x{frame.shape[0]}; no frame saved")
            return
        import tkinter as tk
        root = tk.Tk()
        try:
            if not camera.isOpened():
                raise RuntimeError("Camera unavailable; close other camera apps or try --camera 1")
            window = FaceWindow(root, camera, args)
            window.tick()
        except (OSError, ValueError, RuntimeError, cv2.error) as error:
            for child in root.winfo_children():
                child.destroy()
            tk.Label(root, text=str(error), wraplength=600, padx=20, pady=20).pack()
        root.mainloop()
    finally:
        camera.release()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error))
