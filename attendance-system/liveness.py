"""Landmark-based head pose estimation and micro-motion heuristics for kiosk verification."""
import math
from collections import deque
from typing import Tuple, List, Optional, Dict, Any


def extract_landmarks(face_data: Any) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Extract 5 facial landmarks from YuNet detection output.
    YuNet face vector: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
    """
    if face_data is None or len(face_data) < 14:
        return None
    return {
        "right_eye": (float(face_data[4]), float(face_data[5])),
        "left_eye": (float(face_data[6]), float(face_data[7])),
        "nose_tip": (float(face_data[8]), float(face_data[9])),
        "right_mouth": (float(face_data[10]), float(face_data[11])),
        "left_mouth": (float(face_data[12]), float(face_data[13])),
    }


def estimate_pose(landmarks: Dict[str, Tuple[float, float]]) -> Tuple[bool, float, float, str]:
    """
    Estimate head orientation (yaw & pitch) from 5 facial landmarks.
    Returns (is_frontal, yaw_ratio, pitch_ratio, orientation_label).
    """
    x_re, y_re = landmarks["right_eye"]
    x_le, y_le = landmarks["left_eye"]
    x_nt, y_nt = landmarks["nose_tip"]
    x_rc, y_rc = landmarks["right_mouth"]
    x_lc, y_lc = landmarks["left_mouth"]

    eye_span = math.hypot(x_le - x_re, y_le - y_re)
    if eye_span < 1e-4:
        return False, 0.0, 0.0, "INVALID_GEOMETRY"

    eye_center_x = (x_re + x_le) / 2.0
    eye_center_y = (y_re + y_le) / 2.0
    mouth_center_y = (y_rc + y_lc) / 2.0
    facial_height = abs(mouth_center_y - eye_center_y)

    if facial_height < 1e-4:
        return False, 0.0, 0.0, "INVALID_GEOMETRY"

    # Yaw ratio: normalized horizontal nose offset from inter-ocular midpoint
    # Zero represents centered frontal; negative indicates turned left; positive turned right
    yaw_ratio = (x_nt - eye_center_x) / (eye_span / 2.0)

    # Pitch ratio: relative vertical position of nose tip between eye line and mouth line
    # Normal frontal range is ~0.45 to 0.75
    pitch_ratio = (y_nt - eye_center_y) / facial_height

    is_frontal = (-0.35 <= yaw_ratio <= 0.35) and (0.40 <= pitch_ratio <= 0.78)

    if not is_frontal:
        if yaw_ratio < -0.35:
            desc = "TURNED_LEFT"
        elif yaw_ratio > 0.35:
            desc = "TURNED_RIGHT"
        elif pitch_ratio < 0.40:
            desc = "TILTED_UP"
        else:
            desc = "TILTED_DOWN"
    else:
        desc = "FRONTAL"

    return is_frontal, round(yaw_ratio, 3), round(pitch_ratio, 3), desc


class MotionTracker:
    """Tracks landmark jitter over a rolling window to detect static photos vs natural motion."""

    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)

    def add_observation(self, nose_tip: Tuple[float, float]) -> None:
        self.history.append(nose_tip)

    def reset(self) -> None:
        self.history.clear()

    def evaluate_liveness(self) -> Tuple[str, float]:
        """
        Evaluate coordinate variance over the rolling history.
        Returns (status_label, standard_deviation).
        """
        if len(self.history) < self.window_size:
            return "COLLECTING_BUFFER", 0.0

        xs = [pt[0] for pt in self.history]
        ys = [pt[1] for pt in self.history]

        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)

        var_x = sum((x - mean_x) ** 2 for x in xs) / len(xs)
        var_y = sum((y - mean_y) ** 2 for y in ys) / len(ys)
        std_dev = math.sqrt(var_x + var_y)

        # Static photo held still typically has std_dev < 0.15 pixels
        if std_dev < 0.15:
            return "SUSPICIOUS_STATIC", round(std_dev, 3)
        # Extreme sudden movement
        elif std_dev > 25.0:
            return "EXCESSIVE_MOTION", round(std_dev, 3)
        else:
            return "NATURAL_MOTION", round(std_dev, 3)
