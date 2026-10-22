"""Unit tests for head pose estimation and motion variance tracking."""
import unittest
from liveness import extract_landmarks, estimate_pose, MotionTracker


class TestLiveness(unittest.TestCase):
    def test_extract_landmarks(self):
        # Dummy YuNet vector with 15 values
        face_data = [10, 20, 100, 120, 30, 40, 70, 40, 50, 60, 40, 90, 60, 90, 0.99]
        landmarks = extract_landmarks(face_data)
        self.assertIsNotNone(landmarks)
        self.assertEqual(landmarks["right_eye"], (30.0, 40.0))
        self.assertEqual(landmarks["left_eye"], (70.0, 40.0))
        self.assertEqual(landmarks["nose_tip"], (50.0, 60.0))

    def test_estimate_pose_frontal(self):
        # Symmetric frontal face
        landmarks = {
            "right_eye": (30.0, 40.0),
            "left_eye": (70.0, 40.0),
            "nose_tip": (50.0, 65.0),
            "right_mouth": (35.0, 85.0),
            "left_mouth": (65.0, 85.0),
        }
        is_frontal, yaw, pitch, desc = estimate_pose(landmarks)
        self.assertTrue(is_frontal)
        self.assertEqual(desc, "FRONTAL")
        self.assertAlmostEqual(yaw, 0.0, places=2)

    def test_estimate_pose_turned(self):
        # Nose shifted far right -> TURNED_RIGHT
        landmarks = {
            "right_eye": (30.0, 40.0),
            "left_eye": (70.0, 40.0),
            "nose_tip": (65.0, 65.0),
            "right_mouth": (35.0, 85.0),
            "left_mouth": (65.0, 85.0),
        }
        is_frontal, yaw, pitch, desc = estimate_pose(landmarks)
        self.assertFalse(is_frontal)
        self.assertEqual(desc, "TURNED_RIGHT")

    def test_motion_tracker_static_vs_live(self):
        tracker = MotionTracker(window_size=10)

        # 1. Perfectly static coordinates (photo)
        for _ in range(10):
            tracker.add_observation((100.0, 150.0))

        status, std = tracker.evaluate_liveness()
        self.assertEqual(status, "SUSPICIOUS_STATIC")
        self.assertEqual(std, 0.0)

        # 2. Natural subtle jitter
        tracker.reset()
        jitter_points = [
            (100.0, 150.0), (100.5, 150.2), (99.8, 149.9), (100.2, 150.4),
            (100.8, 150.1), (99.6, 150.0), (100.1, 150.3), (100.4, 150.1),
            (99.9, 149.8), (100.3, 150.2)
        ]
        for pt in jitter_points:
            tracker.add_observation(pt)

        status, std = tracker.evaluate_liveness()
        self.assertEqual(status, "NATURAL_MOTION")
        self.assertGreater(std, 0.15)
        self.assertLess(std, 5.0)


if __name__ == "__main__":
    unittest.main()
