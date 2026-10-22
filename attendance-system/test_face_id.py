from face_id import CAPTURE_SECONDS, CAPTURE_STEPS, SESSION_SECONDS, SAMPLE_RATE, GuidedCapture, enrollment_name, identify, normalize


def main():
    gallery = {"Alice": [[1, 0, 0]], "Bob": [[0, 1, 0]]}
    assert identify([2, 0, 0], gallery)[0] == "Alice"
    assert identify([0, 3, 0], gallery)[0] == "Bob"
    assert identify([0, 0, 1], gallery)[0] == "Unknown"
    assert identify([1, 1, 0], gallery)[0] == "Unknown"
    assert identify([1, 0, 0], {})[0] == "Unknown"
    assert identify([1, 0, 0], {"Alice": [[0, 1, 0], [1, 0, 0]]})[0] == "Alice"
    for bad in [[], [0, 0], [float("nan"), 1], [float("inf"), 1]]:
        try:
            normalize(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid embedding accepted")
    try:
        identify([1, 0], gallery)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched dimensions accepted")
    assert enrollment_name("  New Person  ", gallery) == "New Person"
    assert enrollment_name("alice", gallery, replace=True) == "Alice"
    for value in ["", "Unknown", "a" * 61, "Alice"]:
        try:
            enrollment_name(value, gallery)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid or duplicate enrollment accepted")
    assert CAPTURE_SECONDS == 35
    assert SESSION_SECONDS == 47
    assert SAMPLE_RATE == 10
    capture = GuidedCapture(100)
    assert capture.update(100, [1, 0]) == 0
    capture.update(100.05, [1, 0])
    assert len(capture.samples) == 1
    assert capture.update(100.15, [1, 0]) == 0
    assert len(capture.samples) == 2
    assert capture.update(110, [1, 0]) == 1
    assert not capture.due(111)
    assert len(capture.samples) == 2  # Guidance pauses never collect samples.
    for moment, step in [(112, 2), (117, 4), (122, 6), (127, 8), (132, 10), (137, 12)]:
        assert capture.update(moment, [1, 0]) == step
    assert capture.complete()
    count = len(capture.samples)
    assert capture.update(147, [1, 0]) is None
    assert len(capture.samples) == count
    missed = GuidedCapture(0)
    for moment in [0, 0.15, 0.25, 0.35, 0.45]:
        missed.update(moment, [1, 0])
    missed.update(12, None)
    assert len(missed.samples) == 5 and not missed.complete()
    assert missed.complete(early=True)  # ENTER allows partial guided coverage.
    assert missed.update(47, [1, 0]) is None
    assert not GuidedCapture(0).complete(early=True)
    short = GuidedCapture(0)
    short.update(0, [1, 0])
    assert not short.complete(early=True)
    full = GuidedCapture(0)
    for tick in range(int(SESSION_SECONDS * 100)):
        full.update(tick / 100, [1, 0])
    assert len(full.samples) == 350
    assert full.complete()
    assert [count for count, (_, _, active) in zip(full.counts, CAPTURE_STEPS) if active] == [100, 30, 30, 30, 30, 30, 100]
    assert all(count == 0 for count, (_, _, active) in zip(full.counts, CAPTURE_STEPS) if not active)
    stalled = GuidedCapture(0)
    stalled.update(0, [1, 0])
    stalled.update(9.95, [1, 0])
    assert len(stalled.samples) == 2  # Slow inference never duplicates missed slots.
    print("PASS: matching, timed phases, 10 samples/s slots, pauses, deadline and early finish")



def check_window():
    import tkinter as tk
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    from face_id import FaceWindow, read_gallery, read_settings, save_gallery, validate_settings

    class Camera:
        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            pass

    args = SimpleNamespace(enroll=None, replace=False, min_face=80,
                           min_sharpness=50, threshold=0.5, margin=0.08)
    with tempfile.TemporaryDirectory() as folder, patch("face_id.GALLERY", Path(folder) / "gallery.json"), patch("face_id.SETTINGS", Path(folder) / "settings.json"):
        root = tk.Tk()
        root.withdraw()
        try:
            window = FaceWindow(root, Camera(), args)
            assert window.mode == "check" and window.page == "check"
            window.tick()  # Actual model inference and Tk camera-image rendering.
            assert window.photo.width() == 640 and window.photo.height() == 480
            window.select_check()
            assert window.mode == "check" and not window.gallery
            window.name.set("Test Person")
            window.select_capture()
            assert window.mode == "enroll" and window.page == "enroll"
            assert window.pages["enroll"].winfo_manager() == "pack"
            assert window.pages["check"].winfo_manager() == ""
            window.ready = True
            window.start()
            assert window.capture is not None
            window.select_check()
            assert window.mode == "enroll"  # No silent loss while switching modes.
            vector = [1] + [0] * 127
            started = window.capture.started
            for seconds in [0, 0.15, 0.25, 0.35, 0.45]:
                window.capture.update(started + seconds, vector)
            window.stop()
            assert read_gallery() == {"Test Person": [vector] * 5}
            assert window.capture is None and root.winfo_exists()
            window.select_check()
            assert window.mode == "check"
            window.replace.set(True)
            window.select_capture()
            window.ready = True
            window.start()
            window.capture.update(window.capture.started, vector)
            window.cancel()
            assert read_gallery() == {"Test Person": [vector] * 5}
            window.start()
            assert window.capture is not None
            window.stop()  # Too few references: stop without overwriting.
            assert window.capture is None and read_gallery() == {"Test Person": [vector] * 5}
            with patch("face_id.save_gallery", side_effect=OSError("test disk failure")):
                window.start()
                for seconds in [0, 0.15, 0.25, 0.35, 0.45]:
                    window.capture.update(window.capture.started + seconds, vector)
                window.stop()
                assert window.capture is not None  # Retain samples for save retry.
                assert read_gallery() == {"Test Person": [vector] * 5}
            window.cancel()
            window.select_admin()
            assert window.page == "admin" and window.mode == "check"
            window.setting_vars["threshold"].set("nan")
            window.apply_settings()
            assert not (Path(folder) / "settings.json").exists()
            assert args.threshold == 0.5
            window.setting_vars["threshold"].set("0.6")
            window.apply_settings()
            assert read_settings()["threshold"] == args.threshold == 0.6
            with patch("face_id.save_settings", side_effect=OSError("test disk failure")):
                window.setting_vars["threshold"].set("0.7")
                window.apply_settings()
                assert args.threshold == 0.6
            for key, value in [("margin", "3"), ("min_face", "0"), ("min_face", "2.5"), ("min_sharpness", "inf")]:
                values = read_settings()
                values[key] = value
                try:
                    validate_settings(values)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Invalid settings accepted")
            # Verify names are actually drawn above boxes for multiple faces.
            from unittest.mock import Mock
            import cv2
            noise = np.random.default_rng(0).integers(0, 256, (112, 112, 3), dtype=np.uint8)
            faces = np.array([[20, 50, 120, 140] + [0] * 10 + [0.99],
                              [250, 50, 120, 140] + [0] * 10 + [0.99]], dtype=np.float32)
            detector = Mock()
            detector.detect.return_value = (2, faces)
            recognizer = Mock()
            recognizer.alignCrop.return_value = noise
            recognizer.feature.side_effect = [np.array([vector], dtype=np.float32),
                                             np.array([[0, 1] + [0] * 126], dtype=np.float32)]
            with patch.object(window, "detector", detector), patch.object(window, "recognizer", recognizer), patch("cv2.putText", wraps=cv2.putText) as draw:
                window.tick()
                assert [call.args[1] for call in draw.call_args_list] == ["Test Person", "Unknown"]
                assert [label[0] for label in window.last_labels] == ["Test Person", "Unknown"]
            # Confirmation, selected removal, all removal and failed write keep the right data.
            window.gallery["Second"] = [vector]
            save_gallery(window.gallery)
            window.refresh_people()
            window.people.selection_set(window.person_names.index("Test Person"))
            window.request_delete()
            window.confirm_delete()
            assert len(read_gallery()) == 2  # No DELETE confirmation yet.
            window.delete_text.set("DELETE")
            with patch("face_id.save_gallery", side_effect=OSError("test disk failure")):
                window.confirm_delete()
                assert len(read_gallery()) == len(window.gallery) == 2
            window.confirm_delete()
            assert read_gallery() == window.gallery == {"Second": [vector]}
            window.request_delete(all_people=True)
            window.delete_text.set("DELETE")
            window.confirm_delete()
            assert read_gallery() == window.gallery == {}
            recognizer.feature.reset_mock()
            with patch.object(window, "detector", detector), patch.object(window, "recognizer", recognizer), patch("cv2.putText", wraps=cv2.putText) as draw:
                window.tick()
                assert [call.args[1] for call in draw.call_args_list] == ["Unknown", "Unknown"]
                recognizer.feature.assert_not_called()
            window.select_check()
            assert window.page == "check" and window.mode == "check"
            assert not any(isinstance(child, tk.Toplevel) for child in root.winfo_children())
        finally:
            for task in root.tk.call("after", "info"):
                root.after_cancel(task)
            root.destroy()
    print("PASS: single window, default live check, capture/admin pages, settings, confirmed deletion, box labels and save failures")



if __name__ == "__main__":
    main()
    check_window()
