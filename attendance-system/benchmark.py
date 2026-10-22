"""Automated latency and gallery scale benchmark harness for edge face recognition."""
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


def run_benchmark(iterations: int = 50):
    import cv2
    from face_id import normalize, identify

    print("=" * 65)
    print("  FACE RECOGNITION EDGE INFERENCE & LATENCY BENCHMARK")
    print("=" * 65)

    yunet_path = str(MODELS_DIR / "yunet.onnx")
    sface_path = str(MODELS_DIR / "sface.onnx")

    detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.9)
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    detector.setInputSize((640, 480))

    # Warmup
    for _ in range(5):
        detector.detect(dummy_frame)
        dummy_aligned = np.zeros((112, 112, 3), dtype=np.uint8)
        recognizer.feature(dummy_aligned)

    # 1. Detector Latency
    det_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.detect(dummy_frame)
        det_times.append((time.perf_counter() - t0) * 1000)

    mean_det = np.mean(det_times)
    p95_det = np.percentile(det_times, 95)
    fps_det = 1000.0 / mean_det

    # 2. Embedding Extraction Latency
    rec_times = []
    dummy_crop = np.zeros((112, 112, 3), dtype=np.uint8)
    for _ in range(iterations):
        t0 = time.perf_counter()
        recognizer.feature(dummy_crop)
        rec_times.append((time.perf_counter() - t0) * 1000)

    mean_rec = np.mean(rec_times)
    p95_rec = np.percentile(rec_times, 95)

    # Total Pipeline Latency
    total_latency = mean_det + mean_rec
    pipelined_fps = 1000.0 / total_latency

    print("\n[Stage 1: Neural Network Inference]")
    print(f"  YuNet Detection (640x480):  Mean = {mean_det:6.2f} ms | P95 = {p95_det:6.2f} ms | {fps_det:5.1f} FPS")
    print(f"  SFace Embeddings (112x112): Mean = {mean_rec:6.2f} ms | P95 = {p95_rec:6.2f} ms")
    print(f"  Total Single-Face Pipeline: Mean = {total_latency:6.2f} ms | P95 = {p95_det + p95_rec:6.2f} ms | {pipelined_fps:5.1f} FPS")

    # 3. Gallery Search Scaling (Linear Scan Benchmark)
    print("\n[Stage 2: Gallery Search Scalability (128-d Cosine Metric)]")
    print(f"{'Enrolled People':<18} | {'Total References':<18} | {'Search Time (us)':<18} | {'Status'}")
    print("-" * 65)

    query = normalize(np.random.randn(128).tolist())
    for num_people in [2, 10, 25, 50, 100, 250, 500]:
        # Assume 100 references per person
        gallery = {
            f"Person_{i}": [normalize(np.random.randn(128).tolist()) for _ in range(100)]
            for i in range(num_people)
        }
        total_refs = num_people * 100

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            identify(query, gallery)
            times.append((time.perf_counter() - t0) * 1_000_000)

        mean_us = np.mean(times)
        status = "Optimal (<1ms)" if mean_us < 1000 else "Acceptable"
        print(f"{num_people:<18} | {total_refs:<18} | {mean_us:16.1f}   | {status}")

    print("=" * 65)
    print("  BENCHMARK COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark(iterations=30)
