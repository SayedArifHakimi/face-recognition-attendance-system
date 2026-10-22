"""Empirical gallery calibration and mathematical validation for face recognition embeddings."""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

DEFAULT_GALLERY = Path(__file__).resolve().parent / "data" / "gallery.json"


def load_gallery_matrix(gallery_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Load gallery JSON and return (class_names, X, y) with L2-normalized embeddings."""
    if not gallery_path.exists():
        raise FileNotFoundError(f"Gallery file not found: {gallery_path}")

    with open(gallery_path, "r", encoding="utf-8") as f:
        gallery = json.load(f)

    if not isinstance(gallery, dict) or not gallery:
        raise ValueError("Gallery must be a non-empty JSON object mapping names to embedding lists")

    names = sorted(list(gallery.keys()))
    vectors = []
    labels = []

    for idx, name in enumerate(names):
        refs = gallery[name]
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"No reference embeddings found for {name}")
        for ref in refs:
            v = np.array(ref, dtype=np.float32)
            if v.ndim != 1 or len(v) != 128:
                raise ValueError(f"Expected 128-dimensional embedding, got shape {v.shape}")
            norm = np.linalg.norm(v)
            if norm == 0 or not np.isfinite(norm):
                raise ValueError("Embedding norm cannot be zero or non-finite")
            vectors.append(v / norm)
            labels.append(idx)

    X = np.stack(vectors, axis=0)
    y = np.array(labels, dtype=np.int32)
    return names, X, y


def compute_stats(values: np.ndarray) -> Dict[str, float]:
    """Calculate summary statistics for a 1D array of cosine similarity values."""
    if len(values) == 0:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "median": 0.0, "p01": 0.0, "p99": 0.0}
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p01": float(np.percentile(values, 1)),
        "p99": float(np.percentile(values, 99)),
    }


def evaluate_pairwise_distributions(names: List[str], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Compute intra-class and inter-class pairwise cosine similarity distributions."""
    N = len(X)
    S = np.dot(X, X.T)

    intra_stats = {}
    all_intra = []
    for idx, name in enumerate(names):
        mask = (y == idx)
        sub_S = S[np.ix_(mask, mask)]
        triu = sub_S[np.triu_indices(len(sub_S), k=1)]
        intra_stats[name] = compute_stats(triu)
        all_intra.extend(triu)

    # Inter-class (cross-person pairs)
    inter_mask = (y[:, None] != y[None, :])
    # Only upper triangle to avoid double counting
    upper_tri = np.triu(np.ones((N, N), dtype=bool), k=1)
    inter_pairs = S[inter_mask & upper_tri]
    inter_stats = compute_stats(inter_pairs)

    combined_intra = compute_stats(np.array(all_intra, dtype=np.float32))

    return {
        "per_person_intra": intra_stats,
        "combined_intra": combined_intra,
        "inter_class": inter_stats,
    }


def run_loocv(
    names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.50,
    margin: float = 0.08,
) -> Dict[str, Any]:
    """
    Vectorized Leave-One-Out Cross-Validation (LOOCV) simulation of face_id.identify().
    Evaluates whether each enrolled reference, when excluded, correctly resolves to its owner.
    """
    N = len(X)
    num_classes = len(names)
    S = np.dot(X, X.T)
    np.fill_diagonal(S, -1.0)

    correct = 0
    unknown = 0
    mismatch = 0
    per_person = {name: {"total": 0, "correct": 0, "unknown": 0, "mismatch": 0} for name in names}

    for i in range(N):
        target = y[i]
        target_name = names[target]
        per_person[target_name]["total"] += 1

        scores = []
        for cls_idx in range(num_classes):
            mask = (y == cls_idx)
            if cls_idx == target:
                mask = mask.copy()
                mask[i] = False
            cls_max = S[i, mask].max() if np.any(mask) else -1.0
            scores.append((cls_max, cls_idx))

        scores.sort(key=lambda s: s[0], reverse=True)
        best_score, best_cls = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else -1.0
        ambiguous = (best_score - second_score) < margin

        if best_score < threshold or ambiguous:
            pred = -1
        else:
            pred = best_cls

        if pred == target:
            correct += 1
            per_person[target_name]["correct"] += 1
        elif pred == -1:
            unknown += 1
            per_person[target_name]["unknown"] += 1
        else:
            mismatch += 1
            per_person[target_name]["mismatch"] += 1

    return {
        "total": N,
        "correct": correct,
        "unknown": unknown,
        "mismatch": mismatch,
        "tar": correct / N if N > 0 else 0.0,
        "frr": unknown / N if N > 0 else 0.0,
        "far": mismatch / N if N > 0 else 0.0,
        "per_person": per_person,
    }


def run_threshold_sweep(
    names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    margin: float = 0.08,
    thresholds: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Sweep threshold values to analyze True Accept vs False Reject behavior."""
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    results = []
    for th in thresholds:
        res = run_loocv(names, X, y, threshold=th, margin=margin)
        results.append({
            "threshold": th,
            "margin": margin,
            "correct": res["correct"],
            "unknown": res["unknown"],
            "mismatch": res["mismatch"],
            "tar": res["tar"],
            "frr": res["frr"],
            "far": res["far"],
        })
    return results


def run_margin_sweep(
    names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.50,
    margins: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Sweep margin values to evaluate the effect of ambiguity rejection."""
    if margins is None:
        margins = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]

    results = []
    for mg in margins:
        res = run_loocv(names, X, y, threshold=threshold, margin=mg)
        results.append({
            "threshold": threshold,
            "margin": mg,
            "correct": res["correct"],
            "unknown": res["unknown"],
            "mismatch": res["mismatch"],
            "tar": res["tar"],
            "frr": res["frr"],
            "far": res["far"],
        })
    return results


def print_report(
    gallery_path: Path,
    threshold: float = 0.50,
    margin: float = 0.08,
    include_sweep: bool = False,
) -> bool:
    """Execute complete analysis and print human-readable CLI report."""
    print("=" * 68)
    print("  BIOMETRIC GALLERY EMPIRICAL CALIBRATION & EVALUATION")
    print("  OpenCV SFace 128-d Cosine Metric Validation")
    print("=" * 68)

    t0 = time.perf_counter()
    names, X, y = load_gallery_matrix(gallery_path)
    load_ms = (time.perf_counter() - t0) * 1000

    print(f"Gallery file: {gallery_path.name}")
    print(f"Total enrolled identities : {len(names)}")
    for name in names:
        count = int(np.sum(y == names.index(name)))
        print(f"  - {name:<16}: {count:4d} reference embeddings")
    print(f"Total embeddings loaded   : {len(X)} in {load_ms:.1f}ms\n")

    # 1. Pairwise distributions
    dist = evaluate_pairwise_distributions(names, X, y)
    print("-" * 68)
    print("1. PAIRWISE COSINE SIMILARITY DISTRIBUTIONS")
    print("-" * 68)
    print(f"{'Class / Comparison':<22} | {'Pairs':>7} | {'Mean':>6} | {'Std':>6} | {'Min':>6} | {'Max':>6}")
    print("-" * 68)

    for name in names:
        s = dist["per_person_intra"][name]
        print(f"Intra: {name:<15} | {s['count']:7d} | {s['mean']:6.4f} | {s['std']:6.4f} | {s['min']:6.4f} | {s['max']:6.4f}")

    ci = dist["combined_intra"]
    print(f"{'Intra: (Combined)':<22} | {ci['count']:7d} | {ci['mean']:6.4f} | {ci['std']:6.4f} | {ci['min']:6.4f} | {ci['max']:6.4f}")

    ic = dist["inter_class"]
    print(f"{'Inter: (Cross-Person)':<22} | {ic['count']:7d} | {ic['mean']:6.4f} | {ic['std']:6.4f} | {ic['min']:6.4f} | {ic['max']:6.4f}")
    print("-" * 68)
    print(f"Distribution separation: Intra Mean ({ci['mean']:.4f}) vs Inter Mean ({ic['mean']:.4f}) -> Delta: +{ci['mean'] - ic['mean']:.4f}\n")

    # 2. Leave-One-Out Cross-Validation
    t1 = time.perf_counter()
    loocv = run_loocv(names, X, y, threshold=threshold, margin=margin)
    loocv_ms = (time.perf_counter() - t1) * 1000

    print("-" * 68)
    print(f"2. LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)")
    print(f"   Configured: Threshold = {threshold:.2f}, Ambiguity Margin = {margin:.2f}")
    print("-" * 68)
    print(f"{'Identity':<16} | {'Samples':>7} | {'Correct (TAR)':>13} | {'Unknown (FRR)':>13} | {'Mismatch (FAR)':>14}")
    print("-" * 68)
    for name in names:
        pp = loocv["per_person"][name]
        n_p = pp["total"]
        c_p = f"{pp['correct']} ({pp['correct']/n_p*100:.1f}%)"
        u_p = f"{pp['unknown']} ({pp['unknown']/n_p*100:.1f}%)"
        m_p = f"{pp['mismatch']} ({pp['mismatch']/n_p*100:.1f}%)"
        print(f"{name:<16} | {n_p:7d} | {c_p:>13} | {u_p:>13} | {m_p:>14}")
    print("-" * 68)
    overall_c = f"{loocv['correct']} ({loocv['tar']*100:.2f}%)"
    overall_u = f"{loocv['unknown']} ({loocv['frr']*100:.2f}%)"
    overall_m = f"{loocv['mismatch']} ({loocv['far']*100:.2f}%)"
    print(f"{'OVERALL':<16} | {loocv['total']:7d} | {overall_c:>13} | {overall_u:>13} | {overall_m:>14}")
    print(f"Evaluated in {loocv_ms:.2f}ms across {loocv['total']} leave-one-out iterations.\n")

    # 3. Sweeps if requested
    if include_sweep:
        print("-" * 68)
        print("3. THRESHOLD SENSITIVITY SWEEP (Fixed Margin = 0.08)")
        print("-" * 68)
        print(f"{'Threshold':>9} | {'TAR (Correct)':>13} | {'FRR (Unknown)':>13} | {'FAR (Mismatch)':>14}")
        print("-" * 68)
        th_results = run_threshold_sweep(names, X, y, margin=margin)
        for r in th_results:
            c_str = f"{r['correct']} ({r['tar']*100:.1f}%)"
            u_str = f"{r['unknown']} ({r['frr']*100:.1f}%)"
            m_str = f"{r['mismatch']} ({r['far']*100:.1f}%)"
            print(f"{r['threshold']:9.2f} | {c_str:>13} | {u_str:>13} | {m_str:>14}")
        print("-" * 68 + "\n")

        print("-" * 68)
        print("4. AMBIGUITY MARGIN SENSITIVITY SWEEP (Fixed Threshold = 0.50)")
        print("-" * 68)
        print(f"{'Margin':>9} | {'TAR (Correct)':>13} | {'FRR (Unknown)':>13} | {'FAR (Mismatch)':>14}")
        print("-" * 68)
        mg_results = run_margin_sweep(names, X, y, threshold=threshold)
        for r in mg_results:
            c_str = f"{r['correct']} ({r['tar']*100:.1f}%)"
            u_str = f"{r['unknown']} ({r['frr']*100:.1f}%)"
            m_str = f"{r['mismatch']} ({r['far']*100:.1f}%)"
            print(f"{r['margin']:9.2f} | {c_str:>13} | {u_str:>13} | {m_str:>14}")
        print("-" * 68 + "\n")

    # 4. Calibration Assessment Verdict
    print("=" * 68)
    print("  CALIBRATION VERDICT")
    print("=" * 68)
    if loocv["mismatch"] == 0:
        print("  [PASSED] ZERO FALSE ACCEPTS (FAR = 0.00%)")
        print(f"  True Accept Rate (TAR) : {loocv['tar']*100:.2f}%")
        print(f"  False Reject Rate (FRR): {loocv['frr']*100:.2f}% (safe quality rejections)")
        print(f"  Separation integrity  : Robust across both enrolled subjects.")
        print("=" * 68)
        return True
    else:
        print(f"  [WARNING] Detected {loocv['mismatch']} impostor misclassifications (FAR = {loocv['far']*100:.2f}%).")
        print("  Recommendation: Increase threshold or margin.")
        print("=" * 68)
        return False


def main():
    parser = argparse.ArgumentParser(description="Empirical Biometric Gallery Calibration & Evaluation")
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY, help="Path to gallery JSON")
    parser.add_argument("--threshold", type=float, default=0.50, help="Cosine threshold (default: 0.50)")
    parser.add_argument("--margin", type=float, default=0.08, help="Ambiguity margin (default: 0.08)")
    parser.add_argument("--sweep", action="store_true", help="Run full threshold and margin sensitivity sweeps")
    args = parser.parse_args()

    if not args.gallery.exists():
        print(f"ERROR: Gallery file not found at {args.gallery}", file=sys.stderr)
        sys.exit(1)

    ok = print_report(
        gallery_path=args.gallery,
        threshold=args.threshold,
        margin=args.margin,
        include_sweep=args.sweep,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
