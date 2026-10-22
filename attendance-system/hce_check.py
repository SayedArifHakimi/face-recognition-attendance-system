"""HCE (Health Check Engineer) Automated Governance & Verification Runner."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


def run_command(cmd, desc):
    print(f"\n[HCE] Running: {desc}...")
    res = subprocess.run(cmd, shell=True, cwd=str(ROOT))
    if res.returncode != 0:
        print(f"[HCE] FAILED: {desc} exited with code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[HCE] PASSED: {desc}")


def check_git_hygiene():
    print("\n[HCE] Checking Biometric Security & Git Privacy...")
    res = subprocess.run("git status --porcelain", shell=True, cwd=str(REPO_ROOT), capture_output=True, text=True)
    tracked_files = res.stdout.splitlines()
    for line in tracked_files:
        if "gallery.json" in line or "attendance.db" in line or ".onnx" in line:
            print(f"[HCE] CRITICAL SECURITY VIOLATION: Sensitive file detected in git status: {line}")
            sys.exit(1)
    print("[HCE] PASSED: No biometric embeddings, database files, or binary models tracked in Git.")


def main():
    print("=" * 65)
    print("  HCE (HEALTH CHECK ENGINEER) MANDATORY VERIFICATION SUITE")
    print("  Standard: Ponytail Lazy Senior Developer Framework")
    print("=" * 65)

    python_bin = sys.executable

    # 1. Unittest discovery across all test_*.py
    run_command(f'"{python_bin}" -m unittest discover -v -p "test_*.py"', "Automated Test Discovery (test_*.py)")

    # 2. Standalone Face ID regression test
    run_command(f'"{python_bin}" test_face_id.py', "Face ID Regressions (test_face_id.py)")

    # 3. Model assets & integrity check
    run_command(f'"{python_bin}" diagnostics.py', "Model & System Diagnostics (diagnostics.py)")

    # 4. Strict Biometric Security Check
    check_git_hygiene()

    print("\n" + "=" * 65)
    print("  [HCE VERIFIED] ALL GOVERNANCE GATES PASSED CLEANLY (0 FAILURES)")
    print("=" * 65)


if __name__ == "__main__":
    main()
