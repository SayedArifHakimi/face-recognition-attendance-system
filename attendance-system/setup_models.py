"""Download OpenCV's published models and retain their licenses and hashes."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent / "models"
FILES = {
    "yunet.onnx": "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface.onnx": "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def main():
    ROOT.mkdir(exist_ok=True)
    with urlopen("https://api.github.com/repos/opencv/opencv_zoo/commits/main", timeout=30) as response:
        commit = json.load(response)["sha"]
    manifest = {"repository": "opencv/opencv_zoo", "commit": commit, "files": {}}
    for name, relative in FILES.items():
        for target, url in [
            (name, f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{commit}/models/{relative}"),
            (name + ".LICENSE", f"https://raw.githubusercontent.com/opencv/opencv_zoo/{commit}/models/{relative.split('/')[0]}/LICENSE"),
        ]:
            path = ROOT / target
            if path.exists():
                raise FileExistsError(f"{path} exists; preserve it or move the model folder before rerunning setup.")
            with urlopen(url, timeout=120) as response:
                payload = response.read()
            if target.endswith(".onnx") and len(payload) < 10000:
                raise ValueError(f"Invalid model response: {target}")
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(path)
            manifest["files"][target] = {"url": url, "sha256": hashlib.sha256(payload).hexdigest()}
            print(f"Downloaded {target}")
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
