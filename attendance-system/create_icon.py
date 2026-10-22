"""Generate website favicon icons (SVG and ICO)."""
import struct
from pathlib import Path
import cv2
import numpy as np

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

SVG_PATH = STATIC_DIR / "favicon.svg"
ICO_PATH = STATIC_DIR / "favicon.ico"

SVG_CONTENT = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <rect width="32" height="32" rx="7" fill="#0f172a"/>
  <!-- Corner Scan Reticles -->
  <path d="M6 11V7a1 1 0 0 1 1-1h4M21 6h4a1 1 0 0 1 1 1v4M26 21v4a1 1 0 0 1-1 1h-4M11 26H7a1 1 0 0 1-1-1v-4" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <!-- Stylized Biometric Face Contour -->
  <circle cx="16" cy="12.5" r="4.2" fill="#38bdf8"/>
  <path d="M9.5 23.5c0-3.3 2.9-5.5 6.5-5.5s6.5 2.2 6.5 5.5" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" fill="none"/>
  <!-- Active Green Verified Dot -->
  <circle cx="22" cy="9.5" r="1.8" fill="#10b981"/>
</svg>
"""


def generate_icons():
    SVG_PATH.write_text(SVG_CONTENT.strip(), encoding="utf-8")

    # Render 32x32 RGBA icon
    img = np.zeros((32, 32, 4), dtype=np.uint8)
    # Slate/navy rounded rectangle background (#0f172a -> BGR 42, 23, 15)
    cv2.rectangle(img, (2, 2), (29, 29), (42, 23, 15, 255), -1)

    cyan = (248, 189, 56, 255)  # #38bdf8 in BGRA
    # Corner brackets
    cv2.line(img, (6, 11), (6, 7), cyan, 2)
    cv2.line(img, (6, 7), (10, 7), cyan, 2)
    cv2.line(img, (25, 11), (25, 7), cyan, 2)
    cv2.line(img, (25, 7), (21, 7), cyan, 2)
    cv2.line(img, (6, 20), (6, 24), cyan, 2)
    cv2.line(img, (6, 24), (10, 24), cyan, 2)
    cv2.line(img, (25, 20), (25, 24), cyan, 2)
    cv2.line(img, (25, 24), (21, 24), cyan, 2)

    # Face silhouette
    cv2.circle(img, (16, 13), 4, cyan, -1)
    cv2.ellipse(img, (16, 23), (6, 4), 0, 180, 360, cyan, 2)

    # Emerald verified dot (#10b981 -> BGR 129, 185, 16)
    cv2.circle(img, (22, 9), 2, (129, 185, 16, 255), -1)

    ok, png_data = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode PNG for ICO")
    png_bytes = png_data.tobytes()

    # Pack into standard ICO container
    header = struct.pack("<HHH", 0, 1, 1)
    entry = struct.pack("<BBBBHHII", 32, 32, 0, 0, 1, 32, len(png_bytes), 22)
    ico_bytes = header + entry + png_bytes
    ICO_PATH.write_bytes(ico_bytes)

    print(f"Generated {SVG_PATH.name} ({len(SVG_CONTENT)} bytes)")
    print(f"Generated {ICO_PATH.name} ({len(ico_bytes)} bytes)")


if __name__ == "__main__":
    generate_icons()
