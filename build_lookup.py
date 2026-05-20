#!/usr/bin/env python3
"""
Build a lookup table of car contours. Each PNG is processed in a fresh
subprocess so memory cannot accumulate. Slower (process startup overhead)
but works regardless of any allocator leaks in OpenCV/PIL/numpy.

Usage:
  python build_lookup.py --in frames/ --out car_positions.jsonl
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


WORKER = r'''
import sys, json
import cv2
import numpy as np

path = sys.argv[1]
threshold = int(sys.argv[2])
epsilon = float(sys.argv[3])

rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
    print(json.dumps({"error": "not_rgba"}))
    sys.exit(0)

h, w = rgba.shape[:2]
alpha = rgba[:, :, 3]
binary = (alpha > threshold).astype(np.uint8)
if binary.sum() == 0:
    print(json.dumps({"error": "empty", "height": h, "width": w}))
    sys.exit(0)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print(json.dumps({"error": "no_contour", "height": h, "width": w}))
    sys.exit(0)

biggest = max(contours, key=cv2.contourArea)
if epsilon > 0:
    biggest = cv2.approxPolyDP(biggest, epsilon, closed=True)
pts = biggest.reshape(-1, 2).astype(int).tolist()
print(json.dumps({"height": h, "width": w, "points": pts}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha-threshold", type=int, default=30)
    ap.add_argument("--epsilon", type=float, default=1.5)
    ap.add_argument("--pattern", default="*.png")
    args = ap.parse_args()

    files = sorted(Path(args.in_dir).glob(args.pattern))
    if not files:
        sys.exit(f"No PNGs in {args.in_dir}")
    print(f"Found {len(files)} files.")

    # Write the worker once to a temp file so we don't pay parse cost each call
    worker_path = Path(args.out).with_suffix(".worker.py")
    worker_path.write_text(WORKER)

    h_out = w_out = None
    n_kept = 0
    total_points = 0

    try:
        with open(args.out, "w") as out_f:
            header_written = False

            for i, fp in enumerate(files):
                try:
                    result = subprocess.run(
                        [sys.executable, str(worker_path), str(fp),
                         str(args.alpha_threshold), str(args.epsilon)],
                        capture_output=True, text=True, timeout=60
                    )
                except subprocess.TimeoutExpired:
                    print(f"  skip (timeout): {fp.name}")
                    continue

                if result.returncode != 0:
                    print(f"  skip (worker failed): {fp.name}")
                    if result.stderr:
                        print(f"    {result.stderr.strip()}")
                    continue

                try:
                    obj = json.loads(result.stdout.strip().split("\n")[-1])
                except Exception:
                    print(f"  skip (bad output): {fp.name}")
                    continue

                if "error" in obj:
                    print(f"  skip ({obj['error']}): {fp.name}")
                    continue

                if h_out is None:
                    h_out, w_out = obj["height"], obj["width"]
                    out_f.write(json.dumps({
                        "type": "header", "height": h_out, "width": w_out
                    }) + "\n")
                    header_written = True

                out_f.write(json.dumps({
                    "filename": fp.name,
                    "points": obj["points"],
                }) + "\n")
                out_f.flush()

                n_kept += 1
                total_points += len(obj["points"])

                if (i + 1) % 25 == 0:
                    avg = total_points / max(n_kept, 1)
                    print(f"  processed {i + 1}/{len(files)} (kept {n_kept}, "
                          f"avg {avg:.0f} pts/frame)")
    finally:
        worker_path.unlink(missing_ok=True)

    if n_kept == 0:
        sys.exit("No usable masks found.")

    size_kb = Path(args.out).stat().st_size / 1024
    print(f"\nSaved {n_kept} contours to {args.out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
