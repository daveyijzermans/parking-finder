#!/usr/bin/env python3
"""
Find a parking spot for my car in a live camera image.

Inputs:
  --lookup : JSONL of car-position contours built by build_lookup.py
  --image  : current frame from the doorbell camera

Steps:
  1. YOLO detects other vehicles in the live image.
  2. Their masks are combined into one "obstacles" image.
  3. For each candidate position, rasterize its contour and check overlap.
  4. Among fitting positions, pick the one closest to the image center.
  5. Save a visualization.

Usage:
  python find_parking.py --lookup car_positions.jsonl --image live.jpg --out result.jpg

  --model  : YOLO model (default yolov8s-seg.pt)
  --conf   : detection confidence threshold (default 0.25)
  --classes: COCO classes to treat as obstacles (default 2 5 7 = car,bus,truck)
  --imgsz  : YOLO inference image size (default 1280)
  --dilate : safety margin in pixels around obstacles (default 10)
"""
import argparse
import json
import sys

import cv2
import numpy as np
from ultralytics import YOLO


def load_lookup(path):
    """Read JSONL lookup. First line is header with height/width, rest are contours."""
    contours = []
    filenames = []
    h = w = None
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") == "header":
                h, w = obj["height"], obj["width"]
            else:
                contours.append(np.array(obj["points"], dtype=np.int32))
                filenames.append(obj["filename"])
    if h is None:
        raise ValueError("No header found in lookup file")
    return contours, filenames, h, w


def rasterize(contour, h, w):
    """Return a uint8 mask (0/255) for one contour."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour.reshape(-1, 1, 2).astype(np.int32)], 255)
    return mask


def detect_obstacles(model, image_bgr, conf, classes, imgsz, target_hw, dilate_px):
    th, tw = target_hw
    r = model.predict(source=image_bgr, imgsz=imgsz, conf=conf,
                      classes=classes, verbose=False)[0]
    obstacles = np.zeros((th, tw), dtype=np.uint8)
    boxes = []
    if r.masks is not None and len(r.masks.data) > 0:
        for m in r.masks.data.cpu().numpy():
            m_resized = cv2.resize(m, (tw, th), interpolation=cv2.INTER_LINEAR)
            obstacles |= ((m_resized > 0.5).astype(np.uint8) * 255)
    if r.boxes is not None:
        for box, c, cls in zip(r.boxes.xyxy.cpu().numpy(),
                               r.boxes.conf.cpu().numpy(),
                               r.boxes.cls.cpu().numpy()):
            boxes.append((box.tolist(), float(c), int(cls)))
    if dilate_px > 0 and obstacles.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (dilate_px * 2 + 1, dilate_px * 2 + 1))
        obstacles = cv2.dilate(obstacles, k)
    return obstacles, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="parking_result.jpg")
    ap.add_argument("--model", default="yolov8s-seg.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", type=int, nargs="+", default=[2, 5, 7])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--dilate", type=int, default=10)
    args = ap.parse_args()

    print(f"Loading lookup: {args.lookup}")
    contours, filenames, lh, lw = load_lookup(args.lookup)
    print(f"  {len(contours)} candidate positions at {lw}x{lh}")

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Could not read {args.image}")
    if img.shape[:2] != (lh, lw):
        print(f"  resizing image from {img.shape[1]}x{img.shape[0]} to {lw}x{lh}")
        img = cv2.resize(img, (lw, lh), interpolation=cv2.INTER_AREA)

    print(f"Loading YOLO: {args.model}")
    model = YOLO(args.model)

    print("Detecting other vehicles...")
    obstacles, boxes = detect_obstacles(model, img, args.conf, args.classes,
                                        args.imgsz, (lh, lw), args.dilate)
    print(f"  {len(boxes)} vehicle(s) detected")

    # For each candidate position, rasterize and check overlap.
    # Optimization: compute the bounding box of the contour, AND only that
    # region of obstacles against the rasterized contour — much faster than
    # full-frame AND for small cars.
    print("Checking which positions fit...")
    cx_img, cy_img = lw / 2.0, lh / 2.0
    fitting = []  # list of (idx, centroid_distance_squared)
    for i, c in enumerate(contours):
        x_min, y_min = c.min(axis=0)
        x_max, y_max = c.max(axis=0)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(lw, x_max + 1), min(lh, y_max + 1)

        # Rasterize within the bounding box only
        bw, bh = x_max - x_min, y_max - y_min
        if bw <= 0 or bh <= 0:
            continue
        local_mask = np.zeros((bh, bw), dtype=np.uint8)
        shifted = c - np.array([x_min, y_min], dtype=np.int32)
        cv2.fillPoly(local_mask, [shifted.reshape(-1, 1, 2).astype(np.int32)], 255)

        obstacle_patch = obstacles[y_min:y_max, x_min:x_max]
        if np.any(local_mask & obstacle_patch):
            continue  # overlap — doesn't fit

        # Compute centroid (use the rasterized region for a true mask centroid)
        ys, xs = np.where(local_mask > 0)
        if len(xs) == 0:
            continue
        cx = xs.mean() + x_min
        cy = ys.mean() + y_min
        dist2 = (cx - cx_img) ** 2 + (cy - cy_img) ** 2
        fitting.append((i, dist2))

    print(f"  {len(fitting)} positions fit")

    chosen_idx = None
    if fitting:
        chosen_idx, _ = min(fitting, key=lambda t: t[1])
        print(f"  best: {filenames[chosen_idx]}")

    # Build visualization
    viz = img.copy()
    if obstacles.any():
        overlay = viz.copy()
        overlay[obstacles > 0] = (0, 0, 255)
        viz = cv2.addWeighted(overlay, 0.35, viz, 0.65, 0)
    for (x1, y1, x2, y2), c, cls in boxes:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(viz, f"cls{cls} {c:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Faint green for all fitting spots
    if fitting:
        all_fits = np.zeros((lh, lw), dtype=np.uint8)
        for idx, _ in fitting:
            cv2.fillPoly(all_fits,
                         [contours[idx].reshape(-1, 1, 2).astype(np.int32)],
                         255)
        overlay = viz.copy()
        overlay[all_fits > 0] = (0, 255, 0)
        viz = cv2.addWeighted(overlay, 0.20, viz, 0.80, 0)

    # Bright green outline for the chosen spot
    if chosen_idx is not None:
        cv2.drawContours(viz,
                         [contours[chosen_idx].reshape(-1, 1, 2).astype(np.int32)],
                         -1, (0, 255, 0), 4)
        cv2.drawMarker(viz, (int(lw / 2), int(lh / 2)), (255, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=40, thickness=3)

    label = (f"BEST SPOT: {filenames[chosen_idx]}  ({len(fitting)} options)"
             if chosen_idx is not None else "NO PARKING AVAILABLE")
    cv2.putText(viz, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
    cv2.putText(viz, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    cv2.imwrite(args.out, viz, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
