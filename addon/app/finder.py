"""Parking-spot detection — reusable form of the find_parking.py CLI.

A ParkingFinder loads the YOLO model and the contour lookup table once, then
analyse() can be called repeatedly on fresh camera frames.
"""
import json
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


def _load_lookup(path):
    """Read the JSONL lookup. First line is a header with height/width."""
    contours, filenames = [], []
    h = w = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "header":
                h, w = obj["height"], obj["width"]
            else:
                contours.append(np.array(obj["points"], dtype=np.int32))
                filenames.append(obj["filename"])
    if h is None:
        raise ValueError("No header found in lookup file")
    return contours, filenames, h, w


@dataclass
class ParkingResult:
    space_available: bool
    free_spots: int
    cars_detected: int
    chosen_spot: str | None
    own_vehicle_home: bool
    own_vehicle_spot: str | None
    own_vehicle_iou: float
    image_jpeg: bytes


class ParkingFinder:
    def __init__(self, lookup_path, model_path="yolov8s-seg.pt",
                 conf=0.25, classes=(2, 5, 7), imgsz=1280, dilate=10,
                 own_vehicle_iou=0.85):
        self.conf = conf
        self.classes = list(classes)
        self.imgsz = imgsz
        self.dilate = dilate
        self.own_vehicle_iou = own_vehicle_iou
        self.contours, self.filenames, self.lh, self.lw = _load_lookup(lookup_path)
        self.model = YOLO(model_path)

    def _detect_obstacles(self, image_bgr):
        th, tw = self.lh, self.lw
        r = self.model.predict(source=image_bgr, imgsz=self.imgsz, conf=self.conf,
                               classes=self.classes, verbose=False)[0]
        obstacles = np.zeros((th, tw), dtype=np.uint8)
        vehicle_masks = []
        boxes = []
        if r.masks is not None and len(r.masks.data) > 0:
            for m in r.masks.data.cpu().numpy():
                m_resized = cv2.resize(m, (tw, th), interpolation=cv2.INTER_LINEAR)
                mask = (m_resized > 0.5).astype(np.uint8) * 255
                vehicle_masks.append(mask)
                obstacles |= mask
        if r.boxes is not None:
            for box, c, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                   r.boxes.conf.cpu().numpy(),
                                   r.boxes.cls.cpu().numpy()):
                boxes.append((box.tolist(), float(c), int(cls)))
        if self.dilate > 0 and obstacles.any():
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.dilate * 2 + 1, self.dilate * 2 + 1))
            obstacles = cv2.dilate(obstacles, k)
        return obstacles, boxes, vehicle_masks

    def _best_own_match(self, vehicle_masks):
        """Find the candidate contour that best matches a detected vehicle.

        Returns (contour_index, IoU). A high IoU means a detected car aligns
        almost exactly with a known position — i.e. it is our own vehicle
        parked in one of its usual spots.
        """
        best_idx, best_iou = None, 0.0
        for vmask in vehicle_masks:
            v_area = int(np.count_nonzero(vmask))
            if v_area == 0:
                continue
            vys, vxs = np.where(vmask > 0)
            vx0, vy0, vx1, vy1 = vxs.min(), vys.min(), vxs.max(), vys.max()
            for i, c in enumerate(self.contours):
                x_min, y_min = c.min(axis=0)
                x_max, y_max = c.max(axis=0)
                # cheap reject: contour and vehicle bounding boxes disjoint
                if x_max < vx0 or x_min > vx1 or y_max < vy0 or y_min > vy1:
                    continue
                x0, y0 = max(0, int(x_min)), max(0, int(y_min))
                x1 = min(self.lw, int(x_max) + 1)
                y1 = min(self.lh, int(y_max) + 1)
                if x1 <= x0 or y1 <= y0:
                    continue
                local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                shifted = c - np.array([x0, y0], dtype=np.int32)
                cv2.fillPoly(local,
                             [shifted.reshape(-1, 1, 2).astype(np.int32)], 255)
                c_area = int(np.count_nonzero(local))
                if c_area == 0:
                    continue
                inter = int(np.count_nonzero(local & vmask[y0:y1, x0:x1]))
                union = v_area + c_area - inter
                iou = inter / union if union else 0.0
                if iou > best_iou:
                    best_iou, best_idx = iou, i
        return best_idx, best_iou

    def analyse(self, image_bgr) -> ParkingResult:
        lh, lw = self.lh, self.lw
        if image_bgr.shape[:2] != (lh, lw):
            image_bgr = cv2.resize(image_bgr, (lw, lh),
                                   interpolation=cv2.INTER_AREA)

        obstacles, boxes, vehicle_masks = self._detect_obstacles(image_bgr)

        cx_img, cy_img = lw / 2.0, lh / 2.0
        fitting = []
        for i, c in enumerate(self.contours):
            x_min, y_min = c.min(axis=0)
            x_max, y_max = c.max(axis=0)
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(lw, x_max + 1), min(lh, y_max + 1)
            bw, bh = x_max - x_min, y_max - y_min
            if bw <= 0 or bh <= 0:
                continue
            local_mask = np.zeros((bh, bw), dtype=np.uint8)
            shifted = c - np.array([x_min, y_min], dtype=np.int32)
            cv2.fillPoly(local_mask,
                         [shifted.reshape(-1, 1, 2).astype(np.int32)], 255)
            obstacle_patch = obstacles[y_min:y_max, x_min:x_max]
            if np.any(local_mask & obstacle_patch):
                continue
            ys, xs = np.where(local_mask > 0)
            if len(xs) == 0:
                continue
            cx = xs.mean() + x_min
            cy = ys.mean() + y_min
            dist2 = (cx - cx_img) ** 2 + (cy - cy_img) ** 2
            fitting.append((i, dist2))

        chosen_idx = None
        if fitting:
            chosen_idx, _ = min(fitting, key=lambda t: t[1])

        own_idx, own_iou = self._best_own_match(vehicle_masks)
        own_home = own_idx is not None and own_iou >= self.own_vehicle_iou
        own_match_idx = own_idx if own_home else None

        viz = self._visualize(image_bgr, obstacles, boxes, fitting, chosen_idx,
                              own_match_idx)
        ok, buf = cv2.imencode(".jpg", viz, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Failed to encode result image")

        return ParkingResult(
            space_available=chosen_idx is not None,
            free_spots=len(fitting),
            cars_detected=len(boxes),
            chosen_spot=self.filenames[chosen_idx] if chosen_idx is not None else None,
            own_vehicle_home=own_home,
            own_vehicle_spot=self.filenames[own_match_idx] if own_home else None,
            own_vehicle_iou=round(own_iou, 3),
            image_jpeg=buf.tobytes(),
        )

    def _visualize(self, img, obstacles, boxes, fitting, chosen_idx,
                   own_match_idx=None):
        lh, lw = self.lh, self.lw
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
        if fitting:
            all_fits = np.zeros((lh, lw), dtype=np.uint8)
            for idx, _ in fitting:
                cv2.fillPoly(
                    all_fits,
                    [self.contours[idx].reshape(-1, 1, 2).astype(np.int32)], 255)
            overlay = viz.copy()
            overlay[all_fits > 0] = (0, 255, 0)
            viz = cv2.addWeighted(overlay, 0.20, viz, 0.80, 0)
        if chosen_idx is not None:
            cv2.drawContours(
                viz,
                [self.contours[chosen_idx].reshape(-1, 1, 2).astype(np.int32)],
                -1, (0, 255, 0), 4)
            cv2.drawMarker(viz, (int(lw / 2), int(lh / 2)), (255, 255, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=40, thickness=3)
        if own_match_idx is not None:
            c = self.contours[own_match_idx].reshape(-1, 1, 2).astype(np.int32)
            cv2.drawContours(viz, [c], -1, (255, 128, 0), 4)
            mx, my = c.reshape(-1, 2).mean(axis=0).astype(int)
            cv2.putText(viz, "MY CAR", (mx - 60, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(viz, "MY CAR", (mx - 60, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 128, 0), 2)
        label = (f"BEST SPOT: {self.filenames[chosen_idx]}  ({len(fitting)} options)"
                 if chosen_idx is not None else "NO PARKING AVAILABLE")
        cv2.putText(viz, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 0), 6)
        cv2.putText(viz, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), 2)
        return viz
