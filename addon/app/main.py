"""Add-on entry point: poll the camera, run detection, publish to MQTT."""
import json
import sys
import time

import cv2
import numpy as np

from app import supervisor
from app.finder import ParkingFinder
from app.mqtt_publisher import Publisher

OPTIONS_PATH = "/data/options.json"

# COCO class IDs for the vehicle types selectable in the add-on config
COCO_CLASS_IDS = {
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}


def log(msg):
    print(f"[parking-finder] {msg}", flush=True)


def load_options():
    with open(OPTIONS_PATH) as f:
        return json.load(f)


def decode_image(raw_bytes):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode camera snapshot")
    return img


def run_once(opts, finder, publisher):
    raw = supervisor.fetch_camera_snapshot(opts["camera_entity"])
    img = decode_image(raw)
    result = finder.analyse(img)
    publisher.publish_result(result)
    log(f"cars={result.cars_detected} free_spots={result.free_spots} "
        f"chosen={result.chosen_spot}")


def main():
    opts = load_options()
    log(f"Starting. camera={opts['camera_entity']} "
        f"interval={opts['scan_interval']}s")

    class_ids = [COCO_CLASS_IDS[c] for c in opts["classes"]]
    try:
        finder = ParkingFinder(
            lookup_path=opts["lookup_file"],
            conf=opts["conf"],
            classes=class_ids,
            imgsz=opts["imgsz"],
            dilate=opts["dilate"],
        )
    except FileNotFoundError:
        sys.exit(f"Lookup file not found: {opts['lookup_file']} — "
                 f"generate it with build_lookup.py and place it there.")
    log(f"Loaded {len(finder.contours)} candidate spots "
        f"at {finder.lw}x{finder.lh}")

    mqtt_cfg = supervisor.get_mqtt_config()
    log(f"Connecting to MQTT {mqtt_cfg['host']}:{mqtt_cfg['port']}")
    publisher = Publisher(**mqtt_cfg)
    publisher.publish_discovery()

    try:
        while True:
            try:
                run_once(opts, finder, publisher)
            except Exception as e:  # one bad frame must not kill the loop
                log(f"cycle failed: {e}")
            time.sleep(opts["scan_interval"])
    finally:
        publisher.close()


if __name__ == "__main__":
    main()
