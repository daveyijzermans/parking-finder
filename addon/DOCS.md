# Parking Finder

Detects open parking spots in a camera image and exposes the result to Home
Assistant as three MQTT-discovered entities:

- **binary_sensor** `Parking space available` — on when at least one spot fits.
  Attributes: `chosen_spot`, `free_spots`, `cars_detected`.
- **sensor** `Cars detected` — number of vehicles seen in the frame.
- **camera** `Parking debug image` — the visualization (red = obstacles,
  green = free spots, bright outline = chosen spot).

## Requirements

- The MQTT integration / a broker (e.g. the Mosquitto add-on) must be set up.
- A lookup file built with `build_lookup.py` from the parking-finder project.

## Setup

1. Build `car_positions.jsonl` with `build_lookup.py` (see the project README).
2. Copy it into the add-on's mapped `share` folder, e.g.
   `/share/parking_finder/car_positions.jsonl`.
3. Configure the options below and start the add-on.

## Options

| Option | Description |
|---|---|
| `camera_entity` | Camera entity to analyse, e.g. `camera.doorbell`. |
| `scan_interval` | Seconds between detections (minimum 30). |
| `lookup_file` | Path to the contour lookup file inside the container. |
| `conf` | YOLO detection confidence threshold (0–1). |
| `dilate` | Safety margin in pixels around detected vehicles. |
| `imgsz` | YOLO inference image size. |
| `classes` | COCO classes treated as obstacles (2=car, 5=bus, 7=truck). |

The camera snapshot and MQTT broker credentials are obtained automatically
from the Supervisor — no extra configuration needed.
