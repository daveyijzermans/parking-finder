# parking-finder

Find an open parking spot for a car in a live camera image.

The idea: collect a set of frames showing the car parked in every position it
has historically occupied, turn each into a contour ("a spot the car fits in"),
then check a live camera image to see which of those spots is currently free of
other vehicles.

## How it works

1. **`build_lookup.py`** — takes a folder of RGBA PNGs (each one a cut-out of the
   car against a transparent background) and extracts the largest alpha contour
   from each. The contours are written to a JSONL lookup file. Every PNG is
   processed in a fresh subprocess so memory cannot accumulate across frames.

2. **`find_parking.py`** — loads the lookup file and a live camera image:
   - YOLO detects other vehicles (car/bus/truck) and builds an "obstacles" mask.
   - Each candidate contour is rasterized and tested for overlap with obstacles.
   - Among the spots that fit, the one closest to the image center is chosen.
   - A visualization is saved (red = obstacles, faint green = all fitting spots,
     bright green outline = chosen spot).

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`ultralytics` will download the YOLO weights (`yolov8s-seg.pt`) automatically on
first run.

## Usage

Build the lookup table from a folder of car cut-outs:

```sh
python build_lookup.py --in frames/ --out car_positions.jsonl
```

Options: `--alpha-threshold` (default 30), `--epsilon` (contour simplification,
default 1.5), `--pattern` (default `*.png`).

Find a free spot in a live image:

```sh
python find_parking.py --lookup car_positions.jsonl --image live.jpg --out result.jpg
```

Options: `--model` (default `yolov8s-seg.pt`), `--conf` (default 0.25),
`--classes` (COCO classes treated as obstacles, default `2 5 7`),
`--imgsz` (default 1280), `--dilate` (safety margin in pixels, default 10).

## Notes

Camera frames, the generated lookup file, live images, results, and the YOLO
weights are excluded from version control via `.gitignore` — they are personal
data, not part of the project.
