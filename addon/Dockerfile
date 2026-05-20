FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# libglib2.0-0 is required by opencv even in the headless build
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/parking-finder

# Install the CPU-only torch wheel first so ultralytics does not pull CUDA.
# ultralytics depends on the full opencv-python (needs X11/GUI libs); swap it
# for the headless build afterwards so no GUI system libraries are required.
COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python || true \
    && pip install --no-cache-dir --force-reinstall opencv-python-headless==4.13.0.92

# Bake the YOLO weights into the image so the first run needs no network
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8s-seg.pt')"

COPY app ./app

CMD ["python3", "-m", "app.main"]
