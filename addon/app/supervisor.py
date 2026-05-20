"""Talk to the Home Assistant Supervisor API.

The Supervisor injects SUPERVISOR_TOKEN into the add-on container. With it we
can both reach Home Assistant core (camera snapshots) and discover the MQTT
broker credentials, so the user never has to configure either by hand.
"""
import os

import requests

_TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")
_HEADERS = {"Authorization": f"Bearer {_TOKEN}"}


def get_mqtt_config():
    """Return broker connection info from the `mqtt:need` service."""
    r = requests.get("http://supervisor/services/mqtt",
                     headers=_HEADERS, timeout=10)
    r.raise_for_status()
    data = r.json()["data"]
    return {
        "host": data["host"],
        "port": int(data["port"]),
        "username": data.get("username") or None,
        "password": data.get("password") or None,
        "ssl": bool(data.get("ssl", False)),
    }


def fetch_camera_snapshot(entity_id):
    """Return the current JPEG/PNG bytes of a camera entity."""
    url = f"http://supervisor/core/api/camera_proxy/{entity_id}"
    r = requests.get(url, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return r.content
