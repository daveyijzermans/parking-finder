"""Publish parking results to Home Assistant over MQTT discovery.

Discovery auto-creates three entities under one device:
  - binary_sensor : space available (with chosen spot / counts as attributes)
  - sensor        : number of cars detected
  - camera        : the debug visualization image
"""
import json

import paho.mqtt.client as mqtt

_PREFIX = "parking_finder"
_DISCOVERY = "homeassistant"

_DEVICE = {
    "identifiers": [_PREFIX],
    "name": "Parking Finder",
    "manufacturer": "parking-finder",
    "model": "YOLO segmentation",
}

# state/image topics
T_SPACE = f"{_PREFIX}/space/state"
T_ATTRS = f"{_PREFIX}/space/attributes"
T_CARS = f"{_PREFIX}/cars/state"
T_IMAGE = f"{_PREFIX}/debug/image"
T_AVAIL = f"{_PREFIX}/status"


class Publisher:
    def __init__(self, host, port, username=None, password=None, ssl=False):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=_PREFIX)
        if username:
            self.client.username_pw_set(username, password)
        if ssl:
            self.client.tls_set()
        self.client.will_set(T_AVAIL, "offline", retain=True)
        self.client.connect(host, port, keepalive=60)
        self.client.loop_start()

    def publish_discovery(self):
        avail = {"availability_topic": T_AVAIL,
                 "payload_available": "online",
                 "payload_not_available": "offline"}

        self._cfg("binary_sensor", "space", {
            "name": "Parking space available",
            "unique_id": f"{_PREFIX}_space",
            "state_topic": T_SPACE,
            "payload_on": "ON",
            "payload_off": "OFF",
            "json_attributes_topic": T_ATTRS,
            "icon": "mdi:parking",
            **avail,
        })
        self._cfg("sensor", "cars", {
            "name": "Cars detected",
            "unique_id": f"{_PREFIX}_cars",
            "state_topic": T_CARS,
            "state_class": "measurement",
            "icon": "mdi:car-multiple",
            **avail,
        })
        self._cfg("camera", "debug", {
            "name": "Parking debug image",
            "unique_id": f"{_PREFIX}_debug",
            "topic": T_IMAGE,
            **avail,
        })
        self.client.publish(T_AVAIL, "online", retain=True)

    def _cfg(self, component, object_id, payload):
        topic = f"{_DISCOVERY}/{component}/{_PREFIX}/{object_id}/config"
        payload["device"] = _DEVICE
        self.client.publish(topic, json.dumps(payload), retain=True)

    def publish_result(self, result):
        self.client.publish(T_SPACE, "ON" if result.space_available else "OFF",
                             retain=True)
        self.client.publish(T_CARS, str(result.cars_detected), retain=True)
        self.client.publish(T_ATTRS, json.dumps({
            "chosen_spot": result.chosen_spot,
            "free_spots": result.free_spots,
            "cars_detected": result.cars_detected,
        }), retain=True)
        self.client.publish(T_IMAGE, result.image_jpeg, retain=True)

    def close(self):
        self.client.publish(T_AVAIL, "offline", retain=True)
        self.client.loop_stop()
        self.client.disconnect()
