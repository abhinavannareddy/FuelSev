"""MQTT plumbing: topics, a thin paho client wrapper, and store-and-forward.

Topic layout (``v1`` is the schema version):

    fuelsev/v1/<machine_id>/telemetry   QoS 0, 1 Hz, dropped while offline (it is a live view)
    fuelsev/v1/<machine_id>/coach       QoS 0, cab-display messages
    fuelsev/v1/<machine_id>/alerts      QoS 1, buffered on the machine while offline
    fuelsev/v1/<machine_id>/trips       QoS 1, buffered on the machine while offline
    fuelsev/v1/<machine_id>/status      QoS 1, retained; "offline" is set as the last will

Indian haul routes pass through ghats, cuttings and quarries with no mobile coverage, so
alerts and trip summaries must not be lost when the link drops. The ``ReliableSender``
keeps them in a bounded queue on the machine and sends them, oldest first, once the link
is back. Alert ids are deterministic, so a message delivered twice is stored once.
"""

from __future__ import annotations

import logging
import os
import ssl
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import paho.mqtt.client as mqtt

from .schemas import MACHINE_ID_RE

log = logging.getLogger(__name__)

TOPIC_ROOT = "fuelsev/v1"
KINDS = ("telemetry", "coach", "alerts", "trips", "status")
MAX_PAYLOAD_BYTES = 64 * 1024


def topic(machine_id: str, kind: str) -> str:
    if not MACHINE_ID_RE.match(machine_id):
        raise ValueError(f"invalid machine id {machine_id!r}")
    if kind not in KINDS:
        raise ValueError(f"unknown topic kind {kind!r}")
    return f"{TOPIC_ROOT}/{machine_id}/{kind}"


def parse_topic(t: str) -> tuple[str, str] | None:
    """Return (machine_id, kind) for a well-formed fuelsev topic, otherwise None."""
    parts = t.split("/")
    if len(parts) != 4 or f"{parts[0]}/{parts[1]}" != TOPIC_ROOT:
        return None
    machine_id, kind = parts[2], parts[3]
    if not MACHINE_ID_RE.match(machine_id) or kind not in KINDS:
        return None
    return machine_id, kind


@dataclass(frozen=True)
class BrokerConfig:
    host: str = "localhost"
    port: int = 1883
    username: str | None = None
    password: str | None = None
    tls: bool = False
    keepalive: int = 30

    @classmethod
    def from_env(cls) -> BrokerConfig:
        return cls(
            host=os.environ.get("FUELSEV_MQTT_HOST", "localhost"),
            port=int(os.environ.get("FUELSEV_MQTT_PORT", "1883")),
            username=os.environ.get("FUELSEV_MQTT_USER") or None,
            password=os.environ.get("FUELSEV_MQTT_PASSWORD") or None,
            tls=os.environ.get("FUELSEV_MQTT_TLS", "0") == "1",
        )


class Link:
    """A paho client with automatic reconnect and an optional last-will status message."""

    def __init__(
        self,
        cfg: BrokerConfig,
        client_id: str,
        *,
        will: tuple[str, str] | None = None,
        on_message: Callable[[str, bytes], None] | None = None,
        subscriptions: tuple[tuple[str, int], ...] = (),
        persistent_session: bool = False,
    ):
        self.cfg = cfg
        self._subscriptions = subscriptions
        self._on_message = on_message
        self._connected = threading.Event()
        self.simulated_outage = False  # lets a demo cut the link without touching the socket
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            clean_session=not persistent_session,
        )
        if cfg.username:
            self.client.username_pw_set(cfg.username, cfg.password)
        if cfg.tls:
            self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        if will:
            self.client.will_set(will[0], will[1], qos=1, retain=True)
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.on_connect = self._handle_connect
        self.client.on_disconnect = self._handle_disconnect
        self.client.on_message = self._handle_message

    # -- callbacks (paho network thread) ---------------------------------------------
    def _handle_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.warning("MQTT connect refused: %s", reason_code)
            return
        log.info("MQTT connected to %s:%s", self.cfg.host, self.cfg.port)
        for topic_filter, qos in self._subscriptions:
            client.subscribe(topic_filter, qos)
        self._connected.set()

    def _handle_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        log.warning("MQTT disconnected: %s", reason_code)
        self._connected.clear()

    def _handle_message(self, client, userdata, msg):
        if self._on_message is not None:
            self._on_message(msg.topic, bytes(msg.payload))

    # -- public --------------------------------------------------------------------
    def start(self) -> None:
        self.client.connect_async(self.cfg.host, self.cfg.port, self.cfg.keepalive)
        self.client.loop_start()

    def wait_connected(self, timeout: float = 10.0) -> bool:
        return self._connected.wait(timeout)

    @property
    def connected(self) -> bool:
        return self._connected.is_set() and not self.simulated_outage

    def publish(self, topic_name: str, payload: str, qos: int = 0, retain: bool = False) -> bool:
        """Publish if the link is up. Returns False when nothing was sent."""
        if not self.connected:
            return False
        info = self.client.publish(topic_name, payload, qos=qos, retain=retain)
        return info.rc == mqtt.MQTT_ERR_SUCCESS

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()


class ReliableSender:
    """Bounded FIFO that holds messages until they can be sent.

    ``send`` tries to publish straight away, but only if nothing older is waiting, so
    order is preserved. ``flush`` drains the queue when the link returns. When the queue
    is full the oldest message is dropped (and counted) rather than blocking the machine.
    """

    def __init__(self, publish: Callable[[str, str, int, bool], bool], maxlen: int = 5000):
        self._publish = publish
        self._queue: deque[tuple[str, str, int, bool]] = deque()
        self._maxlen = maxlen
        self.dropped = 0

    def __len__(self) -> int:
        return len(self._queue)

    def send(self, topic_name: str, payload: str, qos: int = 1, retain: bool = False) -> None:
        if not self._queue and self._publish(topic_name, payload, qos, retain):
            return
        if len(self._queue) >= self._maxlen:
            self._queue.popleft()
            self.dropped += 1
        self._queue.append((topic_name, payload, qos, retain))

    def flush(self) -> int:
        sent = 0
        while self._queue:
            item = self._queue[0]
            if not self._publish(*item):
                break
            self._queue.popleft()
            sent += 1
        return sent
