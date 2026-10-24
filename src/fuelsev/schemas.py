"""Message schemas shared by the machine, the in-cab agent and the supervisor API.

All MQTT payloads are validated against these models on receipt. The bounds are loose
physical sanity limits, so one corrupted message cannot poison the database or reports.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MACHINE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")

Severity = Literal["info", "warning", "critical"]
Category = Literal["safety", "fault", "efficiency", "load"]
Leg = Literal["loaded", "empty"]

_ALERT_NS = uuid.UUID("6f0c1a52-7c1e-4a53-9d0a-5b8f2f4f6a11")


def valid_machine_id(value: str) -> str:
    if not MACHINE_ID_RE.match(value):
        raise ValueError("machine_id must be 1-40 characters of letters, digits, '-' or '_'")
    return value


class Telemetry(BaseModel):
    """One 1 Hz sample from the machine. Counters are cumulative and monotonic."""

    model_config = ConfigDict(extra="ignore")

    machine_id: str
    ts: float = Field(ge=0)  # wall-clock epoch seconds when the sample was produced
    sim_t: float = Field(ge=0)  # machine clock in seconds since start
    trip_no: int = Field(ge=0, le=10_000_000)
    route_id: str = Field(max_length=40)
    leg: Leg
    pos_m: float = Field(ge=0, le=500_000)
    speed_kmh: float = Field(ge=0, le=200)
    accel_ms2: float = Field(ge=-20, le=20)
    gear: int = Field(ge=0, le=20)  # 0 = neutral or mid-shift
    rpm: float = Field(ge=0, le=4000)
    torque_pct: float = Field(ge=0, le=150)
    throttle_pct: float = Field(ge=0, le=100)
    brake_pct: float = Field(ge=0, le=100)
    grade_pct: float = Field(ge=-40, le=40)
    speed_limit_kmh: float = Field(ge=0, le=200)
    payload_t: float = Field(ge=0, le=200)
    fuel_lph: float = Field(ge=0, le=1000)
    fuel_total_l: float = Field(ge=0)
    fuel_level_l: float = Field(ge=0, le=5000)
    odometer_km: float = Field(ge=0)
    engine_s: float = Field(ge=0)
    idle_s: float = Field(ge=0)
    idle_fuel_l: float = Field(ge=0)
    coolant_c: float = Field(ge=-40, le=250)
    oil_kpa: float = Field(ge=0, le=2000)
    brake_temp_c: float = Field(ge=-40, le=1500)
    engine_on: bool

    @field_validator("machine_id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        return valid_machine_id(v)


class Alert(BaseModel):
    """Something a supervisor should know about."""

    model_config = ConfigDict(extra="ignore")

    id: str
    machine_id: str
    ts: float = Field(ge=0)
    sim_t: float = Field(ge=0)
    code: str = Field(max_length=40)
    severity: Severity
    category: Category
    message: str = Field(max_length=300)
    route_id: str = Field(default="", max_length=40)
    pos_m: float = Field(default=0.0, ge=0)
    evidence: dict[str, float | int | str | bool | None] = Field(default_factory=dict)

    @field_validator("machine_id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        return valid_machine_id(v)

    @staticmethod
    def make_id(machine_id: str, code: str, sim_t: float) -> str:
        """Deterministic id, so a message redelivered by MQTT QoS 1 is stored once."""
        key = f"{machine_id}|{code}|{sim_t:.1f}"
        return str(uuid.uuid5(_ALERT_NS, hashlib.sha1(key.encode()).hexdigest()))


class CoachMessage(BaseModel):
    """A short instruction for the operator's cab display."""

    model_config = ConfigDict(extra="ignore")

    machine_id: str
    ts: float = Field(ge=0)
    sim_t: float = Field(ge=0)
    code: str = Field(max_length=40)
    priority: int = Field(ge=1, le=3)  # 1 = safety, 2 = fault, 3 = efficiency
    lang: str = Field(max_length=5)
    text: str = Field(max_length=200)
    params: dict[str, float | int | str] = Field(default_factory=dict)


class TripSummary(BaseModel):
    """What one haul cycle (loaded leg plus return) cost."""

    model_config = ConfigDict(extra="ignore")

    machine_id: str
    trip_no: int = Field(ge=0)
    route_id: str = Field(max_length=40)
    start_ts: float = Field(ge=0)
    end_ts: float = Field(ge=0)
    duration_s: float = Field(ge=0)
    payload_t: float = Field(ge=0)
    distance_km: float = Field(ge=0)  # whole cycle: loaded leg plus return
    loaded_distance_km: float = Field(ge=0)  # loaded leg only
    fuel_l: float = Field(ge=0)  # whole cycle
    idle_s: float = Field(ge=0)
    idle_fuel_l: float = Field(ge=0)
    harsh_brake_events: int = Field(ge=0)
    overspeed_s: float = Field(ge=0)
    eco_score: float = Field(ge=0, le=100)

    @property
    def trip_id(self) -> str:
        return f"{self.machine_id}:{self.trip_no}"

    @field_validator("machine_id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        return valid_machine_id(v)


class Status(BaseModel):
    model_config = ConfigDict(extra="ignore")

    machine_id: str
    online: bool
    ts: float = Field(ge=0)

    @field_validator("machine_id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        return valid_machine_id(v)
