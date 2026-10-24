"""Driver behaviour models for the simulator.

Three human styles and one planner-following driver:

* ``eco``        : anticipates, short-shifts, keeps a long following gap, switches the
                   engine off when waiting at a toll or level crossing.
* ``typical``    : drives at the speed traffic allows, shifts late, brakes harder, and
                   sometimes coasts downhill in neutral.
* ``aggressive`` : above the traffic speed, late braking, short headway, high revs,
                   frequent neutral coasting.
* ``planner``    : tracks the DP planner's speed and gear plan (the "what could be
                   achieved" benchmark).

On top of its style, a driver can have short unsafe *episodes* (overspeeding, tailgating,
coasting in neutral, erratic speed control, for example from fatigue). Episodes are the
hidden ground truth used to evaluate the detectors; the detectors never see them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DriverProfile:
    name: str
    cap_factor: float  # fraction of the effective speed cap the driver cruises at
    look_decel: float  # m/s^2: starts slowing when the needed decel reaches this
    a_acc: float  # maximum commanded acceleration, m/s^2
    a_dec_max: float  # maximum commanded deceleration, m/s^2
    kp: float  # speed-tracking gain, 1/s
    headway_s: float  # time gap to the vehicle ahead
    shift_up_rpm: float
    shift_down_rpm: float
    lug_frac: float  # downshift when torque demand exceeds this fraction of full load
    descent_rpm: float  # on steep descents hold the highest gear with rpm <= this
    neutral_coast_prob: float  # chance of coasting in neutral on a given steep descent
    idle_shutdown_s: float | None  # engine off after waiting this long (None = never)
    episode_rate_per_h: float  # unsafe episodes per hour of driving
    speed_noise: float  # std of speed-target wander, m/s
    overload_prob: float  # chance a trip is loaded above the rated payload
    overload_max: float  # maximum overload fraction
    payload_lo: float = 0.88  # normal load as a fraction of rated payload
    payload_hi: float = 1.00
    follow_plan: bool = False


ECO = DriverProfile(
    name="eco",
    cap_factor=0.92,
    look_decel=0.5,
    a_acc=0.6,
    a_dec_max=2.0,
    kp=0.35,
    headway_s=3.0,
    shift_up_rpm=1500,
    shift_down_rpm=1000,
    lug_frac=0.92,
    descent_rpm=1900,
    neutral_coast_prob=0.0,
    idle_shutdown_s=45.0,
    episode_rate_per_h=0.3,
    speed_noise=0.25,
    overload_prob=0.0,
    overload_max=0.0,
)

TYPICAL = DriverProfile(
    name="typical",
    cap_factor=1.0,
    look_decel=0.9,
    a_acc=0.9,
    a_dec_max=3.2,
    kp=0.5,
    headway_s=2.2,
    shift_up_rpm=1800,
    shift_down_rpm=1100,
    lug_frac=0.97,
    descent_rpm=1500,
    neutral_coast_prob=0.25,
    idle_shutdown_s=None,
    episode_rate_per_h=2.0,
    speed_noise=0.5,
    overload_prob=0.15,
    overload_max=0.20,
    payload_hi=1.05,
)

AGGRESSIVE = DriverProfile(
    name="aggressive",
    cap_factor=1.12,
    look_decel=1.8,
    a_acc=1.3,
    a_dec_max=4.5,
    kp=0.8,
    headway_s=1.0,
    shift_up_rpm=2000,
    shift_down_rpm=1300,
    lug_frac=1.0,
    descent_rpm=1300,
    neutral_coast_prob=0.6,
    idle_shutdown_s=None,
    episode_rate_per_h=8.0,
    speed_noise=0.9,
    overload_prob=0.40,
    overload_max=0.30,
    payload_lo=0.95,
    payload_hi=1.05,
)

PLANNER = DriverProfile(
    name="planner",
    cap_factor=1.0,
    look_decel=0.5,
    a_acc=1.0,
    a_dec_max=2.0,
    kp=0.8,
    headway_s=3.0,
    shift_up_rpm=1500,
    shift_down_rpm=1000,
    lug_frac=0.95,
    descent_rpm=1900,
    neutral_coast_prob=0.0,
    idle_shutdown_s=45.0,
    episode_rate_per_h=0.0,
    speed_noise=0.05,
    overload_prob=0.0,
    overload_max=0.0,
    follow_plan=True,
)

PROFILES: dict[str, DriverProfile] = {p.name: p for p in (ECO, TYPICAL, AGGRESSIVE, PLANNER)}


def get_profile(name: str | DriverProfile) -> DriverProfile:
    if isinstance(name, DriverProfile):
        return name
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown driver {name!r}; choose from {sorted(PROFILES)}") from None


# --- Unsafe episodes (hidden ground truth) ----------------------------------------------
EPISODE_KINDS = ("overspeed", "tailgate", "neutral_coast", "erratic")
EPISODE_WEIGHTS = (0.30, 0.30, 0.15, 0.25)


@dataclass(frozen=True)
class Episode:
    kind: str
    start_t: float
    end_t: float


# --- Injected machine faults -----------------------------------------------------------------
FAULT_KINDS = ("overheat", "low_oil", "fuel_leak", "fuel_theft", "injector")


@dataclass(frozen=True)
class FaultSpec:
    kind: str
    start_s: float  # machine clock time at which the fault begins

    def __post_init__(self) -> None:
        if self.kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault {self.kind!r}; choose from {FAULT_KINDS}")
