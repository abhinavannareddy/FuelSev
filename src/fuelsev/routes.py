"""Road scenarios representing Indian haul conditions.

Each scenario is an illustrative composite of conditions common on Indian roads:
patched national highways with toll plazas, ghat roads with hairpins, quarry and mine
haul roads with weighbridges, towns full of speed breakers and railway level crossings,
and monsoon mud. They are NOT surveyed data. Segment lengths, grades, speed caps and
hazard rates are assumptions chosen to be plausible and are easy to edit here.

Speed caps have two parts:
  * ``legal_kmh``   : the posted or statutory limit for a heavy goods vehicle (assumed values;
                      check the limits that apply to your state and vehicle class).
  * ``traffic_kmh`` : the speed mixed traffic actually allows (slow tractors, handcarts,
                      cattle, two-wheelers). The effective cap is the smaller of the two.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

KMH = 1.0 / 3.6


@dataclass(frozen=True)
class Surface:
    name: str
    crr: float  # rolling-resistance coefficient
    mu: float  # tyre-road friction (limits braking)


SURFACES: dict[str, Surface] = {
    s.name: s
    for s in (
        Surface("smooth_nh", 0.007, 0.70),  # good four-lane highway
        Surface("patchy", 0.011, 0.65),  # patched tarmac
        Surface("potholed", 0.020, 0.55),  # broken tarmac with potholes
        Surface("gravel", 0.030, 0.45),  # compacted site road
        Surface("rough_gravel", 0.040, 0.40),  # rutted site road
        Surface("mud", 0.055, 0.28),  # monsoon mud
    )
}

# Event kinds
KIND_NONE, KIND_BUMP, KIND_STOP, KIND_HAIRPIN = 0, 1, 2, 3
KIND_NAMES = {KIND_NONE: "none", KIND_BUMP: "bump", KIND_STOP: "stop", KIND_HAIRPIN: "hairpin"}


@dataclass(frozen=True)
class Segment:
    length_m: float
    grade_pct: float
    surface: str
    legal_kmh: float
    traffic_kmh: float | None = None
    hazard_per_km: float = 0.5  # unplanned slow-downs per km (cattle, two-wheelers, tractors)
    label: str = ""

    @property
    def cap_kmh(self) -> float:
        return min(self.legal_kmh, self.traffic_kmh) if self.traffic_kmh else self.legal_kmh


@dataclass(frozen=True)
class Event:
    at_m: float
    kind: str  # "bump" | "stop" | "hairpin"
    limit_kmh: float = 0.0  # speed to pass a bump or hairpin
    dwell_s: float = 0.0  # time stopped (toll, weighbridge, signal, level crossing)
    label: str = ""


@dataclass
class RouteProfile:
    """A route sampled at uniform distance steps for the planner and simulator."""

    route_id: str
    ds: float
    s: np.ndarray  # node positions (n+1)
    grade_pct: np.ndarray  # per interval (n)
    crr: np.ndarray
    mu: np.ndarray
    hazard_per_km: np.ndarray
    cap_ms: np.ndarray  # per node: effective speed cap (0 at stops)
    legal_ms: np.ndarray  # per node: legal limit only
    kind: np.ndarray  # per node: event kind code
    dwell_s: np.ndarray  # per node
    surface: list[str]  # per interval

    @property
    def n_nodes(self) -> int:
        return self.s.size

    @property
    def length_m(self) -> float:
        return float(self.s[-1])

    @property
    def stop_mask(self) -> np.ndarray:
        return self.cap_ms <= 0.0

    def interval_index(self, pos_m: float) -> int:
        i = int(pos_m / self.ds + 1e-9)
        return min(max(i, 0), self.grade_pct.size - 1)


@dataclass(frozen=True)
class Route:
    id: str
    name: str
    description: str
    segments: tuple[Segment, ...]
    events: tuple[Event, ...] = ()

    @property
    def length_m(self) -> float:
        return sum(s.length_m for s in self.segments)

    def reversed(self, new_id: str | None = None) -> Route:
        total = self.length_m
        segs = tuple(replace(s, grade_pct=-s.grade_pct) for s in reversed(self.segments))
        evs = tuple(sorted((replace(e, at_m=total - e.at_m) for e in self.events), key=lambda e: e.at_m))
        return Route(new_id or f"{self.id}_rev", self.name, self.description, segs, evs)

    def segment_at(self, pos_m: float) -> Segment:
        edge = 0.0
        for seg in self.segments:
            edge += seg.length_m
            if pos_m < edge:
                return seg
        return self.segments[-1]

    def profile(self, ds: float = 25.0) -> RouteProfile:
        total = self.length_m
        n = max(1, round(total / ds))
        ds_eff = total / n
        s = np.arange(n + 1) * ds_eff
        mids = (np.arange(n) + 0.5) * ds_eff
        segs = [self.segment_at(m) for m in mids]

        grade = np.array([sg.grade_pct for sg in segs])
        crr = np.array([SURFACES[sg.surface].crr for sg in segs])
        mu = np.array([SURFACES[sg.surface].mu for sg in segs])
        hazard = np.array([sg.hazard_per_km for sg in segs])
        cap_i = np.array([sg.cap_kmh for sg in segs]) * KMH
        legal_i = np.array([sg.legal_kmh for sg in segs]) * KMH

        # A node must respect the slower of the two intervals meeting there.
        cap = np.empty(n + 1)
        legal = np.empty(n + 1)
        cap[0], cap[-1] = cap_i[0], cap_i[-1]
        legal[0], legal[-1] = legal_i[0], legal_i[-1]
        cap[1:-1] = np.minimum(cap_i[:-1], cap_i[1:])
        legal[1:-1] = np.minimum(legal_i[:-1], legal_i[1:])

        kind = np.zeros(n + 1, dtype=int)
        dwell = np.zeros(n + 1)
        floor = 2.5  # m/s: below this the planner could not keep a gear engaged
        cap = np.maximum(cap, floor)
        for ev in self.events:
            idx = int(round(ev.at_m / ds_eff))
            idx = min(max(idx, 1), n - 1)
            if ev.kind == "stop":
                kind[idx] = KIND_STOP
                dwell[idx] = ev.dwell_s
            elif ev.kind in ("bump", "hairpin"):
                if kind[idx] != KIND_STOP:
                    kind[idx] = KIND_BUMP if ev.kind == "bump" else KIND_HAIRPIN
                cap[idx] = min(cap[idx], max(ev.limit_kmh * KMH, floor))
            else:
                raise ValueError(f"unknown event kind: {ev.kind}")
        # Start and end of every leg are stops (loading point, dump point).
        kind[0] = kind[-1] = KIND_STOP
        stops = kind == KIND_STOP
        cap = np.where(stops, 0.0, cap)

        return RouteProfile(
            route_id=self.id,
            ds=ds_eff,
            s=s,
            grade_pct=grade,
            crr=crr,
            mu=mu,
            hazard_per_km=hazard,
            cap_ms=cap,
            legal_ms=legal,
            kind=kind,
            dwell_s=dwell,
            surface=[sg.surface for sg in segs],
        )


@dataclass(frozen=True)
class Scenario:
    """A haul cycle: load at one end, drive the loaded leg, dump, return empty."""

    id: str
    name: str
    description: str
    loaded: Route  # route driven with payload
    empty: Route  # route driven on the return trip
    load_dwell_s: float = 90.0  # loading time at the start
    dump_dwell_s: float = 45.0  # tipping time at the end

    def leg(self, name: str) -> Route:
        if name == "loaded":
            return self.loaded
        if name == "empty":
            return self.empty
        raise ValueError(f"leg must be 'loaded' or 'empty', not {name!r}")


# --------------------------------------------------------------------------------------
# Scenario definitions
# --------------------------------------------------------------------------------------
def _nh_corridor() -> Scenario:
    route = Route(
        id="nh_corridor",
        name="NH corridor",
        description=(
            "12 km of national highway with a patched stretch, a toll plaza and a village "
            "bypass with speed breakers."
        ),
        segments=(
            Segment(3000, +0.5, "smooth_nh", 60, 55, 0.6, "four-lane section"),
            Segment(2500, -0.8, "patchy", 60, 45, 1.0, "patched stretch, slow vehicles"),
            Segment(3000, +1.5, "patchy", 60, 50, 0.8, "rolling terrain"),
            Segment(1000, 0.0, "potholed", 40, 30, 2.0, "village bypass"),
            Segment(2500, -0.5, "smooth_nh", 60, 55, 0.6, "four-lane section"),
        ),
        events=(
            Event(5500, "stop", dwell_s=35, label="toll plaza"),
            Event(8700, "bump", 15, label="speed breaker"),
            Event(9100, "bump", 15, label="speed breaker"),
            Event(9400, "bump", 15, label="speed breaker"),
        ),
    )
    return Scenario(route.id, route.name, route.description, route, route.reversed(route.id))


def _ghat() -> tuple[Scenario, Scenario]:
    climb = Route(
        id="ghat_climb",
        name="Ghat climb (loaded uphill)",
        description="9 km hill road: 6 km at 6.5 % with six hairpins, loaded uphill, empty back down.",
        segments=(
            Segment(1500, +1.0, "patchy", 50, 45, 0.8, "approach"),
            Segment(6000, +6.5, "patchy", 40, 30, 0.6, "ghat section"),
            Segment(1500, +0.5, "smooth_nh", 50, 40, 0.8, "summit plateau"),
        ),
        events=tuple(Event(x, "hairpin", 15, label="hairpin") for x in (2500, 3300, 4100, 4900, 5700, 6500)),
    )
    climb_s = Scenario(climb.id, climb.name, climb.description, climb, climb.reversed(climb.id))
    down_route = climb.reversed("ghat_descent")
    down = Scenario(
        "ghat_descent",
        "Ghat descent (loaded downhill)",
        "The same hill road driven loaded downhill: engine braking and brake temperature matter.",
        down_route,
        down_route.reversed("ghat_descent"),
    )
    return climb_s, down


def _quarry() -> Scenario:
    route = Route(
        id="quarry_haul",
        name="Quarry haul road",
        description="4 km unpaved site road: 8 % pit ramp, weighbridge, rutted patch, site gate bump.",
        segments=(
            Segment(600, +8.0, "gravel", 20, 18, 0.2, "pit ramp"),
            Segment(1600, +1.0, "gravel", 25, 22, 0.2, "main haul road"),
            Segment(600, 0.0, "rough_gravel", 20, 15, 0.3, "rutted patch"),
            Segment(1200, -2.0, "gravel", 25, 22, 0.2, "approach to crusher"),
        ),
        events=(
            Event(2200, "stop", dwell_s=40, label="weighbridge"),
            Event(3500, "bump", 10, label="site gate speed breaker"),
        ),
    )
    return Scenario(route.id, route.name, route.description, route, route.reversed(route.id), 120.0, 60.0)


def _urban() -> Scenario:
    segs = (
        Segment(1500, 0.0, "patchy", 40, 28, 2.5, "market road"),
        Segment(1500, +0.3, "patchy", 40, 24, 2.5, "bazaar stretch"),
        Segment(1500, 0.0, "potholed", 40, 30, 2.5, "ring road, potholes"),
        Segment(1500, -0.3, "patchy", 40, 26, 2.5, "school zone road"),
    )
    bumps = (300, 750, 1200, 1900, 2400, 3300, 3900, 4600, 5200, 5700)
    events = tuple(Event(x, "bump", 20, label="speed breaker") for x in bumps) + (
        Event(1500, "stop", dwell_s=30, label="signal junction"),
        Event(3000, "stop", dwell_s=90, label="railway level crossing (gate closed)"),
        Event(4500, "stop", dwell_s=30, label="signal junction"),
    )
    route = Route(
        "urban_mixed",
        "Town transit",
        "6 km through a town: ten speed breakers, two junctions and a railway level crossing.",
        segs,
        events,
    )
    return Scenario(route.id, route.name, route.description, route, route.reversed(route.id))


def _monsoon(base: Scenario) -> Scenario:
    def wet(route: Route, rid: str) -> Route:
        segs = tuple(
            replace(
                s,
                surface="mud" if s.surface in ("gravel", "rough_gravel") else s.surface,
                legal_kmh=s.legal_kmh * 0.75,
                traffic_kmh=(s.traffic_kmh or s.legal_kmh) * 0.7,
                hazard_per_km=s.hazard_per_km * 1.5,
            )
            for s in route.segments
        )
        return Route(rid, route.name, route.description, segs, route.events)

    loaded = wet(base.loaded, "monsoon_site")
    return Scenario(
        "monsoon_site",
        "Monsoon site road",
        "The quarry haul road in monsoon: mud, lower speeds, poorer grip.",
        loaded,
        loaded.reversed("monsoon_site"),
        base.load_dwell_s,
        base.dump_dwell_s,
    )


def _build_scenarios() -> dict[str, Scenario]:
    nh = _nh_corridor()
    climb, descent = _ghat()
    quarry = _quarry()
    urban = _urban()
    monsoon = _monsoon(quarry)
    return {s.id: s for s in (nh, climb, descent, quarry, urban, monsoon)}


SCENARIOS: dict[str, Scenario] = _build_scenarios()


def get_scenario(scenario_id: str) -> Scenario:
    try:
        return SCENARIOS[scenario_id]
    except KeyError:
        raise KeyError(f"unknown route {scenario_id!r}; choose from {sorted(SCENARIOS)}") from None
