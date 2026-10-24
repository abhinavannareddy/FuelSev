"""Heavy-truck longitudinal dynamics and fuel use.

Everything here is vectorised with NumPy so the same physics serves the DP planner
(thousands of candidate transitions at once), the time-domain simulator and the
in-cab coach (which asks "what would each gear burn right now?").
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .constants import DIESEL_DENSITY_KG_PER_L, G
from .engine import (
    RPM_IDLE,
    RPM_MAX,
    engine_brake_torque,
    friction_power_kw,
    full_load_torque,
)
from .fuelmap import FuelMap, load_default_fuel_map


@dataclass(frozen=True)
class VehicleParams:
    """Representative 6x4 tipper (about 28 t GVW class). Not a manufacturer datasheet."""

    name: str = "Generic 6x4 tipper, 28 t GVW class (representative)"
    mass_empty_kg: float = 11000.0  # includes driver and fuel
    payload_rated_kg: float = 17000.0
    gear_ratios: tuple[float, ...] = (12.0, 8.6, 6.3, 4.6, 3.4, 2.5, 1.8, 1.3, 1.0)
    final_drive: float = 5.0
    wheel_radius_m: float = 0.51  # 11.00R20 class tyre, loaded rolling radius
    driveline_eff: float = 0.92
    cd_a_m2: float = 5.25  # drag coefficient x frontal area
    air_density: float = 1.15  # kg/m^3, hot Indian plains (about 35 C)
    engine_inertia_kgm2: float = 2.0  # engine, flywheel and clutch
    wheel_inertia_kgm2: float = 80.0  # wheels, axles, driveline (reflected to the wheels)
    tank_l: float = 300.0
    launch_speed_ms: float = 1.6  # below this, gears 1-2 may slip the clutch

    # -- helpers ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> VehicleParams:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "gear_ratios" in raw:
            raw["gear_ratios"] = tuple(raw["gear_ratios"])
        return cls(**raw)


class Transition(NamedTuple):
    """Result of moving from speed v0 to v1 over a distance step."""

    fuel_l: np.ndarray
    dt_s: np.ndarray
    feasible: np.ndarray
    service_brake_mj: np.ndarray


@dataclass
class Vehicle:
    params: VehicleParams = field(default_factory=VehicleParams)
    fuel_map: FuelMap = field(default_factory=load_default_fuel_map)

    def __post_init__(self) -> None:
        self._ratios = np.asarray(self.params.gear_ratios, dtype=float) * self.params.final_drive

    # -- drivetrain -------------------------------------------------------------
    @property
    def n_gears(self) -> int:
        return len(self.params.gear_ratios)

    def ratio(self, gear):
        """Overall ratio (gearbox x final drive). Gear 0 is neutral and gives 0."""
        g = np.asarray(gear, dtype=int)
        out = np.where(g > 0, self._ratios[np.clip(g, 1, self.n_gears) - 1], 0.0)
        return float(out) if out.ndim == 0 else out

    def rpm_at(self, v, gear):
        """Engine speed in rpm implied by road speed and gear (no clutch slip)."""
        return np.asarray(v, dtype=float) * self.ratio(gear) / self.params.wheel_radius_m * 30.0 / np.pi

    def mass(self, payload_kg) -> float:
        return self.params.mass_empty_kg + payload_kg

    def m_eff(self, gear, mass):
        """Mass plus rotating inertia reflected to the wheels (large in low gears)."""
        r2 = self.params.wheel_radius_m**2
        ratio = np.asarray(self.ratio(gear))
        return mass + self.params.wheel_inertia_kgm2 / r2 + self.params.engine_inertia_kgm2 * ratio**2 / r2

    # -- forces -----------------------------------------------------------------
    def resistance_n(self, v, mass, grade_pct, crr):
        """Rolling + grade + aerodynamic resistance at the wheels, in N."""
        theta = np.arctan(np.asarray(grade_pct, dtype=float) / 100.0)
        rolling_grade = mass * G * (crr * np.cos(theta) + np.sin(theta))
        aero = 0.5 * self.params.air_density * self.params.cd_a_m2 * np.asarray(v, dtype=float) ** 2
        return rolling_grade + aero

    def force_to_torque(self, force_n, gear):
        """Engine torque (Nm) that produces a wheel force; negative when the engine is dragged."""
        p = self.params
        ratio = np.asarray(self.ratio(gear))
        safe = np.where(ratio > 0, ratio, np.nan)
        f = np.asarray(force_n, dtype=float)
        return np.where(f >= 0, f * p.wheel_radius_m / (safe * p.driveline_eff), f * p.wheel_radius_m * p.driveline_eff / safe)

    def torque_to_force(self, torque_nm, gear):
        p = self.params
        ratio = np.asarray(self.ratio(gear))
        t = np.asarray(torque_nm, dtype=float)
        return np.where(t >= 0, t * ratio * p.driveline_eff / p.wheel_radius_m, t * ratio / (p.wheel_radius_m * p.driveline_eff))

    def engine_brake_force(self, rpm, gear):
        """Largest retarding wheel force from friction plus exhaust brake, in N (positive)."""
        return -self.torque_to_force(-engine_brake_torque(rpm), gear)

    # -- fuel -------------------------------------------------------------------
    def fuel_gps(self, rpm, torque_nm):
        """Fuel flow in g/s for an engine torque request (negative torque = overrun).

        Positive torque reads the fuel map. With negative torque the engine is dragged:
        fuel falls linearly to zero (fuel cut-off) once the drag exceeds friction.
        """
        rpm = np.asarray(rpm, dtype=float)
        t = np.asarray(torque_nm, dtype=float)
        drive = self.fuel_map.rate_gps(rpm, np.maximum(t, 0.0))
        omega = np.maximum(rpm, 1.0) * np.pi / 30.0
        t_fric = friction_power_kw(rpm) * 1000.0 / omega
        zero_torque = self.fuel_map.rate_gps(rpm, 0.0)
        overrun = zero_torque * np.clip(1.0 + t / t_fric, 0.0, 1.0)
        return np.where(t >= 0, drive, overrun)

    @property
    def idle_fuel_gps(self) -> float:
        return float(self.fuel_map.rate_gps(RPM_IDLE, 0.0))

    @staticmethod
    def grams_to_litres(grams):
        return np.asarray(grams, dtype=float) / (DIESEL_DENSITY_KG_PER_L * 1000.0)

    # -- DP transition ---------------------------------------------------------------
    def transition(
        self,
        v0,
        v1,
        gear,
        ds: float,
        grade_pct: float,
        crr: float,
        mass: float,
        *,
        a_acc_max: float = 1.0,
        a_dec_max: float = 1.5,
        rpm_run_min: float = 800.0,
        rpm_run_max: float = 1950.0,
        torque_margin: float = 0.98,
        traction_limit_n: float = np.inf,
    ) -> Transition:
        """Fuel, time and feasibility of covering ``ds`` metres from v0 to v1 in ``gear``.

        Arguments broadcast, so passing (nv,1,1), (1,nv,1) and (1,1,ng) arrays evaluates
        every (v0, v1, gear) combination at once. Speeds are assumed to change at a
        constant rate over the step, and the mean speed sets the aerodynamic and
        engine-speed terms.
        """
        v0 = np.asarray(v0, dtype=float)
        v1 = np.asarray(v1, dtype=float)
        gear = np.asarray(gear, dtype=int)
        p = self.params

        vm = 0.5 * (v0 + v1)
        moving = vm > 0
        safe_vm = np.where(moving, vm, 1.0)
        dt = np.where(moving, ds / safe_vm, np.inf)
        a = (v1**2 - v0**2) / (2.0 * ds)

        rpm_wheel = self.rpm_at(vm, gear)
        launch = (gear <= 2) & (vm < p.launch_speed_ms)
        rpm = np.where(launch, np.maximum(rpm_wheel, rpm_run_min), rpm_wheel)

        # Engine speed must be valid at both ends of the step, not just on average.
        # Only the launch gears may slip the clutch at low speed, so the planner cannot
        # pull away from rest in a high gear.
        rpm0 = self.rpm_at(v0, gear)
        rpm1 = self.rpm_at(v1, gear)
        ends_ok = (
            (((gear <= 2) & (v0 < p.launch_speed_ms)) | (rpm0 >= rpm_run_min - 1e-9))
            & (((gear <= 2) & (v1 < p.launch_speed_ms)) | (rpm1 >= rpm_run_min - 1e-9))
            & (rpm0 <= rpm_run_max)
            & (rpm1 <= rpm_run_max)
        )

        meff = self.m_eff(gear, mass)
        f_need = meff * a + self.resistance_n(vm, mass, grade_pct, crr)
        torque = self.force_to_torque(f_need, gear)

        feasible = (
            moving
            & ends_ok
            & (a <= a_acc_max)
            & (a >= -a_dec_max)
            & (rpm >= rpm_run_min - 1e-9)
            & (rpm <= rpm_run_max)
            & (torque <= torque_margin * full_load_torque(rpm))
            & (f_need <= traction_limit_n)
        )

        dt_safe = np.where(moving, dt, 0.0)  # stationary steps are infeasible anyway
        fuel_gps = self.fuel_gps(rpm, torque)
        fuel_l = self.grams_to_litres(fuel_gps * dt_safe)

        f_eb = self.engine_brake_force(rpm, gear)
        service_n = np.maximum(0.0, -f_need - f_eb)
        service_mj = service_n * vm * dt_safe / 1e6

        fuel_l = np.where(feasible, fuel_l, np.inf)
        return Transition(fuel_l, dt, feasible, np.where(feasible, service_mj, np.inf))

    # -- what would each gear burn right now? ------------------------------------
    def gear_options(
        self,
        v: float,
        accel: float,
        grade_pct: float,
        crr: float,
        mass: float,
        *,
        rpm_min: float = 900.0,
        rpm_max: float = 2000.0,
        torque_margin: float = 0.95,
    ):
        """Fuel flow (g/s), rpm and feasibility for gears 1..N at the current operating state.

        Returns ``(gears, fuel_gps, rpm, feasible, demand_n)``. ``demand_n`` is the wheel
        force required; when it is not positive (overrun) the fuel numbers are not
        meaningful for coaching and the caller should skip them. ``demand_n`` has one entry
        per gear because the reflected engine inertia differs between gears.
        """
        gears = np.arange(1, self.n_gears + 1)
        rpm = self.rpm_at(v, gears)
        meff = self.m_eff(gears, mass)
        demand = meff * accel + self.resistance_n(v, mass, grade_pct, crr)
        torque = self.force_to_torque(demand, gears)
        feasible = (
            (rpm >= rpm_min) & (rpm <= min(rpm_max, RPM_MAX)) & (torque <= torque_margin * full_load_torque(rpm))
        )
        fuel = self.fuel_gps(rpm, torque)
        return gears, fuel, rpm, feasible, demand
