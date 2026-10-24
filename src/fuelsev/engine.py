"""Representative heavy-duty diesel engine model.

This is a *synthetic* engine (about 8 L, 260 kW) built to behave like a modern
heavy-duty diesel: a best-efficiency island near 1200-1500 rpm and 70-90 % load,
rising specific consumption at light load and at high speed, and a fixed fuel
cost for friction and accessories (the Willans-line idea).

It is not Volvo data. To use a measured map, replace ``data/fuel_map.csv`` (see
``fuelmap.FuelMap.from_csv``) and, if needed, the torque curve below.
"""

from __future__ import annotations

import numpy as np

from .constants import DIESEL_LHV_MJ_PER_KG

RPM_IDLE = 600.0
RPM_MAX = 2100.0  # governed maximum

DISPLACEMENT_L = 8.0
PEAK_INDICATED_EFFICIENCY = 0.45

# Full-load torque curve (rpm -> Nm). Peak power about 259 kW near 1700-1900 rpm.
_T_RPM = np.array([600.0, 800.0, 1000.0, 1100.0, 1500.0, 1700.0, 1900.0, 2100.0])
_T_NM = np.array([1000.0, 1350.0, 1550.0, 1600.0, 1600.0, 1450.0, 1300.0, 1050.0])


def full_load_torque(rpm):
    """Maximum engine torque in Nm at the given speed(s)."""
    return np.interp(rpm, _T_RPM, _T_NM)


def engine_brake_torque(rpm):
    """Retarding torque (Nm, positive number) available with the throttle closed.

    Friction plus an exhaust brake. About 230 Nm at 1000 rpm and 540 Nm at
    2100 rpm. Assumption, typical of a heavy-duty exhaust brake.
    """
    return 120.0 + 0.28 * (np.clip(rpm, RPM_IDLE, RPM_MAX) - RPM_IDLE)


def friction_power_kw(rpm):
    """Friction plus accessory power in kW (always drawn while the engine runs)."""
    rpm = np.asarray(rpm, dtype=float)
    fmep_pa = 0.8e5 + 0.9e5 * (rpm / 2000.0) ** 1.5
    cycles_per_s = rpm / 60.0 / 2.0  # four-stroke
    p_fric = fmep_pa * (DISPLACEMENT_L / 1000.0) * cycles_per_s / 1000.0
    p_acc = 2.0 + 1.0 * (rpm / 1000.0)
    return p_fric + p_acc


def synthetic_fuel_rate_gps(rpm, torque_nm):
    """Fuel mass flow in g/s for brake torque >= 0 at the given engine speed."""
    rpm, torque_nm = np.broadcast_arrays(
        np.asarray(rpm, dtype=float), np.asarray(torque_nm, dtype=float)
    )
    torque_nm = np.maximum(torque_nm, 0.0)
    omega = rpm * 2.0 * np.pi / 60.0
    p_brake_kw = torque_nm * omega / 1000.0
    load = np.clip(torque_nm / full_load_torque(rpm), 0.0, 1.15)
    eta_i = (
        PEAK_INDICATED_EFFICIENCY
        * (1.0 - 0.12 * ((rpm - 1300.0) / 800.0) ** 2)
        * (1.0 - 0.10 * ((load - 0.85) / 0.85) ** 2)
    )
    fuel_power_kw = (p_brake_kw + friction_power_kw(rpm)) / eta_i
    return fuel_power_kw / DIESEL_LHV_MJ_PER_KG  # kW / (MJ/kg) = g/s
