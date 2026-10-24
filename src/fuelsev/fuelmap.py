"""Fuel map: a table of fuel flow against engine speed and torque, with SciPy interpolation."""

from __future__ import annotations

import csv
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .engine import synthetic_fuel_rate_gps

DEFAULT_RPM_GRID = np.arange(600.0, 2100.0 + 1, 50.0)
DEFAULT_TORQUE_GRID = np.arange(0.0, 1650.0 + 1, 50.0)

CSV_FIELDS = ("rpm", "torque_nm", "fuel_gps", "bsfc_gkwh")


class FuelMap:
    """Bilinear interpolation of fuel flow (g/s) over a rectangular (rpm, torque) grid.

    Inputs outside the grid are clamped to its edge, so the map never extrapolates.
    """

    def __init__(self, rpm: np.ndarray, torque_nm: np.ndarray, fuel_gps: np.ndarray):
        rpm = np.asarray(rpm, dtype=float)
        torque_nm = np.asarray(torque_nm, dtype=float)
        fuel_gps = np.asarray(fuel_gps, dtype=float)
        if fuel_gps.shape != (rpm.size, torque_nm.size):
            raise ValueError(
                f"fuel grid shape {fuel_gps.shape} does not match axes "
                f"({rpm.size}, {torque_nm.size})"
            )
        if np.any(np.diff(rpm) <= 0) or np.any(np.diff(torque_nm) <= 0):
            raise ValueError("rpm and torque axes must be strictly increasing")
        if not np.all(np.isfinite(fuel_gps)) or np.any(fuel_gps < 0):
            raise ValueError("fuel flow must be finite and non-negative")
        self.rpm = rpm
        self.torque_nm = torque_nm
        self.fuel_gps = fuel_gps
        self._interp = RegularGridInterpolator((rpm, torque_nm), fuel_gps, method="linear")

    # -- lookup -------------------------------------------------------------
    def rate_gps(self, rpm, torque_nm):
        """Fuel flow in g/s. Accepts scalars or arrays (broadcast together)."""
        r, t = np.broadcast_arrays(np.asarray(rpm, dtype=float), np.asarray(torque_nm, dtype=float))
        pts = np.stack(
            [
                np.clip(r, self.rpm[0], self.rpm[-1]),
                np.clip(t, self.torque_nm[0], self.torque_nm[-1]),
            ],
            axis=-1,
        )
        out = self._interp(pts.reshape(-1, 2)).reshape(r.shape)
        return float(out) if out.ndim == 0 else out

    def bsfc_gkwh(self, rpm, torque_nm):
        """Brake specific fuel consumption in g/kWh (NaN at zero torque)."""
        r, t = np.broadcast_arrays(np.asarray(rpm, dtype=float), np.asarray(torque_nm, dtype=float))
        power_kw = t * r * 2.0 * np.pi / 60.0 / 1000.0
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(power_kw > 0, self.rate_gps(r, t) * 3600.0 / power_kw, np.nan)
        return float(out) if out.ndim == 0 else out

    # -- I/O ----------------------------------------------------------------
    @classmethod
    def from_csv(cls, path: str | Path) -> FuelMap:
        """Load a long-format CSV with columns rpm, torque_nm, fuel_gps (bsfc_gkwh is optional)."""
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = {"rpm", "torque_nm", "fuel_gps"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"fuel map CSV is missing columns: {sorted(missing)}")
            rows = [(float(r["rpm"]), float(r["torque_nm"]), float(r["fuel_gps"])) for r in reader]
        if not rows:
            raise ValueError("fuel map CSV has no data rows")
        data = np.array(rows)
        rpm = np.unique(data[:, 0])
        torque = np.unique(data[:, 1])
        if data.shape[0] != rpm.size * torque.size:
            raise ValueError("fuel map CSV must contain a full rpm x torque grid")
        grid = np.full((rpm.size, torque.size), np.nan)
        ri = np.searchsorted(rpm, data[:, 0])
        ti = np.searchsorted(torque, data[:, 1])
        grid[ri, ti] = data[:, 2]
        if np.isnan(grid).any():
            raise ValueError("fuel map CSV has duplicate or missing grid points")
        return cls(rpm, torque, grid)

    def to_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_FIELDS)
            for i, r in enumerate(self.rpm):
                for j, t in enumerate(self.torque_nm):
                    bsfc = self.bsfc_gkwh(r, t)
                    writer.writerow(
                        [
                            f"{r:.0f}",
                            f"{t:.0f}",
                            f"{self.fuel_gps[i, j]:.5f}",
                            "" if np.isnan(bsfc) else f"{bsfc:.1f}",
                        ]
                    )


def build_synthetic_fuel_map() -> FuelMap:
    """Evaluate the synthetic engine model on the default grid."""
    grid = synthetic_fuel_rate_gps(DEFAULT_RPM_GRID[:, None], DEFAULT_TORQUE_GRID[None, :])
    return FuelMap(DEFAULT_RPM_GRID, DEFAULT_TORQUE_GRID, grid)


@lru_cache(maxsize=1)
def load_default_fuel_map() -> FuelMap:
    """Load the fuel map shipped with the package (``data/fuel_map.csv``)."""
    ref = resources.files("fuelsev").joinpath("data/fuel_map.csv")
    with resources.as_file(ref) as path:
        return FuelMap.from_csv(path)
