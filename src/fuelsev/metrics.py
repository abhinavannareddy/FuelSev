"""Sustainability and cost metrics.

Definitions (a "trip" is one haul cycle: load, loaded leg, dump, return empty):

* fuel per tonne      = total trip fuel (loaded leg + return) / payload carried, in L/t.
* fuel per tonne-km   = total trip fuel / (payload x loaded-leg distance), in L/(t.km).
* CO2e per tonne      = fuel per tonne x CO2 factor (kg CO2 per litre of diesel).
* fleet figures       = sums over trips: total fuel / total payload (never an average of ratios).

The CO2 factor is a tailpipe CO2-only default (see ``constants.py`` for the source); CH4 and
N2O are not included, so "CO2e" here is a CO2 figure.
"""

from __future__ import annotations

from collections.abc import Iterable

from .constants import CO2_KG_PER_L_DIESEL, FUEL_PRICE_INR_PER_L
from .schemas import TripSummary


def co2e_kg(fuel_l: float, factor: float = CO2_KG_PER_L_DIESEL) -> float:
    return fuel_l * factor


def fuel_cost_inr(fuel_l: float, price_inr_per_l: float = FUEL_PRICE_INR_PER_L) -> float:
    return fuel_l * price_inr_per_l


def _ratio(num: float, den: float) -> float | None:
    return num / den if den > 0 else None


def eco_score(gear_excess: float, harsh_per_10min: float, overspeed_frac: float, idle_frac: float) -> float:
    """Driver score from 0 to 100. Higher is better.

    Four penalties, each saturating at a stated level:
      * gear_excess      : fuel burned above what the best gear would have used (saturates at 15 %),
                           up to 35 points;
      * harsh_per_10min  : harsh-braking events per 10 minutes driving (saturates at 6), 25 points;
      * overspeed_frac   : share of driving time more than 10 % over the limit (15 %), 20 points;
      * idle_frac        : share of engine-on time spent idling beyond the first minute of each
                           stop, since short waits at tolls and crossings are unavoidable (10 %),
                           20 points.
    """
    score = (
        100.0
        - 35.0 * min(1.0, max(gear_excess, 0.0) / 0.15)
        - 25.0 * min(1.0, max(harsh_per_10min, 0.0) / 6.0)
        - 20.0 * min(1.0, max(overspeed_frac, 0.0) / 0.15)
        - 20.0 * min(1.0, max(idle_frac, 0.0) / 0.10)
    )
    return round(max(0.0, min(100.0, score)), 1)


def trip_kpis(trip: TripSummary, co2_factor: float = CO2_KG_PER_L_DIESEL, price: float = FUEL_PRICE_INR_PER_L) -> dict:
    return {
        "fuel_l": trip.fuel_l,
        "payload_t": trip.payload_t,
        "l_per_tonne": _ratio(trip.fuel_l, trip.payload_t),
        "l_per_tonne_km": _ratio(trip.fuel_l, trip.payload_t * trip.loaded_distance_km),
        "co2e_kg": co2e_kg(trip.fuel_l, co2_factor),
        "co2e_kg_per_tonne": _ratio(co2e_kg(trip.fuel_l, co2_factor), trip.payload_t),
        "fuel_cost_inr": fuel_cost_inr(trip.fuel_l, price),
    }


def aggregate_trips(trips: Iterable[TripSummary], co2_factor: float = CO2_KG_PER_L_DIESEL, price: float = FUEL_PRICE_INR_PER_L) -> dict:
    """Fleet or machine totals over a set of trips."""
    trips = list(trips)
    fuel = sum(t.fuel_l for t in trips)
    payload = sum(t.payload_t for t in trips)
    tkm = sum(t.payload_t * t.loaded_distance_km for t in trips)
    dist = sum(t.distance_km for t in trips)
    co2 = co2e_kg(fuel, co2_factor)
    return {
        "trips": len(trips),
        "fuel_l": round(fuel, 2),
        "distance_km": round(dist, 2),
        "payload_t": round(payload, 2),
        "l_per_tonne": _round(_ratio(fuel, payload), 3),
        "l_per_tonne_km": _round(_ratio(fuel, tkm), 4),
        "co2e_kg": round(co2, 2),
        "co2e_kg_per_tonne": _round(_ratio(co2, payload), 3),
        "fuel_cost_inr": round(fuel_cost_inr(fuel, price), 2),
        "avg_eco_score": _round(sum(t.eco_score for t in trips) / len(trips), 1) if trips else None,
        "harsh_brake_events": sum(t.harsh_brake_events for t in trips),
        "idle_hours": round(sum(t.idle_s for t in trips) / 3600.0, 2),
    }


def _round(x: float | None, n: int) -> float | None:
    return None if x is None else round(x, n)
