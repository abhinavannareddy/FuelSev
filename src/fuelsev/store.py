"""SQLite storage for the supervisor API.

One file, no server to run. Every query is parameterised. Telemetry is down-sampled on the
way in (the machine's cumulative counters carry the totals, so reports stay exact even
though most samples are not stored).
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .metrics import aggregate_trips
from .schemas import Alert, Status, Telemetry, TripSummary

SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
    machine_id TEXT PRIMARY KEY,
    online INTEGER NOT NULL DEFAULT 0,
    status_ts REAL,
    last_seen REAL
);
CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL,
    ts REAL NOT NULL,
    sim_t REAL, trip_no INTEGER, route_id TEXT, leg TEXT, pos_m REAL,
    speed_kmh REAL, gear INTEGER, rpm REAL, payload_t REAL, fuel_lph REAL,
    fuel_total_l REAL, fuel_level_l REAL, odometer_km REAL, engine_s REAL,
    idle_s REAL, idle_fuel_l REAL, coolant_c REAL, brake_temp_c REAL, oil_kpa REAL
);
CREATE INDEX IF NOT EXISTS telemetry_machine_ts ON telemetry(machine_id, ts);
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    machine_id TEXT NOT NULL,
    ts REAL NOT NULL,
    sim_t REAL, code TEXT, severity TEXT, category TEXT, message TEXT,
    route_id TEXT, pos_m REAL, evidence TEXT,
    received_ts REAL,
    acknowledged INTEGER NOT NULL DEFAULT 0,
    ack_by TEXT, ack_ts REAL
);
CREATE INDEX IF NOT EXISTS alerts_machine_ts ON alerts(machine_id, ts);
CREATE TABLE IF NOT EXISTS trips (
    machine_id TEXT NOT NULL,
    trip_no INTEGER NOT NULL,
    route_id TEXT, start_ts REAL, end_ts REAL, duration_s REAL,
    payload_t REAL, distance_km REAL, loaded_distance_km REAL, fuel_l REAL,
    idle_s REAL, idle_fuel_l REAL, harsh_brake_events INTEGER, overspeed_s REAL, eco_score REAL,
    received_ts REAL,
    PRIMARY KEY (machine_id, trip_no)
);
CREATE INDEX IF NOT EXISTS trips_end ON trips(end_ts);
"""


class Store:
    def __init__(self, path: str | Path, store_every_s: float = 5.0, retention_days: float = 14.0):
        self.path = str(path)
        self.store_every_s = store_every_s
        self.retention_days = retention_days
        self._last_stored: dict[str, float] = {}
        self._inserts = 0
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # an in-memory database exists only per connection, so keep one open for it
        self._shared = sqlite3.connect(":memory:", check_same_thread=False) if self.path == ":memory:" else None
        with self._conn() as c:
            c.executescript(SCHEMA)
        self.purge_old()

    @contextmanager
    def _conn(self):
        if self._shared is not None:
            conn = self._shared
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ writes
    def add_telemetry(self, t: Telemetry) -> bool:
        """Store a sample if enough time has passed since the last stored one."""
        last = self._last_stored.get(t.machine_id)
        if last is not None and 0 <= t.sim_t - last < self.store_every_s:
            self._touch(t.machine_id, t.ts)
            return False
        self._last_stored[t.machine_id] = t.sim_t
        with self._conn() as c:
            c.execute(
                "INSERT INTO telemetry (machine_id, ts, sim_t, trip_no, route_id, leg, pos_m, speed_kmh, gear, rpm,"
                " payload_t, fuel_lph, fuel_total_l, fuel_level_l, odometer_km, engine_s, idle_s, idle_fuel_l,"
                " coolant_c, brake_temp_c, oil_kpa) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    t.machine_id, t.ts, t.sim_t, t.trip_no, t.route_id, t.leg, t.pos_m, t.speed_kmh, t.gear, t.rpm,
                    t.payload_t, t.fuel_lph, t.fuel_total_l, t.fuel_level_l, t.odometer_km, t.engine_s, t.idle_s,
                    t.idle_fuel_l, t.coolant_c, t.brake_temp_c, t.oil_kpa,
                ),
            )
            self._upsert_machine(c, t.machine_id, last_seen=t.ts)
        self._inserts += 1
        if self._inserts % 2000 == 0:
            self.purge_old()
        return True

    def _touch(self, machine_id: str, ts: float) -> None:
        with self._conn() as c:
            self._upsert_machine(c, machine_id, last_seen=ts)

    @staticmethod
    def _upsert_machine(c, machine_id: str, last_seen: float | None = None, online: bool | None = None, status_ts: float | None = None):
        c.execute("INSERT OR IGNORE INTO machines (machine_id) VALUES (?)", (machine_id,))
        if last_seen is not None:
            c.execute("UPDATE machines SET last_seen = MAX(COALESCE(last_seen, 0), ?) WHERE machine_id = ?", (last_seen, machine_id))
        if online is not None:
            c.execute("UPDATE machines SET online = ?, status_ts = ? WHERE machine_id = ?", (int(online), status_ts, machine_id))

    def add_alert(self, a: Alert) -> bool:
        """Insert an alert. Returns False if this alert id is already stored (a QoS 1 redelivery)."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO alerts (id, machine_id, ts, sim_t, code, severity, category, message, route_id,"
                " pos_m, evidence, received_ts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (a.id, a.machine_id, a.ts, a.sim_t, a.code, a.severity, a.category, a.message, a.route_id, a.pos_m,
                 json.dumps(a.evidence), time.time()),
            )
            self._upsert_machine(c, a.machine_id)
            return cur.rowcount == 1

    def add_trip(self, t: TripSummary) -> bool:
        with self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO trips (machine_id, trip_no, route_id, start_ts, end_ts, duration_s, payload_t,"
                " distance_km, loaded_distance_km, fuel_l, idle_s, idle_fuel_l, harsh_brake_events, overspeed_s,"
                " eco_score, received_ts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (t.machine_id, t.trip_no, t.route_id, t.start_ts, t.end_ts, t.duration_s, t.payload_t, t.distance_km,
                 t.loaded_distance_km, t.fuel_l, t.idle_s, t.idle_fuel_l, t.harsh_brake_events, t.overspeed_s,
                 t.eco_score, time.time()),
            )
            self._upsert_machine(c, t.machine_id)
            return cur.rowcount == 1

    def set_status(self, s: Status) -> None:
        with self._conn() as c:
            self._upsert_machine(c, s.machine_id, online=s.online, status_ts=s.ts)

    def acknowledge(self, alert_id: str, by: str) -> bool:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE alerts SET acknowledged = 1, ack_by = ?, ack_ts = ? WHERE id = ? AND acknowledged = 0",
                (by, time.time(), alert_id),
            )
            return cur.rowcount == 1

    def purge_old(self) -> int:
        cutoff = time.time() - self.retention_days * 86400.0
        with self._conn() as c:
            return c.execute("DELETE FROM telemetry WHERE ts < ?", (cutoff,)).rowcount

    # ------------------------------------------------------------------ reads
    def machines(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT m.machine_id, m.online, m.status_ts, m.last_seen,"
                " (SELECT COUNT(*) FROM alerts a WHERE a.machine_id = m.machine_id AND a.acknowledged = 0) AS open_alerts,"
                " (SELECT COUNT(*) FROM alerts a WHERE a.machine_id = m.machine_id AND a.acknowledged = 0"
                "  AND a.severity = 'critical') AS open_critical"
                " FROM machines m ORDER BY m.machine_id"
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_telemetry(self, machine_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM telemetry WHERE machine_id = ? ORDER BY ts DESC, id DESC LIMIT 1", (machine_id,)
            ).fetchone()
        return dict(row) if row else None

    def alerts(
        self,
        machine_id: str | None = None,
        severity: str | None = None,
        code: str | None = None,
        acknowledged: bool | None = None,
        since: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        where, args = [], []
        for col, val in (("machine_id", machine_id), ("severity", severity), ("code", code)):
            if val is not None:
                where.append(f"{col} = ?")
                args.append(val)
        if acknowledged is not None:
            where.append("acknowledged = ?")
            args.append(int(acknowledged))
        if since is not None:
            where.append("ts >= ?")
            args.append(since)
        sql = "SELECT * FROM alerts" + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY ts DESC, id LIMIT ? OFFSET ?"
        with self._conn() as c:
            rows = c.execute(sql, (*args, limit, offset)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["evidence"] = json.loads(d["evidence"] or "{}")
            d["acknowledged"] = bool(d["acknowledged"])
            out.append(d)
        return out

    def trips(self, machine_id: str | None = None, since: float | None = None, until: float | None = None, limit: int = 100) -> list[dict]:
        where, args = [], []
        if machine_id:
            where.append("machine_id = ?")
            args.append(machine_id)
        if since is not None:
            where.append("end_ts >= ?")
            args.append(since)
        if until is not None:
            where.append("end_ts <= ?")
            args.append(until)
        sql = "SELECT * FROM trips" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY end_ts DESC LIMIT ?"
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, (*args, limit)).fetchall()]

    def _counter_delta(self, c, machine_id: str, column: str, since: float, until: float) -> float:
        """Total increase of a cumulative counter over a window, ignoring resets (machine restarts)."""
        row = c.execute(
            f"SELECT COALESCE(SUM(CASE WHEN d > 0 THEN d ELSE 0 END), 0) AS total FROM ("
            f" SELECT {column} - LAG({column}) OVER (ORDER BY ts, id) AS d FROM telemetry"
            f" WHERE machine_id = ? AND ts BETWEEN ? AND ?)",
            (machine_id, since, until),
        ).fetchone()
        return float(row["total"])

    def report(self, since: float, until: float, machine_id: str | None = None) -> dict:
        with self._conn() as c:
            ids = [machine_id] if machine_id else [r["machine_id"] for r in c.execute("SELECT machine_id FROM machines ORDER BY machine_id")]
            per_machine = []
            for mid in ids:
                fuel = self._counter_delta(c, mid, "fuel_total_l", since, until)
                dist = self._counter_delta(c, mid, "odometer_km", since, until)
                idle_s = self._counter_delta(c, mid, "idle_s", since, until)
                engine_s = self._counter_delta(c, mid, "engine_s", since, until)
                idle_fuel = self._counter_delta(c, mid, "idle_fuel_l", since, until)
                trip_rows = c.execute(
                    "SELECT * FROM trips WHERE machine_id = ? AND end_ts BETWEEN ? AND ?", (mid, since, until)
                ).fetchall()
                alert_rows = c.execute(
                    "SELECT code, severity, COUNT(*) AS n FROM alerts WHERE machine_id = ? AND ts BETWEEN ? AND ?"
                    " GROUP BY code, severity",
                    (mid, since, until),
                ).fetchall()
                trips = [TripSummary(**{k: r[k] for k in TripSummary.model_fields if k in r.keys()}) for r in trip_rows]
                agg = aggregate_trips(trips)
                from .metrics import co2e_kg, fuel_cost_inr

                per_machine.append(
                    {
                        "machine_id": mid,
                        "window": {
                            "fuel_l": round(fuel, 2),
                            "distance_km": round(dist, 2),
                            "l_per_100km": round(100 * fuel / dist, 2) if dist > 0 else None,
                            "co2e_kg": round(co2e_kg(fuel), 2),
                            "fuel_cost_inr": round(fuel_cost_inr(fuel), 2),
                            "engine_hours": round(engine_s / 3600.0, 3),
                            "idle_hours": round(idle_s / 3600.0, 3),
                            "idle_share": round(idle_s / engine_s, 3) if engine_s > 0 else None,
                            "idle_fuel_l": round(idle_fuel, 2),
                        },
                        "trips": agg,
                        "alerts": {f"{r['code']}:{r['severity']}": r["n"] for r in alert_rows},
                    }
                )
        fleet_trips = [t for m in per_machine for t in [m["trips"]]]
        fleet = {
            "fuel_l": round(sum(m["window"]["fuel_l"] for m in per_machine), 2),
            "distance_km": round(sum(m["window"]["distance_km"] for m in per_machine), 2),
            "co2e_kg": round(sum(m["window"]["co2e_kg"] for m in per_machine), 2),
            "trips": sum(t["trips"] for t in fleet_trips),
            "payload_t": round(sum(t["payload_t"] for t in fleet_trips), 2),
        }
        trip_fuel = sum(t["fuel_l"] for t in fleet_trips)
        fleet["l_per_tonne"] = round(trip_fuel / fleet["payload_t"], 3) if fleet["payload_t"] > 0 else None
        fleet["co2e_kg_per_tonne"] = (
            round(co2e_kg(trip_fuel) / fleet["payload_t"], 3) if fleet["payload_t"] > 0 else None
        )
        return {"since": since, "until": until, "fleet": fleet, "machines": per_machine}
