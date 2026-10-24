"""End-to-end check without Docker: an in-process amqtt broker, the API ingestor, and a machine
with a simulated dead zone. Verifies that alerts raised during the outage arrive afterwards.

    pip install amqtt && python scripts/mqtt_e2e.py
"""

import asyncio
import os
import tempfile
import threading

from amqtt.broker import Broker

from fuelsev.api import Ingestor
from fuelsev.machine import MachineConfig, run_machine
from fuelsev.mqtt import BrokerConfig
from fuelsev.store import Store

PORT = 18830


async def main() -> int:
    broker = Broker({"listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{PORT}"}},
                     "sys_interval": 0, "auth": {"allow-anonymous": True, "plugins": ["auth_anonymous"]}})
    await broker.start()
    cfg = BrokerConfig(host="127.0.0.1", port=PORT)
    store = Store(os.path.join(tempfile.mkdtemp(), "e2e.db"))
    ing = Ingestor(store, cfg)
    ing.start()
    await asyncio.sleep(1.0)

    mc = MachineConfig(machine_id="HT-E2E", route="urban_mixed", driver="aggressive", trips=1, speedup=0,
                       broker=cfg, outages=[(300.0, 900.0)])
    result = {}
    th = threading.Thread(target=lambda: result.update(run_machine(mc)))
    th.start()
    while th.is_alive():
        await asyncio.sleep(0.5)
    await asyncio.sleep(2.0)

    sent_alerts = result["alerts"]
    stored = store.alerts(machine_id="HT-E2E", limit=1000)
    in_gap = [a for a in stored if 300.0 <= a["sim_t"] < 900.0]
    print("machine:", result)
    print(f"alerts sent {sent_alerts}, stored {len(stored)}, raised during outage {len(in_gap)}")
    print("trips stored:", len(store.trips("HT-E2E")), "| ingest accepted/rejected/dup:", ing.accepted, ing.rejected, ing.duplicates)
    ing.stop()
    await broker.shutdown()
    ok = len(stored) == sent_alerts and sent_alerts > 0 and len(in_gap) > 0 and len(store.trips("HT-E2E")) == 1
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
