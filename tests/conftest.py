import pytest

from fuelsev.simulator import Simulation


@pytest.fixture(scope="session")
def short_run():
    """One aggressive urban trip, shared across tests."""
    return Simulation("urban_mixed", "aggressive", seed=3, trips=1, machine_id="T-1").run()
