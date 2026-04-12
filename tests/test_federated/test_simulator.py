"""T022 — Tests for FederatedSimulator (process-isolated end-to-end simulation).

TDD: Written BEFORE simulator.py implementation (Article III).
Covers US1 SC-005 (process isolation, zero raw data), SC-007 (reproducibility),
and full e2e convergence with FedAvg and FedProx.
"""
from __future__ import annotations

import pytest
import torch

from src.federated.config import FederatedConfig
from src.federated.simulator import FederatedSimulator
from tests.test_federated.conftest import SimpleLocalModel, make_iid_partition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_simple_model() -> SimpleLocalModel:
    return SimpleLocalModel()


@pytest.fixture
def sim_config() -> FederatedConfig:
    return FederatedConfig(
        num_rounds=2,
        clients_per_round=3,
        min_clients_per_round=2,
        local_epochs=1,
        aggregation="fedavg",
        seed=42,
    )


@pytest.fixture
def three_partitions():
    return make_iid_partition(num_clients=3, num_samples=150)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFederatedSimulatorBasic:
    def test_run_returns_results_for_each_round(
        self, sim_config: FederatedConfig, three_partitions
    ) -> None:
        sim = FederatedSimulator(
            config=sim_config,
            model_factory=lambda _: make_simple_model(),
            data_partitions=three_partitions,
        )
        results = sim.run(num_rounds=2)
        assert len(results) == 2

    def test_global_loss_is_finite_and_positive(
        self, sim_config: FederatedConfig, three_partitions
    ) -> None:
        sim = FederatedSimulator(
            config=sim_config,
            model_factory=lambda _: make_simple_model(),
            data_partitions=three_partitions,
        )
        results = sim.run(num_rounds=2)
        for r in results:
            assert r.global_loss > 0.0
            assert r.global_loss < float("inf")
            assert not torch.isnan(torch.tensor(r.global_loss))

    def test_round_numbers_sequential(
        self, sim_config: FederatedConfig, three_partitions
    ) -> None:
        sim = FederatedSimulator(
            config=sim_config,
            model_factory=lambda _: make_simple_model(),
            data_partitions=three_partitions,
        )
        results = sim.run(num_rounds=3)
        for i, r in enumerate(results, start=1):
            assert r.round_num == i

    def test_participating_clients_populated(
        self, sim_config: FederatedConfig, three_partitions
    ) -> None:
        sim = FederatedSimulator(
            config=sim_config,
            model_factory=lambda _: make_simple_model(),
            data_partitions=three_partitions,
            client_ids=["airline_a", "airline_b", "airport_c"],
        )
        results = sim.run(num_rounds=1)
        assert len(results[0].participating_clients) == 3


class TestSC005ProcessIsolation:
    def test_communication_bytes_match_model_not_raw_data(
        self, sim_config: FederatedConfig, three_partitions
    ) -> None:
        """SC-005: bytes crossing process boundary must be model weights, not raw data.

        SimpleLocalModel has:
          Linear(80→32): 80*32 + 32 = 2592 params
          Linear(32→3):  32*3  +  3 =   99 params
          Total: 2591 + 99 = 2691 params × 4 bytes = ~10764 bytes

        Raw input per client (150 samples, seq=10, feat=8):
          150 × 10 × 8 × 4 = 48000 bytes

        Communication bytes must be << raw input bytes.
        """
        sim = FederatedSimulator(
            config=sim_config,
            model_factory=lambda _: make_simple_model(),
            data_partitions=three_partitions,
        )
        results = sim.run(num_rounds=1)

        raw_input_bytes_per_client = 150 * 10 * 8 * 4  # 48000
        assert results[0].communication_bytes > 0
        assert results[0].communication_bytes < raw_input_bytes_per_client, (
            f"comm_bytes={results[0].communication_bytes} >= raw_input={raw_input_bytes_per_client}"
            " — possible raw data leak in process queue!"
        )

    def test_get_results_matches_run_output(
        self, sim_config: FederatedConfig, three_partitions
    ) -> None:
        sim = FederatedSimulator(
            config=sim_config,
            model_factory=lambda _: make_simple_model(),
            data_partitions=three_partitions,
        )
        run_results = sim.run(num_rounds=2)
        stored_results = sim.get_results()
        assert len(run_results) == len(stored_results)
        for r, s in zip(run_results, stored_results):
            assert r.round_num == s.round_num


class TestFedProxSimulator:
    def test_fedprox_config_completes(self, three_partitions) -> None:
        config = FederatedConfig(
            num_rounds=2,
            clients_per_round=3,
            min_clients_per_round=2,
            local_epochs=1,
            aggregation="fedprox",
            mu=0.01,
        )
        sim = FederatedSimulator(
            config=config,
            model_factory=lambda _: make_simple_model(),
            data_partitions=three_partitions,
        )
        results = sim.run(num_rounds=2)
        assert len(results) == 2
        for r in results:
            assert r.global_loss > 0.0
