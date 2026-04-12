"""T010 + T015 — Tests for FederatedServer.

TDD: Written BEFORE server.py implementation (Article III).
Covers US1 (FedAvg rounds, quorum) and US2 (FedProx mode).
"""
from __future__ import annotations

import pytest
import torch

from src.federated.config import (
    ClientUpdate,
    FederatedConfig,
    FederationStalledError,
    InsufficientClientsError,
)
from src.federated.server import FederatedServer
from tests.test_federated.conftest import SimpleLocalModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_update(
    client_id: str,
    num_samples: int = 100,
    loss: float = 1.0,
    weight_value: float | None = None,
) -> ClientUpdate:
    model = SimpleLocalModel()
    weights = model.get_weights()
    if weight_value is not None:
        weights = [torch.full_like(w, weight_value) for w in weights]
    return ClientUpdate(
        client_id=client_id,
        weight_delta=weights,
        num_samples=num_samples,
        round_loss=loss,
    )


@pytest.fixture
def basic_config() -> FederatedConfig:
    return FederatedConfig(
        num_rounds=3,
        clients_per_round=3,
        min_clients_per_round=2,
        local_epochs=1,
    )


@pytest.fixture
def server(basic_config: FederatedConfig) -> FederatedServer:
    return FederatedServer(SimpleLocalModel(), basic_config)


# ---------------------------------------------------------------------------
# Tests — US1 (FedAvg, quorum)
# ---------------------------------------------------------------------------


class TestFederatedServerRound:
    def test_run_round_returns_result(self, server: FederatedServer) -> None:
        updates = [make_update("c0"), make_update("c1"), make_update("c2")]
        result = server.run_round(updates)
        assert result.round_num == 1
        assert result.global_loss > 0.0
        assert len(result.participating_clients) == 3

    def test_round_num_increments(self, server: FederatedServer) -> None:
        for expected in [1, 2, 3]:
            updates = [make_update("c0"), make_update("c1")]
            result = server.run_round(updates)
            assert result.round_num == expected

    def test_communication_bytes_positive(self, server: FederatedServer) -> None:
        updates = [make_update("c0"), make_update("c1")]
        result = server.run_round(updates)
        assert result.communication_bytes > 0

    def test_participating_clients_recorded(self, server: FederatedServer) -> None:
        updates = [make_update("airline_a"), make_update("airline_b")]
        result = server.run_round(updates)
        assert "airline_a" in result.participating_clients
        assert "airline_b" in result.participating_clients

    def test_global_weights_change_after_round(self, server: FederatedServer) -> None:
        before = [w.clone() for w in server.get_global_weights()]
        # Force large weight values so aggregation visibly changes the model
        updates = [
            make_update("c0", weight_value=9.0),
            make_update("c1", weight_value=9.0),
        ]
        server.run_round(updates)
        after = server.get_global_weights()
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert changed, "Global model weights must update after aggregation"

    def test_convergence_delta_positive_when_weights_change(
        self, server: FederatedServer
    ) -> None:
        updates = [make_update("c0", weight_value=5.0), make_update("c1", weight_value=5.0)]
        result = server.run_round(updates)
        assert result.convergence_delta >= 0.0


class TestQuorumEnforcement:
    def test_insufficient_clients_raises_error(self, server: FederatedServer) -> None:
        updates = [make_update("c0")]  # Only 1, need min 2
        with pytest.raises(InsufficientClientsError):
            server.run_round(updates)

    def test_empty_updates_raises_insufficient(self, server: FederatedServer) -> None:
        with pytest.raises(InsufficientClientsError):
            server.run_round([])

    def test_three_consecutive_insufficient_raises_stalled(
        self, server: FederatedServer
    ) -> None:
        too_few = [make_update("c0")]
        for _ in range(2):
            with pytest.raises(InsufficientClientsError):
                server.run_round(too_few)
        with pytest.raises(FederationStalledError):
            server.run_round(too_few)

    def test_sufficient_clients_resets_stalled_counter(
        self, server: FederatedServer
    ) -> None:
        too_few = [make_update("c0")]
        with pytest.raises(InsufficientClientsError):
            server.run_round(too_few)

        # Recover with quorum
        server.run_round([make_update("c0"), make_update("c1")])

        # Counter reset: one more insufficient should be InsufficientClients, not Stalled
        with pytest.raises(InsufficientClientsError):
            server.run_round(too_few)

    def test_exactly_min_clients_succeeds(self, server: FederatedServer) -> None:
        updates = [make_update("c0"), make_update("c1")]  # Exactly min_clients_per_round=2
        result = server.run_round(updates)
        assert result is not None


# ---------------------------------------------------------------------------
# T015: FedProx mode tests (US2)
# ---------------------------------------------------------------------------


class TestFedProxMode:
    def test_fedprox_config_runs_without_error(self) -> None:
        config = FederatedConfig(
            num_rounds=2,
            clients_per_round=3,
            min_clients_per_round=2,
            local_epochs=1,
            aggregation="fedprox",
            mu=0.01,
        )
        server = FederatedServer(SimpleLocalModel(), config)
        updates = [make_update("c0"), make_update("c1"), make_update("c2")]
        result = server.run_round(updates)
        assert result.round_num == 1

    def test_fedprox_mu_zero_same_result_as_fedavg(self) -> None:
        """FedProx with µ=0 must produce same aggregated weights as FedAvg."""
        # Build weight deltas matching SimpleLocalModel parameter shapes
        torch.manual_seed(99)
        ref_model = SimpleLocalModel()
        shapes = [p.shape for p in ref_model.get_weights()]
        weights_a = [torch.randn(s) * 0.1 for s in shapes]
        weights_b = [torch.randn(s) * 0.1 for s in shapes]

        config_avg = FederatedConfig(
            num_rounds=1, clients_per_round=2, min_clients_per_round=2,
            local_epochs=1, aggregation="fedavg",
        )
        config_prox = FederatedConfig(
            num_rounds=1, clients_per_round=2, min_clients_per_round=2,
            local_epochs=1, aggregation="fedprox", mu=0.0,
        )

        model_avg = SimpleLocalModel()
        model_prox = SimpleLocalModel()
        # Set identical starting weights
        model_avg.set_weights([torch.zeros_like(w) for w in model_avg.get_weights()])
        model_prox.set_weights([torch.zeros_like(w) for w in model_prox.get_weights()])

        server_avg = FederatedServer(model_avg, config_avg)
        server_prox = FederatedServer(model_prox, config_prox)

        updates = [
            ClientUpdate("c0", weights_a, 100, 0.5),
            ClientUpdate("c1", weights_b, 100, 0.5),
        ]

        server_avg.run_round(updates)
        server_prox.run_round(updates)

        for w_avg, w_prox in zip(
            server_avg.get_global_weights(), server_prox.get_global_weights()
        ):
            assert torch.allclose(w_avg, w_prox, atol=1e-5), (
                "FedProx with µ=0 must be identical to FedAvg"
            )
