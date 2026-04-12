"""T007 — Tests for FederatedConfig, ClientUpdate, and exception types.

TDD: These tests were written BEFORE config.py implementation (Article III).
"""
from __future__ import annotations

import pickle

import pytest
import torch

from src.federated.config import (
    ClientUpdate,
    DifferentialPrivacyConfig,
    FederatedConfig,
    FederatedRoundResult,
    FederationStalledError,
    InsufficientClientsError,
)


class TestFederatedConfig:
    def test_defaults_min_clients(self) -> None:
        config = FederatedConfig(num_rounds=5, clients_per_round=3, local_epochs=1)
        assert config.min_clients_per_round == 2

    def test_defaults_aggregation(self) -> None:
        config = FederatedConfig(num_rounds=5, clients_per_round=3, local_epochs=1)
        assert config.aggregation == "fedavg"

    def test_defaults_dp_config_none(self) -> None:
        config = FederatedConfig(num_rounds=5, clients_per_round=3, local_epochs=1)
        assert config.dp_config is None

    def test_defaults_seed(self) -> None:
        config = FederatedConfig(num_rounds=5, clients_per_round=3, local_epochs=1)
        assert config.seed == 42

    def test_defaults_mu(self) -> None:
        config = FederatedConfig(num_rounds=5, clients_per_round=3, local_epochs=1)
        assert config.mu == 0.01

    def test_raises_on_min_clients_one(self) -> None:
        with pytest.raises(ValueError, match="min_clients_per_round must be >= 2"):
            FederatedConfig(
                num_rounds=5,
                clients_per_round=3,
                local_epochs=1,
                min_clients_per_round=1,
            )

    def test_raises_on_min_clients_zero(self) -> None:
        with pytest.raises(ValueError):
            FederatedConfig(
                num_rounds=5,
                clients_per_round=3,
                local_epochs=1,
                min_clients_per_round=0,
            )

    def test_min_clients_exactly_2_is_valid(self) -> None:
        config = FederatedConfig(
            num_rounds=5,
            clients_per_round=3,
            local_epochs=1,
            min_clients_per_round=2,
        )
        assert config.min_clients_per_round == 2

    def test_fedprox_aggregation_accepted(self) -> None:
        config = FederatedConfig(
            num_rounds=5, clients_per_round=3, local_epochs=1, aggregation="fedprox"
        )
        assert config.aggregation == "fedprox"

    def test_dp_config_embedded(self) -> None:
        dp = DifferentialPrivacyConfig(clip_norm=0.5, noise_sigma=2.0)
        config = FederatedConfig(
            num_rounds=5, clients_per_round=3, local_epochs=1, dp_config=dp
        )
        assert config.dp_config is not None
        assert config.dp_config.clip_norm == 0.5


class TestDifferentialPrivacyConfig:
    def test_defaults(self) -> None:
        dp = DifferentialPrivacyConfig()
        assert dp.clip_norm == 1.0
        assert dp.noise_sigma == 1.0
        assert dp.target_epsilon == 10.0
        assert dp.target_delta == 1e-5
        assert dp.accountant == "rdp"


class TestClientUpdate:
    def test_serialises_and_restores(self) -> None:
        weight_delta = [torch.zeros(10, 5), torch.zeros(5)]
        update = ClientUpdate(
            client_id="client_0",
            weight_delta=weight_delta,
            num_samples=100,
            round_loss=0.5,
        )
        data = pickle.dumps(update)
        restored: ClientUpdate = pickle.loads(data)
        assert restored.client_id == "client_0"
        assert restored.num_samples == 100
        assert abs(restored.round_loss - 0.5) < 1e-6
        assert len(restored.weight_delta) == 2

    def test_sc005_no_raw_input_shape_in_weight_delta(self) -> None:
        """SC-005: weight_delta must NOT contain tensors shaped like raw inputs."""
        weight_delta = [torch.zeros(10, 5), torch.zeros(5)]
        update = ClientUpdate(
            client_id="c0", weight_delta=weight_delta, num_samples=100, round_loss=0.5
        )
        data = pickle.dumps(update)
        restored: ClientUpdate = pickle.loads(data)

        RAW_INPUT_SHAPE = torch.Size([100, 10, 8])  # (samples, seq_len, features)
        for tensor in restored.weight_delta:
            assert tensor.shape != RAW_INPUT_SHAPE, (
                "weight_delta contains tensor with raw input shape — data leak detected!"
            )
            assert tensor.dim() <= 2, (
                f"3D+ tensor in weight_delta (shape={tensor.shape}) suggests raw data leak"
            )

    def test_default_privacy_spent_zero(self) -> None:
        update = ClientUpdate(
            client_id="c0",
            weight_delta=[torch.zeros(5)],
            num_samples=50,
            round_loss=1.0,
        )
        assert update.privacy_spent == 0.0

    def test_privacy_spent_can_be_set(self) -> None:
        update = ClientUpdate(
            client_id="c0",
            weight_delta=[torch.zeros(5)],
            num_samples=50,
            round_loss=1.0,
            privacy_spent=3.14,
        )
        assert abs(update.privacy_spent - 3.14) < 1e-6


class TestExceptions:
    def test_federation_stalled_is_runtime_error(self) -> None:
        err = FederationStalledError("stalled")
        assert isinstance(err, RuntimeError)

    def test_insufficient_clients_is_runtime_error(self) -> None:
        err = InsufficientClientsError("not enough")
        assert isinstance(err, RuntimeError)

    def test_raise_and_catch_stalled(self) -> None:
        with pytest.raises(FederationStalledError, match="stalled"):
            raise FederationStalledError("Federation stalled after 3 rounds")

    def test_insufficient_clients_caught_separately(self) -> None:
        with pytest.raises(InsufficientClientsError):
            raise InsufficientClientsError("only 1 client")
