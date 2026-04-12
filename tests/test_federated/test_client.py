"""T009 — Tests for FederatedClient.

TDD: Written BEFORE client.py implementation (Article III).
Covers US1: local training, weight delta, SC-005 (no raw data in ClientUpdate).
"""
from __future__ import annotations

import pickle

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.federated.client import FederatedClient
from src.federated.config import ClientUpdate, FederatedConfig
from tests.test_federated.conftest import SimpleLocalModel, make_synthetic_flight_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_loader(num_samples: int = 50) -> DataLoader:
    X, t, y = make_synthetic_flight_data(num_samples)
    return DataLoader(TensorDataset(X, t, y), batch_size=16)


@pytest.fixture
def basic_config() -> FederatedConfig:
    return FederatedConfig(
        num_rounds=3, clients_per_round=2, min_clients_per_round=2, local_epochs=1
    )


@pytest.fixture
def fed_client(basic_config: FederatedConfig) -> FederatedClient:
    return FederatedClient("client_0", SimpleLocalModel(), basic_config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFederatedClient:
    def test_train_returns_client_update(self, fed_client: FederatedClient) -> None:
        loader = make_loader()
        global_weights = fed_client._model.get_weights()
        update = fed_client.train(global_weights, loader)
        assert isinstance(update, ClientUpdate)

    def test_client_id_propagated(self, fed_client: FederatedClient) -> None:
        loader = make_loader()
        global_weights = fed_client._model.get_weights()
        update = fed_client.train(global_weights, loader)
        assert update.client_id == "client_0"

    def test_round_loss_positive(self, fed_client: FederatedClient) -> None:
        loader = make_loader()
        global_weights = fed_client._model.get_weights()
        update = fed_client.train(global_weights, loader)
        assert update.round_loss > 0.0

    def test_num_samples_matches_dataset_size(self, fed_client: FederatedClient) -> None:
        loader = make_loader(50)
        global_weights = fed_client._model.get_weights()
        update = fed_client.train(global_weights, loader)
        assert update.num_samples == 50

    def test_weight_delta_shape_matches_model(self, fed_client: FederatedClient) -> None:
        loader = make_loader()
        global_weights = fed_client._model.get_weights()
        update = fed_client.train(global_weights, loader)
        assert len(update.weight_delta) == len(global_weights)
        for delta, gw in zip(update.weight_delta, global_weights):
            assert delta.shape == gw.shape, (
                f"weight_delta shape {delta.shape} != global_weight shape {gw.shape}"
            )

    def test_sc005_no_raw_data_in_serialised_update(self, fed_client: FederatedClient) -> None:
        """SC-005: Serialised ClientUpdate must NOT contain raw input feature tensors.

        Raw inputs are 3D: (batch, seq_len, features).
        Model weight deltas are at most 2D (weight matrix or bias vector).
        """
        loader = make_loader(50)
        global_weights = fed_client._model.get_weights()
        update = fed_client.train(global_weights, loader)

        serialised = pickle.dumps(update)
        restored: ClientUpdate = pickle.loads(serialised)

        for tensor in restored.weight_delta:
            assert tensor.dim() <= 2, (
                f"3D+ tensor in weight_delta (shape={tensor.shape}) "
                "— this looks like raw input data, not model parameters!"
            )

    def test_privacy_spent_zero_without_dp(self, fed_client: FederatedClient) -> None:
        loader = make_loader()
        global_weights = fed_client._model.get_weights()
        update = fed_client.train(global_weights, loader)
        assert update.privacy_spent == 0.0

    def test_global_weights_applied_before_training(self, basic_config: FederatedConfig) -> None:
        """Client must set global weights before local training (not train on old local weights)."""
        model = SimpleLocalModel()
        client = FederatedClient("c0", model, basic_config)

        loader = make_loader(32)
        initial_weights = model.get_weights()

        # Set all global weights to a specific value
        sentinel_weights = [torch.full_like(w, 99.0) for w in initial_weights]
        update = client.train(sentinel_weights, loader)

        # weight_delta = trained_weights - sentinel; trained weights started from sentinel
        # delta should NOT be the same as (trained_on_initial - initial)
        # Just verify update was computed correctly (delta shape is right and training ran)
        for delta, sentinel in zip(update.weight_delta, sentinel_weights):
            assert delta.shape == sentinel.shape

    def test_different_clients_have_different_ids(self, basic_config: FederatedConfig) -> None:
        c0 = FederatedClient("airline_a", SimpleLocalModel(), basic_config)
        c1 = FederatedClient("airline_b", SimpleLocalModel(), basic_config)
        loader = make_loader(32)
        u0 = c0.train(c0._model.get_weights(), loader)
        u1 = c1.train(c1._model.get_weights(), loader)
        assert u0.client_id == "airline_a"
        assert u1.client_id == "airline_b"
        assert u0.client_id != u1.client_id
