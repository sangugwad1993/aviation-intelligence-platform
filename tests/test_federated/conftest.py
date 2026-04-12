"""Shared fixtures for federated learning tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.federated.config import DifferentialPrivacyConfig, FederatedConfig


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def make_synthetic_flight_data(
    num_samples: int = 100, seq_len: int = 10, features: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic data shaped like OpenSky ADS-B data."""
    torch.manual_seed(42)
    X = torch.randn(num_samples, seq_len, features)
    t = torch.linspace(0, 1, seq_len).unsqueeze(0).expand(num_samples, -1)
    y = torch.randn(num_samples, 3)  # lat, lon, alt prediction
    return X, t, y


def make_iid_partition(
    num_clients: int = 3, num_samples: int = 300
) -> list[DataLoader]:
    """Uniform IID split across clients."""
    np.random.seed(42)
    chunk = num_samples // num_clients
    X, t, y = make_synthetic_flight_data(num_samples)
    partitions = []
    for i in range(num_clients):
        start, end = i * chunk, (i + 1) * chunk
        dataset = TensorDataset(X[start:end], t[start:end], y[start:end])
        partitions.append(DataLoader(dataset, batch_size=16, shuffle=False))
    return partitions


def make_non_iid_partition(
    num_clients: int = 3, num_samples: int = 300, alpha: float = 0.1
) -> list[DataLoader]:
    """Dirichlet-based non-IID split — Constitution Article IX: seeded for reproducibility."""
    np.random.seed(42)
    X, t, y = make_synthetic_flight_data(num_samples)
    proportions = np.random.dirichlet([alpha] * num_clients)
    sizes = (proportions * num_samples).astype(int)
    sizes[-1] = num_samples - sizes[:-1].sum()  # ensure exact total

    partitions, idx = [], 0
    for size in sizes:
        size = max(size, 1)  # minimum 1 sample
        dataset = TensorDataset(X[idx : idx + size], t[idx : idx + size], y[idx : idx + size])
        partitions.append(DataLoader(dataset, batch_size=min(16, size), shuffle=False))
        idx += size
    return partitions


# ---------------------------------------------------------------------------
# Minimal LocalModel for testing (satisfies LocalModel protocol without LNN dep)
# ---------------------------------------------------------------------------


class SimpleLocalModel:
    """2-layer linear network for fast test execution."""

    INPUT_DIM = 80  # 10 seq_len * 8 features (flattened)
    OUTPUT_DIM = 3

    def __init__(self) -> None:
        self._model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.INPUT_DIM, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, self.OUTPUT_DIM),
        )
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=0.01)
        self._criterion = torch.nn.MSELoss()

    def train_local(self, data_loader: DataLoader, epochs: int) -> float:
        self._model.train()
        total_loss, steps = 0.0, 0
        for _ in range(epochs):
            for batch in data_loader:
                x = batch[0]  # (batch, seq, features) or (batch, flat)
                y = batch[-1]
                self._optimizer.zero_grad()
                pred = self._model(x)
                loss = self._criterion(pred, y)
                loss.backward()
                self._optimizer.step()
                total_loss += loss.item()
                steps += 1
        return total_loss / max(steps, 1)

    def get_weights(self) -> list[torch.Tensor]:
        return [p.data.clone() for p in self._model.parameters()]

    def set_weights(self, weights: list[torch.Tensor]) -> None:
        for param, weight in zip(self._model.parameters(), weights):
            param.data.copy_(weight)


def make_simple_model() -> SimpleLocalModel:
    return SimpleLocalModel()


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_fed_config() -> FederatedConfig:
    return FederatedConfig(
        num_rounds=3,
        clients_per_round=3,
        min_clients_per_round=2,
        local_epochs=1,
        aggregation="fedavg",
        mu=0.01,
        dp_config=None,
        seed=42,
    )


@pytest.fixture
def dp_config() -> DifferentialPrivacyConfig:
    return DifferentialPrivacyConfig(
        clip_norm=1.0,
        noise_sigma=1.0,
        target_epsilon=10.0,
        target_delta=1e-5,
        accountant="rdp",
    )


@pytest.fixture
def synthetic_flight_data():
    return make_synthetic_flight_data()


@pytest.fixture
def iid_partition() -> list[DataLoader]:
    return make_iid_partition()


@pytest.fixture
def non_iid_partition() -> list[DataLoader]:
    return make_non_iid_partition()


@pytest.fixture
def simple_model() -> SimpleLocalModel:
    return SimpleLocalModel()
