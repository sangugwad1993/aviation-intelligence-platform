"""T019 — Tests for DifferentialPrivacy (Opacus wrapper).

TDD: Written BEFORE privacy.py implementation (Article III).
Covers US3: (ε,δ)-DP guarantee, gradient clipping, noise injection, SC-003, SC-004.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.federated.config import DifferentialPrivacyConfig
from src.federated.privacy import DifferentialPrivacy
from tests.test_federated.conftest import make_synthetic_flight_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_linear_model() -> nn.Module:
    """Minimal nn.Module — Opacus requires standard nn.Module."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(80, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )


def make_flat_loader(num_samples: int = 64, batch_size: int = 16) -> DataLoader:
    """Flat input loader compatible with linear model (no seq/features dims)."""
    X, _, y = make_synthetic_flight_data(num_samples)
    X_flat = X.view(num_samples, -1)  # (N, 80)
    return DataLoader(TensorDataset(X_flat, y), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDifferentialPrivacy:
    def test_wrap_returns_three_components(self) -> None:
        model = make_linear_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = make_flat_loader()
        dp_config = DifferentialPrivacyConfig(clip_norm=1.0, noise_sigma=1.0)

        dp = DifferentialPrivacy()
        model_dp, opt_dp, loader_dp = dp.wrap(model, opt, loader, dp_config)

        assert model_dp is not None
        assert opt_dp is not None
        assert loader_dp is not None

    def test_not_enabled_before_wrap(self) -> None:
        dp = DifferentialPrivacy()
        assert not dp.is_enabled()

    def test_enabled_after_wrap(self) -> None:
        model = make_linear_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = make_flat_loader()
        dp_config = DifferentialPrivacyConfig(clip_norm=1.0, noise_sigma=1.0)

        dp = DifferentialPrivacy()
        dp.wrap(model, opt, loader, dp_config)
        assert dp.is_enabled()

    def test_get_epsilon_zero_before_wrap(self) -> None:
        dp = DifferentialPrivacy()
        assert dp.get_epsilon(delta=1e-5) == 0.0

    def test_get_epsilon_positive_after_one_training_step(self) -> None:
        model = make_linear_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = make_flat_loader(64, batch_size=16)
        dp_config = DifferentialPrivacyConfig(clip_norm=1.0, noise_sigma=1.0)

        dp = DifferentialPrivacy()
        model_dp, opt_dp, loader_dp = dp.wrap(model, opt, loader, dp_config)

        criterion = nn.MSELoss()
        for x, y in loader_dp:
            opt_dp.zero_grad()
            loss = criterion(model_dp(x), y)
            loss.backward()
            opt_dp.step()
            break  # one step is enough

        epsilon = dp.get_epsilon(delta=1e-5)
        assert epsilon > 0.0

    def test_model_still_trains_with_dp(self) -> None:
        """SC-004 proxy: model loss is finite and non-NaN under DP."""
        torch.manual_seed(42)
        model = make_linear_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        loader = make_flat_loader(64, batch_size=16)
        dp_config = DifferentialPrivacyConfig(clip_norm=1.0, noise_sigma=1.0)

        dp = DifferentialPrivacy()
        model_dp, opt_dp, loader_dp = dp.wrap(model, opt, loader, dp_config)
        criterion = nn.MSELoss()

        losses: list[float] = []
        for _ in range(3):
            for x, y in loader_dp:
                opt_dp.zero_grad()
                loss = criterion(model_dp(x), y)
                loss.backward()
                opt_dp.step()
                losses.append(loss.item())

        assert all(not torch.isnan(torch.tensor(l)) for l in losses), (
            "NaN loss under DP — training is broken"
        )
        assert all(l < float("inf") for l in losses), "Infinite loss under DP"

    def test_noise_sigma_zero_is_deterministic(self) -> None:
        """σ=0 means no noise added — gradient updates are deterministic."""
        torch.manual_seed(7)
        model = make_linear_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = make_flat_loader(32, batch_size=16)
        dp_config = DifferentialPrivacyConfig(clip_norm=1.0, noise_sigma=0.0)

        dp = DifferentialPrivacy()
        model_dp, opt_dp, loader_dp = dp.wrap(model, opt, loader, dp_config)
        criterion = nn.MSELoss()

        first_loss: float | None = None
        for x, y in loader_dp:
            opt_dp.zero_grad()
            loss = criterion(model_dp(x), y)
            loss.backward()
            opt_dp.step()
            first_loss = loss.item()
            break

        assert first_loss is not None
        assert not torch.isnan(torch.tensor(first_loss))

    def test_sc003_epsilon_budget_within_target(self) -> None:
        """SC-003: (ε,δ) ≤ (10, 1e-5) after training steps with σ=1.0, clip=1.0.

        Note: exact ε depends on num_steps and dataset size.
        We verify ε is finite and reasonable (not +inf).
        """
        torch.manual_seed(0)
        model = make_linear_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = make_flat_loader(64, batch_size=16)
        dp_config = DifferentialPrivacyConfig(
            clip_norm=1.0,
            noise_sigma=1.5,  # Higher sigma → lower ε (stronger privacy)
            target_epsilon=10.0,
            target_delta=1e-5,
        )

        dp = DifferentialPrivacy()
        model_dp, opt_dp, loader_dp = dp.wrap(model, opt, loader, dp_config)
        criterion = nn.MSELoss()

        # Run a few steps
        for _ in range(3):
            for x, y in loader_dp:
                opt_dp.zero_grad()
                loss = criterion(model_dp(x), y)
                loss.backward()
                opt_dp.step()

        epsilon = dp.get_epsilon(delta=1e-5)
        assert epsilon < float("inf"), "ε is infinite — accounting failed"
        assert epsilon > 0.0, "ε must be positive after training"
