"""T008 + T014 — Tests for FedAvg, FedProx, and communication utilities.

TDD: Written BEFORE aggregation.py (Article III).
Covers US1 (FedAvg) and US2 (FedProx).
"""
from __future__ import annotations

import pytest
import torch

from src.federated.config import ClientUpdate
from src.federated.aggregation import compute_communication_bytes, fedavg, fedprox_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_update(
    client_id: str,
    weights: list[torch.Tensor],
    num_samples: int,
    loss: float = 0.5,
) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id,
        weight_delta=weights,
        num_samples=num_samples,
        round_loss=loss,
    )


# ---------------------------------------------------------------------------
# T008: FedAvg tests (US1)
# ---------------------------------------------------------------------------


class TestFedAvg:
    def test_equal_samples_is_arithmetic_mean(self) -> None:
        """FedAvg with equal num_samples → simple arithmetic mean."""
        w1 = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
        w2 = [torch.tensor([3.0, 4.0]), torch.tensor([5.0])]
        updates = [
            make_update("c0", w1, num_samples=100),
            make_update("c1", w2, num_samples=100),
        ]
        result = fedavg(updates)
        assert torch.allclose(result[0], torch.tensor([2.0, 3.0]))
        assert torch.allclose(result[1], torch.tensor([4.0]))

    def test_weighted_by_num_samples(self) -> None:
        """FedAvg weights contribution proportionally by num_samples."""
        w1 = [torch.tensor([0.0])]
        w2 = [torch.tensor([4.0])]
        updates = [
            make_update("c0", w1, num_samples=100),
            make_update("c1", w2, num_samples=300),
        ]
        result = fedavg(updates)
        # Expected: (0*100 + 4*300) / 400 = 3.0
        assert torch.allclose(result[0], torch.tensor([3.0]))

    def test_single_update_returns_unchanged(self) -> None:
        """FedAvg with a single update returns that update's weights unchanged."""
        w = [torch.tensor([1.0, 2.0, 3.0])]
        updates = [make_update("c0", w, num_samples=50)]
        result = fedavg(updates)
        assert torch.allclose(result[0], w[0])

    def test_three_clients_weighted(self) -> None:
        """FedAvg correctly weights three clients with different sample counts."""
        w1 = [torch.tensor([6.0])]
        w2 = [torch.tensor([3.0])]
        w3 = [torch.tensor([0.0])]
        updates = [
            make_update("c0", w1, num_samples=200),  # 200/600 = 1/3
            make_update("c1", w2, num_samples=200),  # 1/3
            make_update("c2", w3, num_samples=200),  # 1/3
        ]
        result = fedavg(updates)
        # Expected: (6 + 3 + 0) / 3 = 3.0
        assert torch.allclose(result[0], torch.tensor([3.0]))

    def test_result_norm_not_amplified(self) -> None:
        """Aggregated norm should not exceed max input norm (no silent scaling)."""
        torch.manual_seed(99)
        w1 = [torch.randn(10)]
        w2 = [torch.randn(10)]
        updates = [
            make_update("c0", w1, num_samples=100),
            make_update("c1", w2, num_samples=100),
        ]
        result = fedavg(updates)
        result_norm = torch.norm(result[0]).item()
        max_input_norm = max(torch.norm(w1[0]).item(), torch.norm(w2[0]).item())
        assert result_norm <= max_input_norm + 1e-5

    def test_multi_layer_weights_preserved(self) -> None:
        """FedAvg handles multiple parameter tensors (multi-layer model)."""
        w1 = [torch.ones(4, 4), torch.zeros(4), torch.ones(2, 4), torch.zeros(2)]
        w2 = [torch.zeros(4, 4), torch.ones(4), torch.zeros(2, 4), torch.ones(2)]
        updates = [
            make_update("c0", w1, num_samples=100),
            make_update("c1", w2, num_samples=100),
        ]
        result = fedavg(updates)
        assert len(result) == 4
        # Equal samples → mean = 0.5 for all
        for r in result:
            assert torch.allclose(r, torch.full_like(r, 0.5))


# ---------------------------------------------------------------------------
# T014: FedProx tests (US2)
# ---------------------------------------------------------------------------


class TestFedProx:
    def test_mu_zero_equals_base_loss(self) -> None:
        """FedProx with µ=0 adds zero proximal term — identical to base loss."""
        base_loss = torch.tensor(0.5)
        local = [torch.tensor([1.0, 2.0])]
        global_ = [torch.tensor([3.0, 4.0])]
        result = fedprox_loss(base_loss, local, global_, mu=0.0)
        assert torch.allclose(result, base_loss)

    def test_mu_positive_increases_loss_when_weights_differ(self) -> None:
        """FedProx with µ>0 increases loss when local ≠ global."""
        base_loss = torch.tensor(0.5)
        local = [torch.tensor([1.0, 2.0])]
        global_ = [torch.tensor([3.0, 4.0])]
        result = fedprox_loss(base_loss, local, global_, mu=0.01)
        assert result > base_loss

    def test_proximal_term_formula_correct(self) -> None:
        """Proximal term = (µ/2) * ||local - global||² matches hand-calculated value."""
        base_loss = torch.tensor(0.0)
        local = [torch.tensor([1.0, 0.0])]
        global_ = [torch.tensor([0.0, 0.0])]
        # ||[1,0] - [0,0]||² = 1.0 → term = (0.01/2) * 1.0 = 0.005
        result = fedprox_loss(base_loss, local, global_, mu=0.01)
        assert torch.allclose(result, torch.tensor(0.005), atol=1e-6)

    def test_identical_weights_zero_proximal_term(self) -> None:
        """If local == global, proximal term is 0 regardless of µ."""
        base_loss = torch.tensor(1.0)
        w = [torch.tensor([1.0, 2.0, 3.0])]
        result = fedprox_loss(base_loss, w, w, mu=100.0)
        assert torch.allclose(result, base_loss, atol=1e-6)

    def test_proximal_term_accumulates_across_layers(self) -> None:
        """Proximal term sums across all parameter tensors."""
        base_loss = torch.tensor(0.0)
        local = [torch.tensor([1.0]), torch.tensor([1.0])]
        global_ = [torch.tensor([0.0]), torch.tensor([0.0])]
        # Each layer: ||[1] - [0]||² = 1.0 → total = 2.0 → term = (0.1/2) * 2.0 = 0.1
        result = fedprox_loss(base_loss, local, global_, mu=0.1)
        assert torch.allclose(result, torch.tensor(0.1), atol=1e-6)


# ---------------------------------------------------------------------------
# Communication bytes
# ---------------------------------------------------------------------------


class TestCommunicationBytes:
    def test_float32_four_bytes_per_element(self) -> None:
        weights = [torch.zeros(10, dtype=torch.float32)]
        assert compute_communication_bytes(weights) == 40  # 10 × 4

    def test_multiple_tensors_sum(self) -> None:
        weights = [torch.zeros(5, dtype=torch.float32), torch.zeros(3, dtype=torch.float32)]
        assert compute_communication_bytes(weights) == 32  # 8 × 4

    def test_empty_weights_zero_bytes(self) -> None:
        assert compute_communication_bytes([]) == 0
