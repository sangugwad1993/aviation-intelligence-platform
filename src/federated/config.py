"""Data contracts for the Federated Learning infrastructure.

All types defined here are frozen data containers — no FL logic.
Constitution Article IX: seeds and all hyperparams live here for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch


@dataclass
class DifferentialPrivacyConfig:
    """Configuration for Opacus-based differential privacy."""

    clip_norm: float = 1.0
    noise_sigma: float = 1.0
    target_epsilon: float = 10.0
    target_delta: float = 1e-5
    accountant: Literal["rdp", "prv"] = "rdp"


@dataclass
class FederatedConfig:
    """Top-level configuration for a federated training run."""

    num_rounds: int = 10
    clients_per_round: int = 3
    min_clients_per_round: int = 2
    local_epochs: int = 1
    aggregation: Literal["fedavg", "fedprox"] = "fedavg"
    mu: float = 0.01
    dp_config: DifferentialPrivacyConfig | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.min_clients_per_round < 2:
            raise ValueError(
                f"min_clients_per_round must be >= 2, got {self.min_clients_per_round}. "
                "Fewer than 2 clients cannot produce meaningful aggregation."
            )


@dataclass
class ClientUpdate:
    """Weight update returned by a client after local training.

    Constitution Article VII: ONLY model parameter deltas and metadata.
    Raw features, labels, and flight identifiers MUST NOT appear here.
    """

    client_id: str
    weight_delta: list[torch.Tensor]
    num_samples: int
    round_loss: float
    privacy_spent: float = 0.0


@dataclass
class FederatedRoundResult:
    """Aggregated result from one federation round — logged to MLflow."""

    round_num: int
    global_loss: float
    participating_clients: list[str]
    communication_bytes: int
    convergence_delta: float


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InsufficientClientsError(RuntimeError):
    """Raised when fewer clients respond than min_clients_per_round."""


class FederationStalledError(RuntimeError):
    """Raised after 3 consecutive rounds with InsufficientClientsError."""
