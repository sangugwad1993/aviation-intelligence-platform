"""FederatedServer — round orchestration, aggregation dispatch, quorum enforcement.

T013: US1 (FedAvg rounds, quorum, FederationStalledError).
Constitution Article VI (Observability): FederatedRoundResult captures all metrics.
Constitution Article IX (Reproducibility): round results stored for full audit trail.
"""
from __future__ import annotations

import logging

import torch

from .aggregation import compute_communication_bytes, fedavg, fedprox
from .config import (
    ClientUpdate,
    FederatedConfig,
    FederatedRoundResult,
    FederationStalledError,
    InsufficientClientsError,
)
from .protocol import LocalModel

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_SKIPPED = 3


class FederatedServer:
    """Orchestrates federated training rounds.

    Responsibilities:
    - Broadcast global weights to clients
    - Collect and validate ClientUpdates (quorum enforcement)
    - Aggregate via FedAvg or FedProx
    - Track convergence delta and communication bytes per round
    - Raise FederationStalledError after _MAX_CONSECUTIVE_SKIPPED failed rounds
    """

    def __init__(
        self,
        global_model: LocalModel,
        config: FederatedConfig,
    ) -> None:
        self._global_model = global_model
        self._config = config
        self._round_results: list[FederatedRoundResult] = []
        self._consecutive_skipped: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_round(self, client_updates: list[ClientUpdate]) -> FederatedRoundResult:
        """Execute one federation round given a list of collected ClientUpdates.

        Args:
            client_updates: Updates from clients that responded this round.

        Returns:
            FederatedRoundResult with metrics for this round.

        Raises:
            InsufficientClientsError: Fewer than min_clients_per_round responded.
            FederationStalledError: After _MAX_CONSECUTIVE_SKIPPED consecutive failures.
        """
        if len(client_updates) < self._config.min_clients_per_round:
            self._consecutive_skipped += 1
            if self._consecutive_skipped >= _MAX_CONSECUTIVE_SKIPPED:
                raise FederationStalledError(
                    f"Federation stalled: {self._consecutive_skipped} consecutive rounds "
                    f"with < {self._config.min_clients_per_round} clients. "
                    f"Last round had {len(client_updates)} client(s)."
                )
            raise InsufficientClientsError(
                f"Round requires >= {self._config.min_clients_per_round} clients, "
                f"got {len(client_updates)}."
            )

        self._consecutive_skipped = 0

        # Capture old weights for convergence delta calculation
        old_weights = self._global_model.get_weights()

        # Aggregate weight deltas from clients
        if self._config.aggregation == "fedprox":
            aggregated_delta = fedprox(client_updates, self._config.mu)
        else:
            aggregated_delta = fedavg(client_updates)

        # Apply aggregated delta to old weights: w_new = w_old + delta
        new_weights = [
            ow + delta for ow, delta in zip(old_weights, aggregated_delta)
        ]

        # Convergence delta = L2 norm of weight change across all layers
        convergence_delta = float(
            sum(
                torch.norm(delta.float()).item()
                for delta in aggregated_delta
            )
        )

        # Update global model
        self._global_model.set_weights(new_weights)

        # Compute per-round metrics
        total_samples = sum(u.num_samples for u in client_updates)
        global_loss = (
            sum(u.round_loss * u.num_samples for u in client_updates) / total_samples
            if total_samples > 0
            else 0.0
        )
        comm_bytes = compute_communication_bytes(new_weights)

        result = FederatedRoundResult(
            round_num=len(self._round_results) + 1,
            global_loss=global_loss,
            participating_clients=[u.client_id for u in client_updates],
            communication_bytes=comm_bytes,
            convergence_delta=convergence_delta,
        )
        self._round_results.append(result)

        logger.info(
            "round=%d clients=%d loss=%.4f delta=%.4f comm_bytes=%d",
            result.round_num,
            len(client_updates),
            global_loss,
            convergence_delta,
            comm_bytes,
        )
        return result

    def get_global_weights(self) -> list[torch.Tensor]:
        """Return current global model weights."""
        return self._global_model.get_weights()

    def get_round_results(self) -> list[FederatedRoundResult]:
        """Return all round results for MLflow logging or analysis."""
        return list(self._round_results)

    @property
    def global_model(self) -> LocalModel:
        return self._global_model
