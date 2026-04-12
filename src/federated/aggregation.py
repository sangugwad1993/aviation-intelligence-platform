"""FedAvg and FedProx aggregation functions (pure functions — no state).

T011 + T016: US1 (FedAvg) and US2 (FedProx proximal term).
Constitution Article V (Simplicity): pure functions, no classes, independently testable.
"""
from __future__ import annotations

import torch

from .config import ClientUpdate


def fedavg(updates: list[ClientUpdate]) -> list[torch.Tensor]:
    """FedAvg aggregation — McMahan et al. (2017).

    Computes a weighted average of client weight deltas, weighted by num_samples.
    Larger datasets contribute proportionally more to the global update.

    Args:
        updates: Non-empty list of ClientUpdate from this round's participating clients.

    Returns:
        List of aggregated weight tensors matching the shape of each client's weight_delta.

    Raises:
        ValueError: If updates list is empty or total samples is zero.
    """
    if not updates:
        raise ValueError("fedavg requires at least one ClientUpdate")

    total_samples = sum(u.num_samples for u in updates)
    if total_samples == 0:
        raise ValueError("Total samples across all clients is zero — cannot aggregate")

    num_layers = len(updates[0].weight_delta)
    result: list[torch.Tensor] = []

    for layer_idx in range(num_layers):
        weighted_sum = sum(
            u.weight_delta[layer_idx].float() * (u.num_samples / total_samples)
            for u in updates
        )
        result.append(weighted_sum)

    return result


def fedprox_loss(
    criterion_loss: torch.Tensor,
    local_params: list[torch.Tensor],
    global_params: list[torch.Tensor],
    mu: float,
) -> torch.Tensor:
    """FedProx proximal term — Li et al. (2020).

    Adds (µ/2) * ||w_local - w_global||² to the standard loss to penalise
    deviation from the global model during local training. Stabilises convergence
    on non-IID data (Dirichlet α=0.1 partitions).

    When µ=0.0, this is mathematically identical to FedAvg (backward compat).

    Args:
        criterion_loss: Base task loss (e.g. MSELoss output).
        local_params: Current local model parameter tensors.
        global_params: Global model parameter tensors received from server.
        mu: Proximal coefficient. 0.01 recommended per Li et al. (2020).

    Returns:
        Modified loss with proximal term added.
    """
    if mu == 0.0:
        return criterion_loss

    proximal_term: torch.Tensor = sum(  # type: ignore[assignment]
        torch.norm(lp.float() - gp.float()) ** 2
        for lp, gp in zip(local_params, global_params)
    )
    return criterion_loss + (mu / 2.0) * proximal_term


def fedprox(updates: list[ClientUpdate], mu: float) -> list[torch.Tensor]:
    """FedProx server-side aggregation (same as FedAvg — proximal term is client-side).

    The proximal term is added to each client's local loss during training.
    Server-side aggregation remains weighted averaging (same as FedAvg).
    µ=0.0 produces identical results to fedavg().

    Args:
        updates: Client updates collected after FedProx local training.
        mu: Proximal coefficient (used for documentation/logging only here).

    Returns:
        List of aggregated weight tensors (same as FedAvg).
    """
    return fedavg(updates)


def compute_communication_bytes(weights: list[torch.Tensor]) -> int:
    """Compute the total byte size of a list of weight tensors.

    Used for SC-008: communication cost per round logged to MLflow.

    Args:
        weights: List of parameter tensors.

    Returns:
        Total bytes = sum of (num_elements × bytes_per_element) per tensor.
    """
    return sum(w.numel() * w.element_size() for w in weights)
