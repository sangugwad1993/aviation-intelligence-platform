"""LocalModel Protocol — model-agnostic interface for FL clients.

Any model (LiquidNeuralNetwork, TFT, etc.) satisfying this protocol
can be used as a federated local model without modification to FL code.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    pass


@runtime_checkable
class LocalModel(Protocol):
    """Protocol that all FL-compatible models must satisfy.

    Implementations: LNNAdapter, TFTAdapter (ablation only).
    """

    def train_local(self, data_loader: DataLoader, epochs: int) -> float:
        """Train on local data for `epochs` epochs. Returns mean loss."""
        ...

    def get_weights(self) -> list[torch.Tensor]:
        """Return a copy of all trainable parameter tensors."""
        ...

    def set_weights(self, weights: list[torch.Tensor]) -> None:
        """Overwrite all trainable parameter tensors in-place."""
        ...


class LNNAdapter:
    """Wraps LiquidNeuralNetwork to satisfy the LocalModel protocol.

    Primary federated local model per clarify/008-federated-learning/research.md.
    LNN chosen over TFT: cleaner API, 90%+ test coverage, lower communication cost.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._device = device

    def train_local(self, data_loader: DataLoader, epochs: int) -> float:
        self._model.train()
        total_loss, steps = 0.0, 0
        for _ in range(epochs):
            for batch in data_loader:
                x, t, y = batch[0].to(self._device), batch[1].to(self._device), batch[2].to(self._device)
                self._optimizer.zero_grad()
                pred = self._model(x, t)
                loss = self._criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                self._optimizer.step()
                total_loss += loss.item()
                steps += 1
        return total_loss / max(steps, 1)

    def get_weights(self) -> list[torch.Tensor]:
        return [p.data.clone() for p in self._model.parameters()]

    def set_weights(self, weights: list[torch.Tensor]) -> None:
        for param, weight in zip(self._model.parameters(), weights):
            param.data.copy_(weight)
