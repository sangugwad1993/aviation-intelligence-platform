"""DifferentialPrivacy — Opacus PrivacyEngine wrapper.

T020: US3 — per-sample gradient clipping + Rényi DP accounting.
Research: Abadi et al. (2016) DP-SGD, Mironov (2017) Rényi DP.
Library: Opacus 1.4.0 — peer-reviewed, GPU-compatible, PyTorch-native.

Constitution Article VII (Privacy): provides formal (ε,δ)-DP guarantee.
Constitution Article VIII (Academic Integrity): peer-reviewed implementation.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .config import DifferentialPrivacyConfig

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """Wraps Opacus PrivacyEngine to provide (ε,δ)-differential privacy.

    Usage pattern (per Opacus docs):
        dp = DifferentialPrivacy()
        model, optimizer, loader = dp.wrap(model, optimizer, loader, dp_config)
        # ... train as normal ...
        epsilon = dp.get_epsilon(delta=1e-5)

    The wrapped model uses per-sample gradient clipping (not batch-level),
    which provides tighter DP bounds than naive batch clipping.
    """

    def __init__(self) -> None:
        self._privacy_engine: object | None = None
        self._enabled: bool = False

    def wrap(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        data_loader: DataLoader,
        dp_config: DifferentialPrivacyConfig,
    ) -> tuple[nn.Module, Optimizer, DataLoader]:
        """Attach Opacus PrivacyEngine to model/optimizer/loader.

        After calling wrap(), train using the returned (model, optimizer, loader) —
        do NOT use the originals. Gradient clipping and noise injection happen
        automatically inside the wrapped optimizer's step().

        Args:
            model: PyTorch nn.Module to privatise.
            optimizer: Optimizer associated with the model.
            data_loader: DataLoader with the client's local dataset.
            dp_config: Clipping norm, noise σ, target (ε, δ), accountant type.

        Returns:
            Tuple of (dp_model, dp_optimizer, dp_loader) — use these for training.
        """
        from opacus import PrivacyEngine  # lazy import — not installed in all envs

        privacy_engine = PrivacyEngine()
        model_dp, optimizer_dp, loader_dp = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=dp_config.noise_sigma,
            max_grad_norm=dp_config.clip_norm,
        )

        self._privacy_engine = privacy_engine
        self._enabled = True

        logger.debug(
            "DP enabled: clip_norm=%.2f noise_sigma=%.2f",
            dp_config.clip_norm,
            dp_config.noise_sigma,
        )
        return model_dp, optimizer_dp, loader_dp

    def get_epsilon(self, delta: float) -> float:
        """Compute the current ε privacy budget using Rényi DP accounting.

        Args:
            delta: Target δ value (probability of privacy breach).
                   Typically 1e-5 for datasets with ≥ 10^5 records.

        Returns:
            ε — the privacy budget spent so far. Lower is better.
            Returns 0.0 if DP engine has not been initialised.
        """
        if self._privacy_engine is None:
            return 0.0
        return float(self._privacy_engine.get_epsilon(delta=delta))  # type: ignore[union-attr]

    def is_enabled(self) -> bool:
        """Returns True if wrap() has been called and DP is active."""
        return self._enabled
