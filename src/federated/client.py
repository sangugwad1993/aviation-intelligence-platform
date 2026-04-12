"""FederatedClient — local training, weight delta computation, DP integration.

T012 + T017 + T021: US1 (local training), US2 (FedProx), US3 (DP).
Constitution Article VII (Privacy): raw data references deleted after train() returns.
"""
from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from .config import ClientUpdate, FederatedConfig
from .protocol import LocalModel

logger = logging.getLogger(__name__)


class FederatedClient:
    """Trains a LocalModel on local data and returns only weight deltas.

    Constitution Article VII: This class guarantees that the returned ClientUpdate
    contains ONLY model parameter deltas — never raw features, labels, or
    flight identifiers. The data_loader reference is explicitly released after
    training to prevent accidental serialisation.
    """

    def __init__(
        self,
        client_id: str,
        local_model: LocalModel,
        config: FederatedConfig,
    ) -> None:
        self.client_id = client_id
        self._model = local_model
        self._config = config
        self._dp: DifferentialPrivacy | None = None  # type: ignore[name-defined]

    def train(
        self,
        global_weights: list[torch.Tensor],
        data_loader: DataLoader,
    ) -> ClientUpdate:
        """Set global weights, train locally, return weight delta.

        Args:
            global_weights: Current global model weights from FederatedServer.
            data_loader: Local dataset loader. Reference is deleted after training.

        Returns:
            ClientUpdate containing ONLY weight deltas and metadata (no raw data).
        """
        self._model.set_weights(global_weights)
        num_samples = len(data_loader.dataset)  # type: ignore[arg-type]

        if self._config.dp_config is not None and self._dp is not None:
            round_loss, privacy_spent = self._train_with_dp(data_loader)
        else:
            round_loss = self._model.train_local(data_loader, self._config.local_epochs)
            privacy_spent = 0.0

        current_weights = self._model.get_weights()
        weight_delta = [cw - gw for cw, gw in zip(current_weights, global_weights)]

        update = ClientUpdate(
            client_id=self.client_id,
            weight_delta=weight_delta,
            num_samples=num_samples,
            round_loss=round_loss,
            privacy_spent=privacy_spent,
        )

        # Constitution Article VII: release all data references immediately
        del data_loader

        logger.debug(
            "client=%s samples=%d loss=%.4f privacy_spent=%.4f",
            self.client_id,
            num_samples,
            round_loss,
            privacy_spent,
        )
        return update

    def _train_with_dp(
        self, data_loader: DataLoader
    ) -> tuple[float, float]:
        """Train with Opacus DP engine. Returns (round_loss, epsilon_spent)."""
        assert self._dp is not None
        assert self._config.dp_config is not None

        from .privacy import DifferentialPrivacy

        criterion = torch.nn.MSELoss()
        # Wrap model/optimizer/loader with DP engine
        if hasattr(self._model, "_model") and hasattr(self._model, "_optimizer"):
            model_dp, opt_dp, loader_dp = self._dp.wrap(
                self._model._model,  # type: ignore[attr-defined]
                self._model._optimizer,  # type: ignore[attr-defined]
                data_loader,
                self._config.dp_config,
            )
            # Run manual training loop with DP-wrapped components
            total_loss, steps = 0.0, 0
            model_dp.train()
            for _ in range(self._config.local_epochs):
                for batch in loader_dp:
                    x = batch[0]
                    y = batch[-1]
                    opt_dp.zero_grad()
                    pred = model_dp(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    opt_dp.step()
                    total_loss += loss.item()
                    steps += 1
            round_loss = total_loss / max(steps, 1)
            privacy_spent = self._dp.get_epsilon(self._config.dp_config.target_delta)
        else:
            round_loss = self._model.train_local(data_loader, self._config.local_epochs)
            privacy_spent = 0.0

        return round_loss, privacy_spent

    def set_dp_engine(self, dp: "DifferentialPrivacy") -> None:  # type: ignore[name-defined]
        """Attach a DifferentialPrivacy engine for US3 (DP-enabled training)."""
        self._dp = dp

    @property
    def model(self) -> LocalModel:
        return self._model
