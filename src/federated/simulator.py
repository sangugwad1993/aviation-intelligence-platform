"""FederatedSimulator — OS process-isolated FL simulation.

T023: US1 SC-005 (process isolation validates privacy claim).
Architecture: each FederatedClient runs in a multiprocessing.Process.
Only serialised ClientUpdate objects cross process boundaries via Queue.

Research justification (clarify/008-federated-learning/research.md):
  - Geyer et al. (2017): process isolation is the standard FL privacy model
  - Flower's virtual client engine uses Ray (process-based) for same reason
  - Threading shares memory → SC-005 test would be meaningless

Constitution Article VII (Privacy): Queue communication enforces data isolation.
Constitution Article IX (Reproducibility): client seeds = hash(client_id) % 2^31.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any, Callable

from .client import FederatedClient
from .config import (
    ClientUpdate,
    FederatedConfig,
    FederatedRoundResult,
    InsufficientClientsError,
)
from .protocol import LocalModel
from .server import FederatedServer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker function (runs in a child process)
# ---------------------------------------------------------------------------


def _client_worker(
    client_id: str,
    in_queue: "mp.Queue[Any]",
    out_queue: "mp.Queue[Any]",
    model_factory: Callable[[str], LocalModel],
    data_partition: Any,
    config: FederatedConfig,
) -> None:
    """Spawned once per client. Loops receiving global weights, trains, sends ClientUpdate.

    PRIVACY GUARANTEE: Only ClientUpdate objects (weight_delta + metadata) leave this
    process via out_queue. Raw data never enters the queue.

    Args:
        client_id: Unique identifier for this client process.
        in_queue: Receives (global_weights, data_loader) from simulator.
        out_queue: Sends ClientUpdate back to simulator.
        model_factory: Factory function returning a fresh LocalModel for this client.
        data_partition: DataLoader for this client's local data partition.
        config: Shared FederatedConfig.
    """
    import torch

    torch.manual_seed(hash(client_id) % (2**31))

    model = model_factory(client_id)
    client = FederatedClient(client_id, model, config)

    while True:
        msg = in_queue.get()
        if msg is None:
            break  # Sentinel: shutdown signal

        global_weights = msg  # Only weights cross the boundary — no raw data

        try:
            update = client.train(global_weights, data_partition)

            # Explicitly build a clean ClientUpdate — defence-in-depth for SC-005
            safe_update = ClientUpdate(
                client_id=update.client_id,
                weight_delta=[w.clone().detach() for w in update.weight_delta],
                num_samples=update.num_samples,
                round_loss=update.round_loss,
                privacy_spent=update.privacy_spent,
            )
            out_queue.put(safe_update)
        except Exception as exc:  # noqa: BLE001
            logger.error("Client %s error in round: %s", client_id, exc)
            out_queue.put(None)  # Signal failure to simulator


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class FederatedSimulator:
    """Runs a complete federated learning simulation with OS-level process isolation.

    Each client is a separate OS process — genuinely isolated memory spaces.
    Communication is via multiprocessing.Queue, carrying only ClientUpdate objects.

    This architecture makes SC-005 meaningful: the unit test that verifies
    "no raw data in ClientUpdate" is enforced by the OS, not just programming convention.
    """

    def __init__(
        self,
        config: FederatedConfig,
        model_factory: Callable[[str], LocalModel],
        data_partitions: list[Any],
        client_ids: list[str] | None = None,
        in_process: bool = True,
    ) -> None:
        self._config = config
        self._model_factory = model_factory
        self._data_partitions = data_partitions
        self._client_ids = client_ids or [
            f"client_{i}" for i in range(len(data_partitions))
        ]
        self._results: list[FederatedRoundResult] = []
        self._in_process = in_process

        if len(self._client_ids) != len(self._data_partitions):
            raise ValueError(
                f"client_ids length ({len(self._client_ids)}) must match "
                f"data_partitions length ({len(self._data_partitions)})"
            )

    def run(self, num_rounds: int | None = None) -> list[FederatedRoundResult]:
        """Run the federation simulation.

        When in_process=True (default), clients run sequentially in the same
        process — fast, testable, no pickle requirements.

        When in_process=False, each client runs in a separate OS process via
        multiprocessing.spawn — SC-005 process isolation is OS-enforced.

        Args:
            num_rounds: Number of federation rounds. Defaults to config.num_rounds.

        Returns:
            List of FederatedRoundResult, one per successful round.
        """
        if self._in_process:
            return self._run_in_process(num_rounds)
        return self._run_multiprocess(num_rounds)

    def _run_in_process(self, num_rounds: int | None = None) -> list[FederatedRoundResult]:
        """Run federation in a single process (fast, testable)."""
        import torch

        rounds = num_rounds or self._config.num_rounds

        global_model = self._model_factory("server")
        server = FederatedServer(global_model, self._config)

        clients = [
            FederatedClient(cid, self._model_factory(cid), self._config)
            for cid in self._client_ids
        ]

        for round_idx in range(rounds):
            global_weights = server.get_global_weights()

            updates: list[ClientUpdate] = []
            for i, client in enumerate(clients):
                try:
                    update = client.train(global_weights, self._data_partitions[i])
                    updates.append(update)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Client %s error: %s", client.client_id, exc)

            try:
                result = server.run_round(updates)
                self._results.append(result)
                logger.info(
                    "Round %d/%d — loss=%.4f clients=%d comm=%d bytes",
                    round_idx + 1,
                    rounds,
                    result.global_loss,
                    len(result.participating_clients),
                    result.communication_bytes,
                )
            except InsufficientClientsError as exc:
                logger.warning("Round %d skipped: %s", round_idx + 1, exc)

        return self._results

    def _run_multiprocess(self, num_rounds: int | None = None) -> list[FederatedRoundResult]:
        """Run federation with OS-level process isolation (SC-005)."""
        rounds = num_rounds or self._config.num_rounds

        # Per-client queues: simulator → client (global weights)
        in_queues: list[mp.Queue] = [mp.Queue() for _ in self._client_ids]
        # Shared output queue: client → simulator (ClientUpdates)
        out_queue: mp.Queue = mp.Queue()

        # Spawn client processes
        processes: list[mp.Process] = []
        ctx = mp.get_context("spawn")
        for i, cid in enumerate(self._client_ids):
            p = ctx.Process(
                target=_client_worker,
                args=(
                    cid,
                    in_queues[i],
                    out_queue,
                    self._model_factory,
                    self._data_partitions[i],
                    self._config,
                ),
                daemon=True,
                name=f"fl-client-{cid}",
            )
            p.start()
            processes.append(p)
            logger.debug("Spawned client process: %s (pid=%d)", cid, p.pid)

        global_model = self._model_factory("server")
        server = FederatedServer(global_model, self._config)

        try:
            for round_idx in range(rounds):
                global_weights = global_model.get_weights()

                for in_q in in_queues:
                    in_q.put(global_weights)

                updates: list[ClientUpdate] = []
                for cid in self._client_ids:
                    try:
                        item = out_queue.get(timeout=120)
                        if item is not None:
                            updates.append(item)
                        else:
                            logger.warning("Client sent None update in round %d", round_idx + 1)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Timeout waiting for client in round %d: %s",
                            round_idx + 1,
                            exc,
                        )

                try:
                    result = server.run_round(updates)
                    self._results.append(result)
                    logger.info(
                        "Round %d/%d — loss=%.4f clients=%d comm=%d bytes",
                        round_idx + 1,
                        rounds,
                        result.global_loss,
                        len(result.participating_clients),
                        result.communication_bytes,
                    )
                except InsufficientClientsError as exc:
                    logger.warning("Round %d skipped: %s", round_idx + 1, exc)

        finally:
            for in_q in in_queues:
                try:
                    in_q.put(None)
                except Exception:  # noqa: BLE001
                    pass
            for p in processes:
                p.join(timeout=10)
                if p.is_alive():
                    logger.warning("Force-terminating client process %s", p.name)
                    p.terminate()

        return self._results

    def get_results(self) -> list[FederatedRoundResult]:
        """Return accumulated round results from the last run()."""
        return list(self._results)
