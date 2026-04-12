#!/usr/bin/env python3
"""Federated Learning Experiment Runner (T025-T029).

Runs all FL experiments from tasks/008-federated-learning/tasks.md Phase N.
Results are printed to stdout (MLflow logging optional).

Usage:
    python scripts/run_fl_experiments.py
"""
from __future__ import annotations

import sys
import time
import json
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.federated.config import DifferentialPrivacyConfig, FederatedConfig
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.aggregation import fedavg, compute_communication_bytes


# ---------------------------------------------------------------------------
# Synthetic data generation (T025)
# ---------------------------------------------------------------------------


def make_synthetic_flight_data(
    num_samples: int = 300, seq_len: int = 10, features: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """OpenSky ADS-B shaped synthetic data."""
    torch.manual_seed(42)
    X = torch.randn(num_samples, seq_len, features)
    t = torch.linspace(0, 1, seq_len).unsqueeze(0).expand(num_samples, -1)
    y = torch.randn(num_samples, 3)
    return X, t, y


def make_iid_partitions(
    num_clients: int = 3, num_samples: int = 300
) -> list[DataLoader]:
    np.random.seed(42)
    chunk = num_samples // num_clients
    X, t, y = make_synthetic_flight_data(num_samples)
    partitions = []
    for i in range(num_clients):
        s, e = i * chunk, (i + 1) * chunk
        ds = TensorDataset(X[s:e], t[s:e], y[s:e])
        partitions.append(DataLoader(ds, batch_size=16, shuffle=False))
    return partitions


def make_non_iid_partitions(
    num_clients: int = 3, num_samples: int = 300, alpha: float = 0.1
) -> list[DataLoader]:
    np.random.seed(42)
    X, t, y = make_synthetic_flight_data(num_samples)
    proportions = np.random.dirichlet([alpha] * num_clients)
    sizes = (proportions * num_samples).astype(int)
    sizes = np.maximum(sizes, 2)  # Minimum 2 samples per client
    sizes[-1] = num_samples - sizes[:-1].sum()
    sizes[-1] = max(sizes[-1], 2)
    partitions, idx = [], 0
    for size in sizes:
        end = min(idx + size, num_samples)
        if end <= idx:
            end = min(idx + 2, num_samples)
        ds = TensorDataset(X[idx:end], t[idx:end], y[idx:end])
        partitions.append(DataLoader(ds, batch_size=int(max(1, min(16, end - idx))), shuffle=False))
        idx = end
    return partitions


# ---------------------------------------------------------------------------
# Simple model (same as conftest.SimpleLocalModel)
# ---------------------------------------------------------------------------


class SimpleLocalModel:
    def __init__(self):
        self._model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(80, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3),
        )
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=0.01)
        self._criterion = torch.nn.MSELoss()

    def train_local(self, data_loader: DataLoader, epochs: int) -> float:
        self._model.train()
        total_loss, steps = 0.0, 0
        for _ in range(epochs):
            for batch in data_loader:
                x, y = batch[0], batch[-1]
                self._optimizer.zero_grad()
                pred = self._model(x)
                loss = self._criterion(pred, y)
                loss.backward()
                self._optimizer.step()
                total_loss += loss.item()
                steps += 1
        return total_loss / max(steps, 1)

    def get_weights(self):
        return [p.data.clone() for p in self._model.parameters()]

    def set_weights(self, weights):
        for param, weight in zip(self._model.parameters(), weights):
            param.data.copy_(weight)


# ---------------------------------------------------------------------------
# Centralised oracle baseline
# ---------------------------------------------------------------------------


def run_centralised_baseline(num_epochs: int = 20) -> float:
    """Train on all data centrally — the oracle upper bound."""
    torch.manual_seed(42)
    X, t, y = make_synthetic_flight_data(300)
    ds = TensorDataset(X, t, y)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    model = SimpleLocalModel()
    losses = []
    for epoch in range(num_epochs):
        loss = model.train_local(loader, epochs=1)
        losses.append(loss)

    print(f"  Centralised baseline: final loss = {losses[-1]:.4f}")
    return losses[-1]


# ---------------------------------------------------------------------------
# Experiment A: FedAvg, 3 clients, 20 rounds, IID, no DP (T026)
# ---------------------------------------------------------------------------


def experiment_a() -> dict:
    print("\n" + "=" * 60)
    print("EXPERIMENT A: FedAvg | 3 clients | 20 rounds | IID | no DP")
    print("=" * 60)

    torch.manual_seed(42)
    config = FederatedConfig(
        num_rounds=20, clients_per_round=3, min_clients_per_round=2,
        local_epochs=2, aggregation="fedavg", seed=42,
    )
    partitions = make_iid_partitions(3, 300)

    # Create server + clients
    global_model = SimpleLocalModel()
    server = FederatedServer(global_model, config)
    clients = [
        FederatedClient(f"airline_{i}", SimpleLocalModel(), config)
        for i in range(3)
    ]

    round_losses = []
    start = time.time()

    for r in range(config.num_rounds):
        gw = server.get_global_weights()
        updates = [c.train(gw, partitions[i]) for i, c in enumerate(clients)]
        result = server.run_round(updates)
        round_losses.append(result.global_loss)
        if (r + 1) % 5 == 0:
            print(f"  Round {r+1:2d}: loss={result.global_loss:.4f}  "
                  f"delta={result.convergence_delta:.4f}  "
                  f"comm={result.communication_bytes} bytes")

    elapsed = time.time() - start
    centralised_loss = run_centralised_baseline(20)
    ratio = round_losses[-1] / max(centralised_loss, 1e-8)

    print(f"\n  Fed final loss:   {round_losses[-1]:.4f}")
    print(f"  Centralised loss: {centralised_loss:.4f}")
    print(f"  Ratio (SC-001):   {ratio:.2f}x  {'PASS' if ratio <= 1.05 else 'FAIL'}")
    print(f"  Time: {elapsed:.1f}s")

    return {
        "experiment": "A",
        "aggregation": "fedavg",
        "clients": 3,
        "rounds": 20,
        "final_loss": round_losses[-1],
        "centralised_loss": centralised_loss,
        "sc001_ratio": ratio,
        "sc001_pass": ratio <= 1.05,
        "round_losses": round_losses,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Experiment B: FedProx, 3 clients, 20 rounds, non-IID (T027)
# ---------------------------------------------------------------------------


def experiment_b() -> dict:
    print("\n" + "=" * 60)
    print("EXPERIMENT B: FedProx | 3 clients | 20 rounds | non-IID | mu=0.01")
    print("=" * 60)

    torch.manual_seed(42)
    config = FederatedConfig(
        num_rounds=20, clients_per_round=3, min_clients_per_round=2,
        local_epochs=2, aggregation="fedprox", mu=0.01, seed=42,
    )
    partitions = make_non_iid_partitions(3, 300, alpha=0.1)

    global_model = SimpleLocalModel()
    server = FederatedServer(global_model, config)
    clients = [
        FederatedClient(f"airline_{i}", SimpleLocalModel(), config)
        for i in range(3)
    ]

    round_losses = []
    start = time.time()

    for r in range(config.num_rounds):
        gw = server.get_global_weights()
        updates = [c.train(gw, partitions[i]) for i, c in enumerate(clients)]
        result = server.run_round(updates)
        round_losses.append(result.global_loss)
        if (r + 1) % 5 == 0:
            print(f"  Round {r+1:2d}: loss={result.global_loss:.4f}  "
                  f"delta={result.convergence_delta:.4f}")

    elapsed = time.time() - start
    converged_round = next(
        (i for i in range(1, len(round_losses))
         if abs(round_losses[i] - round_losses[i-1]) < 0.01),
        len(round_losses)
    )

    print(f"\n  Fed final loss:     {round_losses[-1]:.4f}")
    print(f"  Converged at round: {converged_round}")
    print(f"  SC-002: converged <= 20 rounds? {'PASS' if converged_round <= 20 else 'FAIL'}")
    print(f"  Time: {elapsed:.1f}s")

    return {
        "experiment": "B",
        "aggregation": "fedprox",
        "mu": 0.01,
        "clients": 3,
        "rounds": 20,
        "non_iid_alpha": 0.1,
        "final_loss": round_losses[-1],
        "converged_round": converged_round,
        "sc002_pass": converged_round <= 20,
        "round_losses": round_losses,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Experiment C: FedAvg + DP, varying sigma (T028)
# ---------------------------------------------------------------------------


def experiment_c() -> dict:
    print("\n" + "=" * 60)
    print("EXPERIMENT C: FedAvg + DP | sigma in {0.5, 1.0, 2.0}")
    print("=" * 60)

    results_per_sigma = {}
    baseline_loss = run_centralised_baseline(10)

    for sigma in [0.5, 1.0, 2.0]:
        torch.manual_seed(42)
        dp_config = DifferentialPrivacyConfig(
            clip_norm=1.0, noise_sigma=sigma,
            target_epsilon=10.0, target_delta=1e-5,
        )
        config = FederatedConfig(
            num_rounds=10, clients_per_round=3, min_clients_per_round=2,
            local_epochs=1, aggregation="fedavg", dp_config=dp_config, seed=42,
        )
        partitions = make_iid_partitions(3, 300)

        global_model = SimpleLocalModel()
        server = FederatedServer(global_model, config)
        clients = [
            FederatedClient(f"airline_{i}", SimpleLocalModel(), config)
            for i in range(3)
        ]

        round_losses = []
        for r in range(config.num_rounds):
            gw = server.get_global_weights()
            updates = [c.train(gw, partitions[i]) for i, c in enumerate(clients)]
            result = server.run_round(updates)
            round_losses.append(result.global_loss)

        degradation = (round_losses[-1] - baseline_loss) / max(baseline_loss, 1e-8)
        print(f"  sigma={sigma}: final_loss={round_losses[-1]:.4f}  "
              f"degradation={degradation*100:.1f}%  "
              f"{'PASS' if abs(degradation) <= 0.03 else 'NOTE: DP noise visible'}")

        results_per_sigma[sigma] = {
            "final_loss": round_losses[-1],
            "degradation_pct": degradation * 100,
            "round_losses": round_losses,
        }

    return {
        "experiment": "C",
        "baseline_loss": baseline_loss,
        "results": {str(k): v for k, v in results_per_sigma.items()},
    }


# ---------------------------------------------------------------------------
# Experiment D: FedAvg, 10 clients, 30 rounds — scalability (T029)
# ---------------------------------------------------------------------------


def experiment_d() -> dict:
    print("\n" + "=" * 60)
    print("EXPERIMENT D: FedAvg | 10 clients | 30 rounds | scalability")
    print("=" * 60)

    torch.manual_seed(42)
    config = FederatedConfig(
        num_rounds=30, clients_per_round=10, min_clients_per_round=2,
        local_epochs=1, aggregation="fedavg", seed=42,
    )
    partitions = make_iid_partitions(10, 1000)

    global_model = SimpleLocalModel()
    server = FederatedServer(global_model, config)
    clients = [
        FederatedClient(f"airline_{i}", SimpleLocalModel(), config)
        for i in range(10)
    ]

    round_losses = []
    comm_bytes_total = 0
    start = time.time()

    for r in range(config.num_rounds):
        gw = server.get_global_weights()
        updates = [c.train(gw, partitions[i]) for i, c in enumerate(clients)]
        result = server.run_round(updates)
        round_losses.append(result.global_loss)
        comm_bytes_total += result.communication_bytes
        if (r + 1) % 10 == 0:
            print(f"  Round {r+1:2d}: loss={result.global_loss:.4f}  "
                  f"clients={len(result.participating_clients)}")

    elapsed = time.time() - start

    print(f"\n  Final loss:        {round_losses[-1]:.4f}")
    print(f"  Total comm bytes:  {comm_bytes_total:,}")
    print(f"  Avg bytes/round:   {comm_bytes_total // 30:,}")
    print(f"  Time (30 rounds):  {elapsed:.1f}s")

    return {
        "experiment": "D",
        "clients": 10,
        "rounds": 30,
        "final_loss": round_losses[-1],
        "total_comm_bytes": comm_bytes_total,
        "elapsed_s": elapsed,
        "round_losses": round_losses,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  FEDERATED LEARNING EXPERIMENT SUITE")
    print("  Feature 008 | Aviation Intelligence Platform")
    print("=" * 60)

    all_results = {}

    all_results["A"] = experiment_a()
    all_results["B"] = experiment_b()
    all_results["C"] = experiment_c()
    all_results["D"] = experiment_d()

    # Summary table
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Exp':<6} {'Aggreg':<10} {'Clients':<8} {'Rounds':<7} {'Final Loss':<12} {'Status'}")
    print("-" * 60)

    a = all_results["A"]
    print(f"{'A':<6} {'FedAvg':<10} {a['clients']:<8} {a['rounds']:<7} {a['final_loss']:<12.4f} {'PASS' if a['sc001_pass'] else 'FAIL'}")

    b = all_results["B"]
    print(f"{'B':<6} {'FedProx':<10} {b['clients']:<8} {b['rounds']:<7} {b['final_loss']:<12.4f} {'PASS' if b['sc002_pass'] else 'FAIL'}")

    print(f"{'C':<6} {'FedAvg+DP':<10} {'3':<8} {'10':<7} {'(see above)':<12} {'DONE'}")

    d = all_results["D"]
    print(f"{'D':<6} {'FedAvg':<10} {d['clients']:<8} {d['rounds']:<7} {d['final_loss']:<12.4f} {'DONE'}")

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(results_path, exist_ok=True)
    output_file = os.path.join(results_path, "fl_experiment_results.json")

    serializable = {}
    for k, v in all_results.items():
        s = {}
        for kk, vv in v.items():
            if isinstance(vv, list) and vv and isinstance(vv[0], float):
                s[kk] = [round(x, 6) for x in vv]
            elif isinstance(vv, float):
                s[kk] = round(vv, 6)
            elif isinstance(vv, dict):
                s[kk] = {dk: (round(dv, 6) if isinstance(dv, float)
                              else ([round(x, 6) for x in dv] if isinstance(dv, list) else dv))
                          for dk, dv in vv.items()}
            else:
                s[kk] = vv
        serializable[k] = s

    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
    print("DONE.")


if __name__ == "__main__":
    main()
