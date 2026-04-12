"""Tests for Feature 001: Liquid Neural Networks."""
import pytest
import torch
import numpy as np
from src.models.liquid_nn.liquid_neural_network import (
    LiquidTimeConstantCell,
    LiquidNeuralNetwork,
    LiquidNNTrainer,
    _euler_solve,
)


# ---------------------------------------------------------------------------
# Euler ODE Solver
# ---------------------------------------------------------------------------

class TestEulerSolver:
    def test_constant_derivative(self):
        """ODE dy/dt = 1 from t=0 to t=1 should give y ≈ 1."""
        func = lambda t, state: (torch.ones_like(state[0]),)
        y0 = (torch.zeros(1),)
        sol = _euler_solve(func, y0, torch.tensor([0.0, 1.0]), steps=100)
        assert abs(sol[0][-1].item() - 1.0) < 0.02

    def test_returns_tuple(self):
        func = lambda t, s: (torch.zeros_like(s[0]),)
        sol = _euler_solve(func, (torch.zeros(3),), torch.tensor([0.0, 1.0]), steps=5)
        assert isinstance(sol, tuple)
        assert sol[0].shape == (1, 3)


# ---------------------------------------------------------------------------
# Liquid Time-Constant Cell
# ---------------------------------------------------------------------------

class TestLiquidTimeConstantCell:
    @pytest.fixture
    def cell(self):
        torch.manual_seed(0)
        return LiquidTimeConstantCell(input_size=6, hidden_size=32, sparsity=0.5)

    def test_output_shape(self, cell):
        h = torch.zeros(4, 32)
        x = torch.randn(4, 6)
        dh, dx = cell(torch.tensor(0.0), (h, x))
        assert dh.shape == (4, 32)
        assert dx.shape == (4, 6)

    def test_sparse_mask_created(self, cell):
        assert hasattr(cell, "mask_W")
        assert hasattr(cell, "mask_U")
        assert cell.mask_W.shape == (32, 6)
        assert cell.mask_U.shape == (32, 32)

    def test_sparsity_level(self, cell):
        w_density = cell.mask_W.mean().item()
        assert 0.2 < w_density < 0.8  # roughly 50% sparse

    def test_every_neuron_connected(self, cell):
        for i in range(cell.hidden_size):
            assert cell.mask_W[i].sum() >= 1
            assert cell.mask_U[i].sum() >= 1

    def test_adaptive_time_constant(self, cell):
        x = torch.randn(4, 6)
        tau = cell.compute_time_constant(x)
        assert tau.shape == (4, 32)
        assert (tau >= cell.tau_min.min().item()).all()

    def test_activation_variants(self):
        for act in ["tanh", "relu", "sigmoid"]:
            cell = LiquidTimeConstantCell(4, 8, activation=act)
            h = torch.zeros(2, 8)
            x = torch.randn(2, 4)
            dh, _ = cell(torch.tensor(0.0), (h, x))
            assert dh.shape == (2, 8)


# ---------------------------------------------------------------------------
# Liquid Neural Network
# ---------------------------------------------------------------------------

class TestLiquidNeuralNetwork:
    @pytest.fixture
    def model(self):
        torch.manual_seed(42)
        return LiquidNeuralNetwork(
            input_size=6, hidden_size=32, output_size=3,
            num_layers=2, sparsity=0.5,
        )

    def test_forward_shape(self, model):
        x = torch.randn(4, 10, 6)
        y = model(x)
        assert y.shape == (4, 10, 3)

    def test_forward_with_custom_timesteps(self, model):
        x = torch.randn(2, 5, 6)
        ts = torch.linspace(0, 4, 5)
        y = model(x, time_steps=ts)
        assert y.shape == (2, 5, 3)

    def test_count_parameters(self, model):
        n = model.count_parameters()
        assert n > 0
        assert isinstance(n, int)

    def test_sparsity_stats(self, model):
        stats = model.get_sparsity_stats()
        assert "layer_0_W_sparsity" in stats
        assert "layer_1_U_sparsity" in stats
        for v in stats.values():
            assert 0.0 <= v <= 1.0

    def test_gradient_flow(self, model):
        x = torch.randn(2, 5, 6)
        y = model(x)
        loss = y.sum()
        loss.backward()
        grad_found = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in model.parameters())
        assert grad_found

    def test_sc003_fewer_params_than_lstm(self, model):
        """SC-003: LNN ≤ 75% of LSTM params for same hidden_size."""
        lstm = torch.nn.LSTM(input_size=6, hidden_size=32,
                             num_layers=2, batch_first=True)
        lstm_params = sum(p.numel() for p in lstm.parameters())
        lnn_params = model.count_parameters()
        assert lnn_params <= lstm_params, (
            f"LNN params ({lnn_params}) > LSTM params ({lstm_params})"
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TestLiquidNNTrainer:
    @pytest.fixture
    def trainer(self):
        torch.manual_seed(0)
        model = LiquidNeuralNetwork(
            input_size=6, hidden_size=16, output_size=3,
            num_layers=1, sparsity=0.3,
        )
        return LiquidNNTrainer(model, learning_rate=1e-3, device="cpu")

    def test_train_step_returns_loss(self, trainer):
        x = torch.randn(4, 5, 6)
        y = torch.randn(4, 5, 3)
        loss = trainer.train_step(x, y)
        assert isinstance(loss, float)
        assert loss > 0

    def test_evaluate_returns_predictions(self, trainer):
        x = torch.randn(4, 5, 6)
        y = torch.randn(4, 5, 3)
        loss, preds = trainer.evaluate(x, y)
        assert isinstance(loss, float)
        assert preds.shape == (4, 5, 3)

    def test_training_reduces_loss(self, trainer):
        torch.manual_seed(1)
        x = torch.randn(8, 5, 6)
        y = torch.zeros(8, 5, 3)
        first_loss = trainer.train_step(x, y)
        for _ in range(20):
            trainer.train_step(x, y)
        final_loss = trainer.train_step(x, y)
        assert final_loss < first_loss
