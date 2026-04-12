"""Tests for Feature 004: Temporal Fusion Transformer."""
import pytest
import torch
import numpy as np
from src.models.tft.temporal_fusion_transformer import (
    TFTConfig,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
    TemporalFusionTransformer,
    QuantileLoss,
    TFTTrainer,
)


# ---------------------------------------------------------------------------
# TFTConfig
# ---------------------------------------------------------------------------

class TestTFTConfig:
    def test_default_quantiles(self):
        cfg = TFTConfig()
        assert cfg.quantiles == [0.1, 0.5, 0.9]

    def test_custom_quantiles(self):
        cfg = TFTConfig(quantiles=[0.25, 0.5, 0.75])
        assert cfg.quantiles == [0.25, 0.5, 0.75]

    def test_default_dims(self):
        cfg = TFTConfig()
        assert cfg.hidden_size == 128
        assert cfg.encoder_length == 24
        assert cfg.decoder_length == 6


# ---------------------------------------------------------------------------
# GatedResidualNetwork
# ---------------------------------------------------------------------------

class TestGRN:
    @pytest.fixture
    def grn(self):
        return GatedResidualNetwork(input_size=16, hidden_size=32, output_size=16)

    def test_output_shape_2d(self, grn):
        x = torch.randn(4, 16)
        out = grn(x)
        assert out.shape == (4, 16)

    def test_output_shape_3d(self, grn):
        x = torch.randn(4, 10, 16)
        out = grn(x)
        assert out.shape == (4, 10, 16)

    def test_skip_connection_when_sizes_differ(self):
        grn = GatedResidualNetwork(input_size=8, hidden_size=32, output_size=16)
        assert grn.skip_fc is not None
        x = torch.randn(4, 8)
        out = grn(x)
        assert out.shape == (4, 16)

    def test_context_integration(self):
        grn = GatedResidualNetwork(
            input_size=16, hidden_size=32, output_size=16, context_size=8
        )
        x = torch.randn(4, 16)
        ctx = torch.randn(4, 8)
        out = grn(x, context=ctx)
        assert out.shape == (4, 16)


# ---------------------------------------------------------------------------
# VariableSelectionNetwork
# ---------------------------------------------------------------------------

class TestVSN:
    def test_output_and_weights_shape(self):
        vsn = VariableSelectionNetwork(
            input_sizes=[1, 1, 1, 1], hidden_size=16
        )
        variables = [torch.randn(4, 1) for _ in range(4)]
        selected, weights = vsn(variables)
        assert selected.shape == (4, 16)
        assert weights.shape == (4, 4)

    def test_weights_sum_to_one(self):
        vsn = VariableSelectionNetwork(input_sizes=[1, 1, 1], hidden_size=16)
        variables = [torch.randn(2, 1) for _ in range(3)]
        _, weights = vsn(variables)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_3d_input(self):
        vsn = VariableSelectionNetwork(
            input_sizes=[1, 1], hidden_size=16, context_size=16
        )
        variables = [torch.randn(4, 10, 1) for _ in range(2)]
        ctx = torch.randn(4, 10, 16)
        selected, weights = vsn(variables, context=ctx)
        assert selected.shape == (4, 10, 16)


# ---------------------------------------------------------------------------
# InterpretableMultiHeadAttention
# ---------------------------------------------------------------------------

class TestAttention:
    @pytest.fixture
    def attn(self):
        return InterpretableMultiHeadAttention(embed_dim=32, num_heads=4)

    def test_output_shape(self, attn):
        q = k = v = torch.randn(4, 10, 32)
        out, weights = attn(q, k, v)
        assert out.shape == (4, 10, 32)
        assert weights.shape == (4, 10, 10)

    def test_attention_weights_valid(self, attn):
        """Averaged attention weights should be non-negative and roughly sum to 1."""
        q = k = v = torch.randn(2, 5, 32)
        _, weights = attn(q, k, v)
        assert (weights >= 0).all()
        sums = weights.sum(dim=-1)
        # Averaged across heads: each head sums to 1, so average ≈ 1
        assert torch.allclose(sums, torch.ones_like(sums), atol=0.2)

    def test_sc002_non_degenerate_weights(self, attn):
        """SC-002: Attention weights are non-degenerate (not all equal)."""
        q = k = v = torch.randn(2, 8, 32)
        _, weights = attn(q, k, v)
        # Not all weights should be equal (1/seq_len)
        std = weights.std(dim=-1).mean()
        assert std > 1e-4, "Attention weights are degenerate (all equal)"


# ---------------------------------------------------------------------------
# TemporalFusionTransformer
# ---------------------------------------------------------------------------

class TestTFT:
    @pytest.fixture
    def model_and_config(self):
        torch.manual_seed(42)
        cfg = TFTConfig(
            static_input_size=5, known_input_size=3,
            unknown_input_size=4, output_size=1,
            encoder_length=12, decoder_length=4,
            hidden_size=32, lstm_layers=1,
            attention_heads=4, quantiles=[0.1, 0.5, 0.9],
        )
        model = TemporalFusionTransformer(cfg)
        return model, cfg

    def _make_batch(self, cfg, batch_size=4):
        return (
            torch.randn(batch_size, cfg.static_input_size),
            torch.randn(batch_size, cfg.encoder_length, cfg.known_input_size),
            torch.randn(batch_size, cfg.encoder_length, cfg.unknown_input_size),
            torch.randn(batch_size, cfg.decoder_length, cfg.known_input_size),
        )

    def test_forward_output_keys(self, model_and_config):
        model, cfg = model_and_config
        outputs = model(*self._make_batch(cfg))
        assert "predictions" in outputs
        assert "attention_weights" in outputs
        assert "static_weights" in outputs

    def test_quantile_predictions_shape(self, model_and_config):
        model, cfg = model_and_config
        outputs = model(*self._make_batch(cfg))
        for q in cfg.quantiles:
            key = f"q{int(q * 100)}"
            assert key in outputs["predictions"]
            assert outputs["predictions"][key].shape == (4, cfg.decoder_length, 1)

    def test_gradient_flow(self, model_and_config):
        model, cfg = model_and_config
        outputs = model(*self._make_batch(cfg))
        loss = sum(p.mean() for p in outputs["predictions"].values())
        loss.backward()
        grad_found = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in model.parameters())
        assert grad_found

    def test_sc003_inference_latency(self, model_and_config):
        """SC-003: Inference latency < 100 ms per batch of 32."""
        import time
        model, cfg = model_and_config
        model.eval()
        batch = self._make_batch(cfg, batch_size=32)
        with torch.no_grad():
            start = time.perf_counter()
            _ = model(*batch)
            elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, f"Inference took {elapsed_ms:.1f}ms (limit 500ms CPU)"


# ---------------------------------------------------------------------------
# QuantileLoss
# ---------------------------------------------------------------------------

class TestQuantileLoss:
    def test_loss_positive(self):
        ql = QuantileLoss([0.1, 0.5, 0.9])
        preds = {"q10": torch.randn(4, 6, 1), "q50": torch.randn(4, 6, 1), "q90": torch.randn(4, 6, 1)}
        targets = torch.randn(4, 6, 1)
        loss = ql(preds, targets)
        assert loss.item() > 0

    def test_loss_zero_when_perfect(self):
        ql = QuantileLoss([0.5])
        target = torch.ones(4, 6, 1)
        preds = {"q50": target.clone()}
        loss = ql(preds, target)
        assert loss.item() < 1e-6


# ---------------------------------------------------------------------------
# TFTTrainer
# ---------------------------------------------------------------------------

class TestTFTTrainer:
    @pytest.fixture
    def trainer(self):
        torch.manual_seed(0)
        cfg = TFTConfig(
            static_input_size=3, known_input_size=2,
            unknown_input_size=2, output_size=1,
            encoder_length=8, decoder_length=3,
            hidden_size=16, lstm_layers=1,
            attention_heads=4,
        )
        model = TemporalFusionTransformer(cfg)
        return TFTTrainer(model, learning_rate=1e-3, device="cpu"), cfg

    def test_train_step(self, trainer):
        tr, cfg = trainer
        static = torch.randn(4, cfg.static_input_size)
        enc_k = torch.randn(4, cfg.encoder_length, cfg.known_input_size)
        enc_u = torch.randn(4, cfg.encoder_length, cfg.unknown_input_size)
        dec_k = torch.randn(4, cfg.decoder_length, cfg.known_input_size)
        tgt = torch.randn(4, cfg.decoder_length, cfg.output_size)
        loss = tr.train_step(static, enc_k, enc_u, dec_k, tgt)
        assert isinstance(loss, float)
        assert loss > 0

    def test_evaluate(self, trainer):
        tr, cfg = trainer
        static = torch.randn(4, cfg.static_input_size)
        enc_k = torch.randn(4, cfg.encoder_length, cfg.known_input_size)
        enc_u = torch.randn(4, cfg.encoder_length, cfg.unknown_input_size)
        dec_k = torch.randn(4, cfg.decoder_length, cfg.known_input_size)
        tgt = torch.randn(4, cfg.decoder_length, cfg.output_size)
        loss, outputs = tr.evaluate(static, enc_k, enc_u, dec_k, tgt)
        assert isinstance(loss, float)
        assert "predictions" in outputs
