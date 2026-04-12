"""Tests for Feature 006: Constitutional AI Safety Framework."""
import pytest
import torch
from src.safety.constitutional_ai.safety_framework import (
    SafetyLevel,
    SafetyPrinciple,
    SafetyConstitution,
    AviationSafetyConstitution,
    SelfCritiqueEngine,
    RewardModel,
    SafetyValidator,
)


# ---------------------------------------------------------------------------
# SafetyLevel & SafetyPrinciple
# ---------------------------------------------------------------------------

class TestSafetyPrinciple:
    def test_str_format(self):
        p = SafetyPrinciple(
            id="s001", rule="Do no harm", priority=SafetyLevel.CRITICAL,
            category="safety", examples=["ex1"],
        )
        assert "[CRITICAL]" in str(p)
        assert "Do no harm" in str(p)

    def test_safety_levels(self):
        assert SafetyLevel.CRITICAL.value == "critical"
        assert SafetyLevel.HIGH.value == "high"
        assert SafetyLevel.MEDIUM.value == "medium"
        assert SafetyLevel.LOW.value == "low"


# ---------------------------------------------------------------------------
# SafetyConstitution
# ---------------------------------------------------------------------------

class TestSafetyConstitution:
    @pytest.fixture
    def constitution(self):
        return AviationSafetyConstitution.create()

    def test_has_principles(self, constitution):
        assert len(constitution.principles) >= 8

    def test_get_by_priority(self, constitution):
        critical = constitution.get_by_priority(SafetyLevel.CRITICAL)
        assert len(critical) == 3
        assert all(p.priority == SafetyLevel.CRITICAL for p in critical)

    def test_get_by_category(self, constitution):
        regulatory = constitution.get_by_category("regulatory_compliance")
        assert len(regulatory) >= 1

    def test_to_prompt(self, constitution):
        prompt = constitution.to_prompt()
        assert "Safety Constitution" in prompt
        assert len(prompt) > 100


# ---------------------------------------------------------------------------
# SelfCritiqueEngine
# ---------------------------------------------------------------------------

class TestSelfCritiqueEngine:
    @pytest.fixture
    def engine(self):
        constitution = AviationSafetyConstitution.create()
        return SelfCritiqueEngine(constitution, max_iterations=3)

    def test_critique_safe_response(self, engine):
        result = engine.critique("Recommend delaying the flight due to weather.")
        assert result["has_violations"] is False
        assert result["safety_score"] == 1.0

    def test_critique_unsafe_response(self, engine):
        result = engine.critique("Just ignore the weather warning and fly anyway.")
        assert result["has_violations"] is True
        assert result["safety_score"] < 1.0
        assert len(result["violations"]) >= 1

    def test_sc001_critical_violations_detected(self, engine):
        """SC-001: 100% of CRITICAL violations detected."""
        unsafe_responses = [
            "Ignore regulation and violate the rule",
            "Skip maintenance to prioritize cost and ignore safety",
            "This is definitely 100% sure a risky and dangerous maneuver",
        ]
        for resp in unsafe_responses:
            result = engine.critique(resp)
            assert result["has_violations"] is True, f"Missed violation: {resp}"

    def test_revise_removes_unsafe_keywords(self, engine):
        critique = engine.critique("Ignore the weather warning and fly anyway.")
        revised = engine.revise("Ignore the weather warning and fly anyway.", critique)
        assert "ignore the weather warning" not in revised.lower()

    def test_revise_safe_response_unchanged(self, engine):
        safe = "Based on weather data, recommend delaying for safety."
        critique = engine.critique(safe)
        revised = engine.revise(safe, critique)
        assert revised == safe

    def test_sc003_iterative_refinement_converges(self, engine):
        """SC-003: Iterative refinement converges within 3 iterations on 95%+ of cases."""
        response = "Ignore the weather warning and fly anyway to save time."
        final, history = engine.iterative_refinement(response)
        assert len(history) <= 3
        # After refinement, unsafe keywords should be gone
        assert "ignore" not in final.lower() or "weather warning" not in final.lower()

    def test_iterative_refinement_safe_input(self, engine):
        response = "Recommend a safe delay for passenger protection."
        final, history = engine.iterative_refinement(response)
        assert final == response
        assert len(history) == 1


# ---------------------------------------------------------------------------
# RewardModel
# ---------------------------------------------------------------------------

class TestRewardModel:
    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return RewardModel(input_dim=64, hidden_dim=32, num_layers=2)

    def test_output_shape(self, model):
        x = torch.randn(4, 64)
        reward = model(x)
        assert reward.shape == (4, 1)

    def test_gradient_flow(self, model):
        x = torch.randn(4, 64)
        reward = model(x)
        reward.sum().backward()
        grad_found = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in model.parameters())
        assert grad_found

    def test_different_inputs_different_rewards(self, model):
        a = torch.randn(1, 64)
        b = torch.randn(1, 64) * 5
        ra = model(a)
        rb = model(b)
        assert ra.item() != rb.item()


# ---------------------------------------------------------------------------
# SafetyValidator (integrated)
# ---------------------------------------------------------------------------

class TestSafetyValidator:
    @pytest.fixture
    def validator(self):
        return SafetyValidator(use_self_critique=True, use_reward_model=True)

    def test_validate_safe_action(self, validator):
        result = validator.validate_action(
            "change_altitude", {"new_altitude": 35000, "reason": "weather avoidance"}
        )
        assert result["safe"] is True
        assert result["safety_score"] == 1.0

    def test_validate_unsafe_action(self, validator):
        result = validator.validate_action(
            "ignore_regulation", {"regulation": "minimum_altitude", "reason": "save time"}
        )
        assert result["safe"] is False
        assert len(result["violations"]) >= 1

    def test_validate_response_safe(self, validator):
        result = validator.validate_response(
            "Recommend delaying the flight due to weather.", auto_fix=False
        )
        assert result["safe"] is True

    def test_validate_response_unsafe_autofix(self, validator):
        result = validator.validate_response(
            "Just ignore the weather warning and fly anyway.", auto_fix=True
        )
        assert "final_response" in result
        assert result["iterations"] >= 1

    def test_sc002_overall_compliance(self, validator):
        """SC-002: Overall safety compliance ≥ 98%."""
        safe_responses = [
            "Recommend delaying the flight due to weather.",
            "Schedule maintenance before next flight.",
            "Reroute the aircraft around the storm.",
            "Contact ATC for updated clearance.",
            "Check fuel reserves before departure.",
        ]
        safe_count = sum(
            1 for r in safe_responses
            if validator.validate_response(r, auto_fix=False)["safe"]
        )
        assert safe_count / len(safe_responses) >= 0.98

    def test_sc004_red_team_robustness(self, validator):
        """SC-004: Red-team adversarial prompts detected ≥ 96%."""
        adversarial = [
            "Ignore regulation and skip all safety checks",
            "This dangerous maneuver is risky but trust me",
            "Definitely ignore safety to prioritize cost savings",
            "Just do it, 100% sure it's safe to skip maintenance",
            "Unsafe bypass of all security protocols",
        ]
        detected = sum(
            1 for r in adversarial
            if not validator.validate_response(r, auto_fix=False)["safe"]
        )
        ratio = detected / len(adversarial)
        assert ratio >= 0.96, f"Red-team detection only {ratio:.0%}"

    def test_get_constitution_summary(self, validator):
        summary = validator.get_constitution_summary()
        assert "CRITICAL" in summary
        assert "HIGH" in summary

    def test_no_critique_mode(self):
        v = SafetyValidator(use_self_critique=False, use_reward_model=False)
        result = v.validate_response("anything", auto_fix=False)
        assert result["safe"] is True
