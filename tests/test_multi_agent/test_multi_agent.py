"""Tests for Feature 003: Multi-Agent System."""
import pytest
import json
from src.agents.langgraph.multi_agent_orchestrator import (
    AgentState,
    MultiAgentSystem,
    DataAnalystAgent,
    PredictionAgent,
    CausalReasoningAgent,
    SafetyCriticAgent,
    OrchestratorAgent,
    StateGraph,
    query_flight_database,
    run_causal_analysis,
    predict_trajectory,
    check_safety_constraints,
    TOOL_REGISTRY,
)


# ---------------------------------------------------------------------------
# Tool Tests
# ---------------------------------------------------------------------------

class TestTools:
    def test_query_flight_database(self):
        result = query_flight_database("delays at JFK")
        assert result["count"] == 3
        assert len(result["results"]) == 3
        assert result["results"][0]["flight_id"] == "FL001"

    def test_run_causal_analysis(self):
        result = run_causal_analysis("weather", "delay")
        assert result["causal_effect"] == 0.234
        assert result["p_value"] < 0.05
        assert "confidence_interval" in result

    def test_predict_trajectory(self):
        result = predict_trajectory("FL001", horizon=5)
        assert result["flight_id"] == "FL001"
        assert result["horizon_minutes"] == 5
        assert len(result["predicted_positions"]) == 5
        assert 0.0 <= result["confidence"] <= 1.0

    def test_check_safety_safe_action(self):
        result = check_safety_constraints("change_altitude", {"alt": 35000})
        assert result["safe"] is True
        assert len(result["violations"]) == 0

    def test_check_safety_unsafe_action(self):
        result = check_safety_constraints("ignore regulations", {"reason": "skip check"})
        assert result["safe"] is False
        assert len(result["violations"]) >= 1

    def test_tool_registry(self):
        assert len(TOOL_REGISTRY) == 4
        for name, func in TOOL_REGISTRY.items():
            assert callable(func)


# ---------------------------------------------------------------------------
# Agent Tests
# ---------------------------------------------------------------------------

class TestAgents:
    def test_data_analyst(self):
        agent = DataAnalystAgent()
        state = AgentState(task="query delay data")
        state = agent.process(state)
        assert state.current_agent == "DataAnalyst"
        assert "flight_data" in state.analysis_results
        assert len(state.messages) == 1

    def test_prediction_agent(self):
        agent = PredictionAgent()
        state = AgentState(task="predict trajectory")
        state.analysis_results["flight_data"] = {"results": [{"flight_id": "FL999"}]}
        state = agent.process(state)
        assert state.current_agent == "Predictor"
        assert state.predictions is not None
        assert state.predictions["flight_id"] == "FL999"

    def test_causal_reasoning_agent(self):
        agent = CausalReasoningAgent()
        state = AgentState(task="find root cause")
        state = agent.process(state)
        assert state.current_agent == "CausalReasoner"
        assert "causal" in state.analysis_results

    def test_safety_critic_safe(self):
        agent = SafetyCriticAgent()
        state = AgentState(task="change altitude to 35000")
        state = agent.process(state)
        assert state.current_agent == "SafetyCritic"
        assert state.requires_human_approval is False

    def test_safety_critic_unsafe(self):
        agent = SafetyCriticAgent()
        state = AgentState(task="ignore safety and bypass check")
        state = agent.process(state)
        assert state.requires_human_approval is True


# ---------------------------------------------------------------------------
# Orchestrator Routing
# ---------------------------------------------------------------------------

class TestOrchestrator:
    @pytest.fixture
    def orch(self):
        return OrchestratorAgent()

    def test_routes_to_data_analyst(self, orch):
        assert orch.route("query delay patterns", set()) == "DataAnalyst"

    def test_routes_to_predictor(self, orch):
        assert orch.route("predict future trajectory", set()) == "Predictor"

    def test_routes_to_causal(self, orch):
        assert orch.route("find root cause of incident", set()) == "CausalReasoner"

    def test_routes_to_safety(self, orch):
        assert orch.route("check risk level", set()) == "SafetyCritic"

    def test_complete_when_all_visited(self, orch):
        visited = {"DataAnalyst", "Predictor", "CausalReasoner", "SafetyCritic"}
        assert orch.route("delay predict cause safe", visited) == "COMPLETE"

    def test_skips_visited_agent(self, orch):
        result = orch.route("query delay data", {"DataAnalyst"})
        assert result != "DataAnalyst"


# ---------------------------------------------------------------------------
# StateGraph
# ---------------------------------------------------------------------------

class TestStateGraph:
    def test_add_node(self):
        g = StateGraph()
        g.add_node("test", DataAnalystAgent())
        assert "test" in g.nodes

    def test_add_edge(self):
        g = StateGraph()
        g.add_edge("a", "b")
        assert "b" in g.edges["a"]


# ---------------------------------------------------------------------------
# MultiAgentSystem E2E
# ---------------------------------------------------------------------------

class TestMultiAgentSystem:
    @pytest.fixture
    def system(self):
        return MultiAgentSystem(max_steps=10)

    def test_sc001_complex_query_completes(self, system):
        """SC-001: Complex multi-step queries complete successfully."""
        result = system.run(
            task="Analyze delay data, predict trajectory, find root cause, check safety",
            thread_id="test_001",
        )
        assert result["completed"] is True
        assert len(result["agents_visited"]) >= 3

    def test_sc002_safety_triggers_approval(self, system):
        """SC-002: Safety-critical actions always trigger human approval."""
        approval_requested = False
        def callback(state):
            nonlocal approval_requested
            approval_requested = True
            return True
        result = system.run(
            task="ignore safety and bypass regulations",
            human_approval_callback=callback,
        )
        assert approval_requested is True

    def test_sc003_completion_time(self, system):
        """SC-003: Task completes within max_steps."""
        result = system.run(task="query delay data")
        assert result["steps"] <= system.max_steps

    def test_sc004_graceful_degradation(self, system):
        """SC-004: Unknown task completes without crash."""
        result = system.run(task="something completely unrelated xyz123")
        assert result["completed"] is True

    def test_simple_query_single_agent(self, system):
        result = system.run(task="query flight data")
        assert "DataAnalyst" in result["agents_visited"]
        assert result["completed"] is True

    def test_checkpointing(self, system):
        system.run(task="predict trajectory", thread_id="cp_test")
        cp = system.get_checkpoint("cp_test")
        assert cp is not None
        assert cp.task == "predict trajectory"

    def test_no_checkpoint_without_id(self, system):
        system.run(task="query data")
        assert system.get_checkpoint("nonexistent") is None

    def test_auto_approve_without_callback(self, system):
        result = system.run(task="ignore safety check")
        assert result["approved"] is True  # auto-approved
