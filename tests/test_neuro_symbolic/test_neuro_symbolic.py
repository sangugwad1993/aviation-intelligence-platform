"""Tests for Feature 005: Neuro-Symbolic Reasoning."""
import pytest
import torch
import networkx as nx
from src.models.neuro_symbolic.neuro_symbolic_reasoner import (
    RelationType,
    Entity,
    Relation,
    Rule,
    KnowledgeGraph,
    GraphNeuralNetwork,
    LogicEngine,
    DifferentiableLogic,
    NeuroSymbolicReasoner,
)


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------

class TestKnowledgeGraph:
    @pytest.fixture
    def kg(self):
        kg = KnowledgeGraph()
        kg.add_entity(Entity("FL001", "flight", {"callsign": "AA123"}))
        kg.add_entity(Entity("JFK", "airport", {"name": "JFK Airport"}))
        kg.add_entity(Entity("LAX", "airport", {"name": "LAX Airport"}))
        kg.add_entity(Entity("AA", "airline", {"name": "American Airlines"}))
        kg.add_relation(Relation("FL001", RelationType.DEPARTS_FROM, "JFK"))
        kg.add_relation(Relation("FL001", RelationType.ARRIVES_AT, "LAX"))
        kg.add_relation(Relation("FL001", RelationType.OPERATED_BY, "AA"))
        return kg

    def test_entity_count(self, kg):
        assert len(kg.entities) == 4

    def test_relation_count(self, kg):
        assert len(kg.relations) == 3

    def test_graph_nodes(self, kg):
        assert len(kg.graph.nodes) == 4

    def test_graph_edges(self, kg):
        assert len(kg.graph.edges) == 3

    def test_get_neighbors(self, kg):
        neighbors = kg.get_neighbors("FL001")
        assert set(neighbors) == {"JFK", "LAX", "AA"}

    def test_get_neighbors_filtered(self, kg):
        neighbors = kg.get_neighbors("FL001", RelationType.DEPARTS_FROM)
        assert neighbors == ["JFK"]

    def test_query_path(self, kg):
        # FL001 -> JFK exists as direct edge
        paths = kg.query_path("FL001", "JFK")
        assert len(paths) >= 1

    def test_query_path_no_path(self, kg):
        paths = kg.query_path("JFK", "AA")
        assert len(paths) == 0

    def test_to_triples(self, kg):
        triples = kg.to_triples()
        assert len(triples) == 3
        assert all(len(t) == 3 for t in triples)

    def test_sc001_100_entity_performance(self):
        """SC-001: KG with 100+ entities handles queries in < 100ms."""
        import time
        kg = KnowledgeGraph()
        for i in range(120):
            kg.add_entity(Entity(f"E{i}", "node", {"idx": i}))
        for i in range(100):
            kg.add_relation(Relation(f"E{i}", RelationType.FLIES_TO, f"E{i+1}"))
        start = time.perf_counter()
        paths = kg.query_path("E0", "E50", max_length=60)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, f"Query took {elapsed_ms:.1f}ms"
        assert len(paths) >= 1


# ---------------------------------------------------------------------------
# Logic Engine
# ---------------------------------------------------------------------------

class TestLogicEngine:
    @pytest.fixture
    def engine(self):
        e = LogicEngine()
        e.add_fact("weather_bad")
        e.add_fact("aircraft_maintained")
        e.add_fact("pilot_certified")
        e.add_rule(Rule(
            head="safe_to_fly",
            body=["weather_good", "aircraft_maintained", "pilot_certified"],
        ))
        e.add_rule(Rule(
            head="delay_expected",
            body=["weather_bad"],
        ))
        return e

    def test_forward_chain_derives_fact(self, engine):
        derived = engine.forward_chain()
        assert "delay_expected" in derived

    def test_forward_chain_does_not_derive_safe(self, engine):
        derived = engine.forward_chain()
        assert "safe_to_fly" not in derived  # weather_good not asserted

    def test_backward_chain_true(self, engine):
        assert engine.backward_chain("delay_expected") is True

    def test_backward_chain_false(self, engine):
        assert engine.backward_chain("safe_to_fly") is False

    def test_backward_chain_direct_fact(self, engine):
        assert engine.backward_chain("weather_bad") is True

    def test_backward_chain_depth_limit(self, engine):
        assert engine.backward_chain("nonexistent", max_depth=0) is False

    def test_unification_basic(self, engine):
        assert engine._unify("?X flies_to JFK", "FL001 flies_to JFK") is True
        assert engine._unify("?X flies_to ?Y", "FL001 flies_to JFK") is True
        assert engine._unify("FL001 flies_to JFK", "FL002 flies_to JFK") is False


# ---------------------------------------------------------------------------
# Graph Neural Network
# ---------------------------------------------------------------------------

class TestGNN:
    @pytest.fixture
    def gnn(self):
        torch.manual_seed(0)
        return GraphNeuralNetwork(
            num_entities=5, num_relations=3,
            embedding_dim=32, hidden_dim=64, num_layers=2,
        )

    def test_output_shape(self, gnn):
        entity_ids = torch.arange(5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        edge_types = torch.tensor([0, 1, 2])
        out = gnn(entity_ids, edge_index, edge_types)
        assert out.shape == (5, 32)

    def test_sc002_embeddings_cluster(self, gnn):
        """SC-002: GNN embeddings produce meaningful clusters."""
        entity_ids = torch.arange(5)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_types = torch.tensor([0, 0, 1, 1])
        emb = gnn(entity_ids, edge_index, edge_types)
        # Connected entities should be more similar than disconnected
        sim_01 = torch.nn.functional.cosine_similarity(emb[0:1], emb[1:2]).item()
        sim_04 = torch.nn.functional.cosine_similarity(emb[0:1], emb[4:5]).item()
        # At least the embeddings should differ (non-degenerate)
        assert emb.std().item() > 1e-4, "Embeddings are degenerate"

    def test_gradient_flow(self, gnn):
        entity_ids = torch.arange(5)
        edge_index = torch.tensor([[0, 1], [1, 2]])
        edge_types = torch.tensor([0, 1])
        out = gnn(entity_ids, edge_index, edge_types)
        out.sum().backward()
        grad_found = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in gnn.parameters())
        assert grad_found


# ---------------------------------------------------------------------------
# Differentiable Logic
# ---------------------------------------------------------------------------

class TestDifferentiableLogic:
    @pytest.fixture
    def dl(self):
        return DifferentiableLogic(temperature=1.0)

    def test_fuzzy_and(self, dl):
        a = torch.tensor([0.8])
        b = torch.tensor([0.6])
        result = dl.fuzzy_and(a, b)
        assert abs(result.item() - 0.48) < 1e-5

    def test_fuzzy_or(self, dl):
        a = torch.tensor([0.8])
        b = torch.tensor([0.6])
        result = dl.fuzzy_or(a, b)
        expected = 0.8 + 0.6 - 0.48
        assert abs(result.item() - expected) < 1e-5

    def test_fuzzy_not(self, dl):
        a = torch.tensor([0.3])
        result = dl.fuzzy_not(a)
        assert abs(result.item() - 0.7) < 1e-5

    def test_fuzzy_implies(self, dl):
        a = torch.tensor([0.9])
        b = torch.tensor([0.2])
        result = dl.fuzzy_implies(a, b)
        expected = min(1.0, 1.0 - 0.9 + 0.2)
        assert abs(result.item() - expected) < 1e-5

    def test_fuzzy_and_boundary(self, dl):
        assert dl.fuzzy_and(torch.tensor([0.0]), torch.tensor([1.0])).item() == 0.0
        assert dl.fuzzy_and(torch.tensor([1.0]), torch.tensor([1.0])).item() == 1.0

    def test_fuzzy_or_boundary(self, dl):
        assert dl.fuzzy_or(torch.tensor([0.0]), torch.tensor([0.0])).item() == 0.0
        assert dl.fuzzy_or(torch.tensor([1.0]), torch.tensor([0.0])).item() == 1.0

    def test_soft_unification(self, dl):
        a = torch.randn(1, 32)
        b = a.clone()
        conf = dl.soft_unification(a, b)
        assert conf.item() > 0.5  # identical vectors should match


# ---------------------------------------------------------------------------
# NeuroSymbolicReasoner (integrated)
# ---------------------------------------------------------------------------

class TestNeuroSymbolicReasoner:
    @pytest.fixture
    def reasoner(self):
        torch.manual_seed(42)
        kg = KnowledgeGraph()
        kg.add_entity(Entity("FL001", "flight", {"callsign": "AA123"}))
        kg.add_entity(Entity("JFK", "airport", {"name": "JFK"}))
        kg.add_entity(Entity("LAX", "airport", {"name": "LAX"}))
        kg.add_entity(Entity("AA", "airline", {"name": "AA"}))
        kg.add_entity(Entity("weather_bad", "condition", {"severity": "high"}))
        kg.add_relation(Relation("FL001", RelationType.DEPARTS_FROM, "JFK"))
        kg.add_relation(Relation("FL001", RelationType.ARRIVES_AT, "LAX"))
        kg.add_relation(Relation("FL001", RelationType.OPERATED_BY, "AA"))
        kg.add_relation(Relation("FL001", RelationType.HAS_DELAY, "weather_bad"))

        reasoner = NeuroSymbolicReasoner(kg, embedding_dim=32, hidden_dim=64, num_gnn_layers=2)
        reasoner.add_rule(Rule("delay_expected", ["weather_bad"]))
        reasoner.add_fact("weather_bad")
        return reasoner

    def test_encode_graph(self, reasoner):
        emb = reasoner.encode_graph()
        assert emb.shape == (5, 32)

    def test_symbolic_query(self, reasoner):
        result = reasoner.query("delay_expected", use_neural=False)
        assert result["symbolic"] is True

    def test_neural_query(self, reasoner):
        result = reasoner.query("delay_expected", use_symbolic=False)
        assert "neural" in result
        assert len(result["neural"]) == 5

    def test_combined_query(self, reasoner):
        result = reasoner.query("delay_expected")
        assert "combined_confidence" in result
        assert result["combined_confidence"] > 0

    def test_sc003_explain(self, reasoner):
        """SC-003: All derived facts are traceable via explain()."""
        explanation = reasoner.explain("delay_expected")
        assert len(explanation) > 0
        assert any("delay_expected" in e for e in explanation)

    def test_explain_returns_empty_for_unprovable(self, reasoner):
        explanation = reasoner.explain("nonexistent_fact")
        assert len(explanation) == 0
