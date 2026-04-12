"""
Neuro-Symbolic Reasoning Layer

Research Area: Combining neural networks with symbolic AI
Key Papers:
- "Neuro-Symbolic AI: The 3rd Wave" (Garcez et al., 2020)
- "DeepProbLog: Neural Probabilistic Logic Programming" (Manhaeve et al., 2018)

Why 99% of AI/ML Engineers Don't Know This:
1. Requires expertise in BOTH deep learning AND logic programming
2. Limited frameworks (DeepProbLog, Scallop, NeuroLog)
3. Not taught in standard ML courses
4. Complex integration of differentiable and discrete reasoning
5. Needs domain knowledge for rule specification

Key Components:
- Knowledge Graph: Aviation domain ontology
- Neural Embedder: Encode entities and relations
- Logic Engine: Prolog-style inference
- Differentiable Bridge: Backprop through logic
"""

from typing import Dict, List, Optional, Tuple, Set, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from dataclasses import dataclass
from enum import Enum
import re


class RelationType(Enum):
    """Types of relations in aviation knowledge graph."""
    FLIES_TO = "flies_to"
    OPERATED_BY = "operated_by"
    DEPARTS_FROM = "departs_from"
    ARRIVES_AT = "arrives_at"
    HAS_DELAY = "has_delay"
    CAUSED_BY = "caused_by"
    REQUIRES = "requires"
    VIOLATES = "violates"
    COMPLIES_WITH = "complies_with"


@dataclass
class Entity:
    """Entity in knowledge graph."""
    id: str
    type: str
    properties: Dict[str, Any]


@dataclass
class Relation:
    """Relation between entities."""
    subject: str
    predicate: RelationType
    object: str
    confidence: float = 1.0


@dataclass
class Rule:
    """Logical rule in Prolog-like syntax."""
    head: str
    body: List[str]
    confidence: float = 1.0
    
    def __str__(self) -> str:
        body_str = ", ".join(self.body)
        return f"{self.head} :- {body_str}."


class KnowledgeGraph:
    """
    Aviation domain knowledge graph.
    
    Stores entities (flights, airports, airlines) and relations
    (routes, delays, regulations).
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.graph = nx.MultiDiGraph()
    
    def add_entity(self, entity: Entity) -> None:
        """Add entity to knowledge graph."""
        self.entities[entity.id] = entity
        self.graph.add_node(
            entity.id,
            type=entity.type,
            **entity.properties
        )
    
    def add_relation(self, relation: Relation) -> None:
        """Add relation to knowledge graph."""
        self.relations.append(relation)
        self.graph.add_edge(
            relation.subject,
            relation.object,
            type=relation.predicate.value,
            confidence=relation.confidence
        )
    
    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None
    ) -> List[str]:
        """Get neighboring entities."""
        neighbors = []
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            if relation_type is None or data['type'] == relation_type.value:
                neighbors.append(target)
        return neighbors
    
    def query_path(
        self,
        start: str,
        end: str,
        max_length: int = 5
    ) -> List[List[str]]:
        """Find paths between entities."""
        try:
            paths = list(nx.all_simple_paths(
                self.graph,
                start,
                end,
                cutoff=max_length
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def to_triples(self) -> List[Tuple[str, str, str]]:
        """Convert to RDF-like triples."""
        triples = []
        for rel in self.relations:
            triples.append((rel.subject, rel.predicate.value, rel.object))
        return triples


class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for learning entity embeddings.
    
    Uses message passing to propagate information through
    the knowledge graph.
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        
        # Relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            nn.Linear(embedding_dim * 2, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.update_layers = nn.ModuleList([
            nn.Linear(hidden_dim, embedding_dim)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        entity_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            entity_ids: Entity IDs [num_entities]
            edge_index: Edge indices [2, num_edges]
            edge_types: Edge types [num_edges]
        
        Returns:
            Updated entity embeddings [num_entities, embedding_dim]
        """
        # Initial embeddings
        x = self.entity_embeddings(entity_ids)
        
        # Message passing
        for message_layer, update_layer, norm in zip(
            self.message_layers,
            self.update_layers,
            self.layer_norms
        ):
            # Gather source and target embeddings
            source = x[edge_index[0]]
            target = x[edge_index[1]]
            
            # Get relation embeddings
            relations = self.relation_embeddings(edge_types)
            
            # Compute messages
            messages = torch.cat([source, relations], dim=-1)
            messages = F.relu(message_layer(messages))
            
            # Aggregate messages (sum) — messages are hidden_dim wide
            num_nodes = x.size(0)
            aggregated = torch.zeros(num_nodes, messages.size(-1), device=x.device)
            aggregated.index_add_(0, edge_index[1], messages)
            
            # Update embeddings (hidden_dim → embedding_dim)
            x_new = update_layer(aggregated)
            x = norm(x + x_new)  # Residual connection
        
        return x


class LogicEngine:
    """
    Logic engine for symbolic reasoning.
    
    Implements Prolog-like inference with forward chaining
    and backward chaining.
    """
    
    def __init__(self):
        self.rules: List[Rule] = []
        self.facts: Set[str] = set()
    
    def add_rule(self, rule: Rule) -> None:
        """Add logical rule."""
        self.rules.append(rule)
    
    def add_fact(self, fact: str) -> None:
        """Add fact to knowledge base."""
        self.facts.add(fact)
    
    def forward_chain(self, max_iterations: int = 100) -> Set[str]:
        """
        Forward chaining inference.
        
        Repeatedly apply rules to derive new facts.
        """
        derived = self.facts.copy()
        
        for _ in range(max_iterations):
            new_facts = set()
            
            for rule in self.rules:
                # Check if all body conditions are satisfied
                if all(self._match(cond, derived) for cond in rule.body):
                    # Derive head
                    new_facts.add(rule.head)
            
            # Check for convergence
            if new_facts.issubset(derived):
                break
            
            derived.update(new_facts)
        
        return derived
    
    def backward_chain(self, goal: str, depth: int = 0, max_depth: int = 10) -> bool:
        """
        Backward chaining inference.
        
        Try to prove goal by working backwards from rules.
        """
        if depth > max_depth:
            return False
        
        # Check if goal is a fact
        if goal in self.facts:
            return True
        
        # Try to prove using rules
        for rule in self.rules:
            if self._match(rule.head, {goal}):
                # Try to prove all body conditions
                if all(
                    self.backward_chain(cond, depth + 1, max_depth)
                    for cond in rule.body
                ):
                    return True
        
        return False
    
    def _match(self, pattern: str, facts: Set[str]) -> bool:
        """Check if pattern matches any fact."""
        # Simple string matching (can be extended with unification)
        for fact in facts:
            if pattern == fact or self._unify(pattern, fact):
                return True
        return False
    
    def _unify(self, pattern: str, fact: str) -> bool:
        """Simple unification (can be extended)."""
        # Extract variables (start with ?)
        pattern_parts = pattern.split()
        fact_parts = fact.split()
        
        if len(pattern_parts) != len(fact_parts):
            return False
        
        bindings = {}
        for p, f in zip(pattern_parts, fact_parts):
            if p.startswith('?'):
                if p in bindings:
                    if bindings[p] != f:
                        return False
                else:
                    bindings[p] = f
            elif p != f:
                return False
        
        return True


class DifferentiableLogic(nn.Module):
    """
    Differentiable logic layer.
    
    Implements fuzzy logic operations that are differentiable,
    allowing backpropagation through logical reasoning.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def fuzzy_and(self, *tensors: torch.Tensor) -> torch.Tensor:
        """Fuzzy AND (product t-norm)."""
        result = tensors[0]
        for t in tensors[1:]:
            result = result * t
        return result
    
    def fuzzy_or(self, *tensors: torch.Tensor) -> torch.Tensor:
        """Fuzzy OR (probabilistic sum)."""
        result = tensors[0]
        for t in tensors[1:]:
            result = result + t - result * t
        return result
    
    def fuzzy_not(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fuzzy NOT."""
        return 1.0 - tensor
    
    def fuzzy_implies(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuzzy implication (Lukasiewicz)."""
        return torch.clamp(1.0 - a + b, 0.0, 1.0)
    
    def soft_unification(
        self,
        pattern_emb: torch.Tensor,
        fact_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft unification using embedding similarity.
        
        Returns confidence that pattern matches fact.
        """
        similarity = F.cosine_similarity(pattern_emb, fact_emb, dim=-1)
        confidence = torch.sigmoid(similarity / self.temperature)
        return confidence


class NeuroSymbolicReasoner(nn.Module):
    """
    Complete Neuro-Symbolic Reasoning System.
    
    Combines:
    - Neural: GNN for learning entity embeddings
    - Symbolic: Logic engine for rule-based reasoning
    - Bridge: Differentiable logic for end-to-end training
    """
    
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3
    ):
        super().__init__()
        self.kg = knowledge_graph
        self.embedding_dim = embedding_dim
        
        # Build entity and relation vocabularies
        self.entity_to_id = {e: i for i, e in enumerate(knowledge_graph.entities.keys())}
        self.id_to_entity = {i: e for e, i in self.entity_to_id.items()}
        
        relation_types = set(rel.predicate for rel in knowledge_graph.relations)
        self.relation_to_id = {r: i for i, r in enumerate(relation_types)}
        
        # Neural component: GNN
        self.gnn = GraphNeuralNetwork(
            num_entities=len(self.entity_to_id),
            num_relations=len(self.relation_to_id),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers
        )
        
        # Symbolic component: Logic engine
        self.logic_engine = LogicEngine()
        
        # Differentiable bridge
        self.diff_logic = DifferentiableLogic(temperature=1.0)
        
        # Rule embeddings (for soft rule matching)
        self.rule_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def add_rule(self, rule: Rule) -> None:
        """Add symbolic rule."""
        self.logic_engine.add_rule(rule)
    
    def add_fact(self, fact: str) -> None:
        """Add symbolic fact."""
        self.logic_engine.add_fact(fact)
    
    def encode_graph(self) -> torch.Tensor:
        """Encode knowledge graph with GNN."""
        # Prepare graph data
        entity_ids = torch.arange(len(self.entity_to_id))
        
        edge_list = []
        edge_types = []
        for rel in self.kg.relations:
            src_id = self.entity_to_id[rel.subject]
            tgt_id = self.entity_to_id[rel.object]
            rel_id = self.relation_to_id[rel.predicate]
            
            edge_list.append([src_id, tgt_id])
            edge_types.append(rel_id)
        
        edge_index = torch.tensor(edge_list).t()
        edge_types = torch.tensor(edge_types)
        
        # Encode with GNN
        embeddings = self.gnn(entity_ids, edge_index, edge_types)
        
        return embeddings
    
    def query(
        self,
        query: str,
        use_neural: bool = True,
        use_symbolic: bool = True
    ) -> Dict[str, Any]:
        """
        Answer query using neuro-symbolic reasoning.
        
        Args:
            query: Query string
            use_neural: Use neural reasoning
            use_symbolic: Use symbolic reasoning
        
        Returns:
            Dictionary with answer and confidence
        """
        results = {}
        
        # Symbolic reasoning
        if use_symbolic:
            symbolic_answer = self.logic_engine.backward_chain(query)
            results['symbolic'] = symbolic_answer
        
        # Neural reasoning
        if use_neural:
            embeddings = self.encode_graph()
            # Query embedding (simplified - would use query encoder)
            query_emb = embeddings.mean(dim=0, keepdim=True)
            
            # Find most similar entities
            similarities = F.cosine_similarity(
                query_emb,
                embeddings,
                dim=-1
            )
            top_k = torch.topk(similarities, k=5)
            
            neural_answer = [
                (self.id_to_entity[idx.item()], score.item())
                for idx, score in zip(top_k.indices, top_k.values)
            ]
            results['neural'] = neural_answer
        
        # Combine (simple voting - can be more sophisticated)
        if use_neural and use_symbolic:
            confidence = 0.5 if results['symbolic'] else 0.0
            if results['neural']:
                confidence += 0.5 * results['neural'][0][1]
            results['combined_confidence'] = confidence
        
        return results
    
    def explain(self, query: str) -> List[str]:
        """
        Explain reasoning process.
        
        Returns chain of rules and facts used.
        """
        explanation = []
        
        # Trace backward chaining
        if self.logic_engine.backward_chain(query):
            explanation.append(f"Goal: {query}")
            
            # Find applicable rules
            for rule in self.logic_engine.rules:
                if query in rule.head:
                    explanation.append(f"Applied rule: {rule}")
                    for cond in rule.body:
                        if cond in self.logic_engine.facts:
                            explanation.append(f"  Fact: {cond}")
        
        return explanation


# Example usage and aviation rules
if __name__ == "__main__":
    # Create knowledge graph
    kg = KnowledgeGraph()
    
    # Add entities
    kg.add_entity(Entity("FL001", "flight", {"callsign": "AA123"}))
    kg.add_entity(Entity("JFK", "airport", {"name": "JFK Airport"}))
    kg.add_entity(Entity("LAX", "airport", {"name": "LAX Airport"}))
    kg.add_entity(Entity("AA", "airline", {"name": "American Airlines"}))
    kg.add_entity(Entity("weather_bad", "condition", {"severity": "high"}))
    
    # Add relations
    kg.add_relation(Relation("FL001", RelationType.DEPARTS_FROM, "JFK"))
    kg.add_relation(Relation("FL001", RelationType.ARRIVES_AT, "LAX"))
    kg.add_relation(Relation("FL001", RelationType.OPERATED_BY, "AA"))
    kg.add_relation(Relation("FL001", RelationType.HAS_DELAY, "weather_bad"))
    
    # Create neuro-symbolic reasoner
    reasoner = NeuroSymbolicReasoner(
        knowledge_graph=kg,
        embedding_dim=64,
        hidden_dim=128,
        num_gnn_layers=2
    )
    
    # Add aviation safety rules
    reasoner.add_rule(Rule(
        head="safe_to_fly(FL001)",
        body=["weather_good", "aircraft_maintained", "pilot_certified"]
    ))
    
    reasoner.add_rule(Rule(
        head="delay_expected(FL001)",
        body=["has_delay(FL001, weather_bad)"]
    ))
    
    reasoner.add_rule(Rule(
        head="violates_regulation(FL001)",
        body=["flies_in_restricted_airspace(FL001)"]
    ))
    
    # Add facts
    reasoner.add_fact("has_delay(FL001, weather_bad)")
    reasoner.add_fact("aircraft_maintained")
    reasoner.add_fact("pilot_certified")
    
    # Query
    print("=== Neuro-Symbolic Reasoning ===")
    
    query1 = "delay_expected(FL001)"
    result1 = reasoner.query(query1)
    print(f"\nQuery: {query1}")
    print(f"Symbolic answer: {result1.get('symbolic', False)}")
    print(f"Explanation: {reasoner.explain(query1)}")
    
    query2 = "safe_to_fly(FL001)"
    result2 = reasoner.query(query2)
    print(f"\nQuery: {query2}")
    print(f"Symbolic answer: {result2.get('symbolic', False)}")
    
    # Encode graph
    embeddings = reasoner.encode_graph()
    print(f"\nEntity embeddings shape: {embeddings.shape}")
    
    # Knowledge graph statistics
    print(f"\nKnowledge Graph Statistics:")
    print(f"  Entities: {len(kg.entities)}")
    print(f"  Relations: {len(kg.relations)}")
    print(f"  Rules: {len(reasoner.logic_engine.rules)}")
    print(f"  Facts: {len(reasoner.logic_engine.facts)}")
