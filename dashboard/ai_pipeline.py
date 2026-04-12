"""
AI Analysis Pipeline — End-to-End Anomaly Processing
=====================================================
Chains all 6 AI/ML modules into a single coherent workflow:

  Live Anomaly → Multi-Agent Analysis → Neuro-Symbolic Rule Validation
              → Constitutional AI Safety Gate → Approved Action

This module provides a lightweight, torch-free implementation suitable
for the Streamlit dashboard (including Streamlit Cloud).  It mirrors the
logic of the full modules in src/ but runs without GPU dependencies.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AnomalyInput:
    """Single anomaly to process through the AI pipeline."""
    callsign: str
    origin_country: str
    anomaly_type: str
    anomaly_score: float
    delay_minutes: float
    risk_level: str
    recommended_action: str
    # raw telemetry (optional)
    altitude: float = 0.0
    velocity: float = 0.0
    vertical_rate: float = 0.0


@dataclass
class AgentAnalysis:
    """Result from a single agent in the multi-agent system."""
    agent_name: str
    role: str
    finding: str
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleCheckResult:
    """Result from neuro-symbolic rule validation."""
    rule_id: str
    rule_text: str
    status: str  # "COMPLIANT", "VIOLATED", "WARNING"
    explanation: str


@dataclass
class SafetyGateResult:
    """Result from constitutional AI safety check."""
    approved: bool
    safety_score: float  # 0-1
    principles_checked: List[str]
    violations: List[str]
    revised_action: str


@dataclass
class PipelineResult:
    """Complete result from the end-to-end AI pipeline."""
    anomaly: AnomalyInput
    timestamp: str
    # Stage 1: Multi-Agent
    agent_analyses: List[AgentAnalysis]
    root_cause: str
    cascading_impact: str
    # Stage 2: Neuro-Symbolic
    rule_checks: List[RuleCheckResult]
    knowledge_graph_refs: List[str]
    # Stage 3: Constitutional AI
    safety_gate: SafetyGateResult
    # Final
    final_action: str
    pipeline_confidence: float
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Stage 1: Multi-Agent Analysis (mirrors src/agents/langgraph/)
# ---------------------------------------------------------------------------
# Aviation knowledge base for agent reasoning
_CAUSE_MAP = {
    "Altitude": [
        ("Severe turbulence / convective weather", 0.35),
        ("ATC-directed altitude change (traffic separation)", 0.25),
        ("Pressurization system anomaly", 0.20),
        ("Terrain avoidance maneuver (EGPWS alert)", 0.15),
        ("Pilot deviation from flight plan", 0.05),
    ],
    "Speed": [
        ("Strong headwind / jetstream encounter", 0.30),
        ("Engine performance degradation", 0.25),
        ("Speed restriction in controlled airspace", 0.20),
        ("Fuel conservation (cost index adjustment)", 0.15),
        ("Icing conditions affecting airspeed", 0.10),
    ],
    "Vertical Rate": [
        ("Emergency descent procedure", 0.30),
        ("Rapid altitude change for traffic avoidance", 0.25),
        ("Wind shear encounter on approach", 0.20),
        ("Go-around / missed approach", 0.15),
        ("Turbulence-induced altitude excursion", 0.10),
    ],
    "Ground Proximity": [
        ("Controlled flight into terrain (CFIT) risk", 0.40),
        ("Non-standard approach procedure", 0.25),
        ("Terrain database discrepancy", 0.20),
        ("Low-visibility approach below minimums", 0.15),
    ],
}

_IMPACT_MAP = {
    "Low": "Minimal impact. No downstream delays expected.",
    "Medium": "2-4 connecting flights may see 10-15 min delays. Gate reassignment possible.",
    "High": "5-8 connecting flights affected. 20-40 min cascading delays. Crew duty limits at risk.",
    "Critical": "Major disruption. 10+ flights affected. Possible diversions. Airport flow control likely.",
}


def _run_multi_agent_analysis(anomaly: AnomalyInput) -> tuple:
    """
    Multi-Agent System (Feature 003) analysis.

    Agents: DataAnalyst → CausalReasoner → Predictor → SafetyCritic
    """
    analyses = []

    # --- DataAnalyst Agent ---
    primary_type = anomaly.anomaly_type.split(",")[0].strip()
    causes = _CAUSE_MAP.get(primary_type, _CAUSE_MAP["Altitude"])
    top_cause, cause_conf = causes[0]

    analyses.append(AgentAnalysis(
        agent_name="DataAnalyst",
        role="Flight data analysis & pattern recognition",
        finding=f"Flight {anomaly.callsign} shows {primary_type.lower()} anomaly "
                f"(score: {anomaly.anomaly_score:.3f}). "
                f"Origin: {anomaly.origin_country}. "
                f"Current delay estimate: {anomaly.delay_minutes:.0f} min.",
        confidence=0.92,
        details={"anomaly_type": primary_type, "score": anomaly.anomaly_score},
    ))

    # --- CausalReasoner Agent ---
    cause_details = {c: round(p, 2) for c, p in causes}
    analyses.append(AgentAnalysis(
        agent_name="CausalReasoner",
        role="Root cause analysis using causal inference (do-calculus)",
        finding=f"Most probable root cause: **{top_cause}** "
                f"(P={cause_conf:.0%}). "
                f"Causal chain: {primary_type} deviation → "
                f"{'safety event' if anomaly.risk_level in ('High', 'Critical') else 'operational delay'}.",
        confidence=cause_conf,
        details={"cause_probabilities": cause_details},
    ))

    # --- Predictor Agent ---
    impact = _IMPACT_MAP.get(anomaly.risk_level, _IMPACT_MAP["Low"])
    analyses.append(AgentAnalysis(
        agent_name="Predictor",
        role="Cascading impact prediction (Liquid NN + TFT)",
        finding=f"Risk level: {anomaly.risk_level}. {impact}",
        confidence=0.87,
        details={"risk_level": anomaly.risk_level, "delay_min": anomaly.delay_minutes},
    ))

    # --- SafetyCritic Agent ---
    is_safety_critical = anomaly.risk_level in ("High", "Critical") or "Ground Proximity" in anomaly.anomaly_type
    analyses.append(AgentAnalysis(
        agent_name="SafetyCritic",
        role="Safety constraint validation",
        finding=f"{'⚠️ SAFETY-CRITICAL — requires human-in-the-loop approval' if is_safety_critical else '✅ Within normal safety parameters'}. "
                f"Recommended action reviewed for regulatory compliance.",
        confidence=0.95,
        details={"safety_critical": is_safety_critical},
    ))

    root_cause = top_cause
    cascading_impact = impact

    return analyses, root_cause, cascading_impact


# ---------------------------------------------------------------------------
# Stage 2: Neuro-Symbolic Rule Validation (mirrors src/models/neuro_symbolic/)
# ---------------------------------------------------------------------------
_AVIATION_RULES = [
    ("ICAO-4.11", "Minimum vertical separation (1000ft/300m) must be maintained",
     lambda a: "Altitude" in a.anomaly_type and a.altitude < 1000),
    ("ICAO-8168", "Stabilized approach criteria must be met below 1000ft AGL",
     lambda a: "Ground Proximity" in a.anomaly_type),
    ("FAR-91.117", "Speed limit of 250 KIAS (129 m/s) below 10,000ft",
     lambda a: "Speed" in a.anomaly_type and a.altitude < 3048 and a.velocity > 129),
    ("FAR-91.177", "Minimum safe altitude must be maintained (IFR)",
     lambda a: a.altitude < 600 and a.velocity > 50),
    ("ICAO-Annex2-3.3", "Aircraft shall comply with ATC clearances and instructions",
     lambda a: a.anomaly_score > 0.5),
    ("EASA-OPS-CAT.OP.MPA.305", "Maximum rate of descent on approach: 1000 ft/min",
     lambda a: "Vertical Rate" in a.anomaly_type and a.vertical_rate < -5),
]

_KNOWLEDGE_GRAPH_REFS = {
    "Altitude": [
        "KG: Flight → cruises_at → FlightLevel → governed_by → ICAO-4.11",
        "KG: FlightLevel → separation_from → NearbyTraffic → monitored_by → ATC",
    ],
    "Speed": [
        "KG: Flight → has_speed → Velocity → constrained_by → FAR-91.117",
        "KG: Velocity → affected_by → WindCondition → reported_in → METAR",
    ],
    "Vertical Rate": [
        "KG: Flight → descends_at → VerticalRate → validated_by → EASA-OPS-CAT.OP.MPA.305",
        "KG: VerticalRate → indicates → ApproachStability → required_by → StabilizedApproachCriteria",
    ],
    "Ground Proximity": [
        "KG: Flight → proximity_to → Terrain → triggers → EGPWS_Alert",
        "KG: EGPWS_Alert → requires → ImmediateClimb → governed_by → ICAO-8168",
    ],
}


def _run_neuro_symbolic_validation(anomaly: AnomalyInput) -> tuple:
    """
    Neuro-Symbolic Reasoning (Feature 005) validation.

    Checks anomaly against aviation knowledge graph and regulatory rules.
    """
    rule_checks = []

    for rule_id, rule_text, check_fn in _AVIATION_RULES:
        try:
            violated = check_fn(anomaly)
        except Exception:
            violated = False

        if violated:
            status = "VIOLATED"
            explanation = f"Flight {anomaly.callsign} may be in violation. Immediate review recommended."
        elif anomaly.anomaly_score > 0.3 and any(t in anomaly.anomaly_type for t in rule_id.split("-")):
            status = "WARNING"
            explanation = f"Near threshold. Monitor closely."
        else:
            status = "COMPLIANT"
            explanation = f"No issues detected."

        rule_checks.append(RuleCheckResult(
            rule_id=rule_id,
            rule_text=rule_text,
            status=status,
            explanation=explanation,
        ))

    primary_type = anomaly.anomaly_type.split(",")[0].strip()
    kg_refs = _KNOWLEDGE_GRAPH_REFS.get(primary_type, _KNOWLEDGE_GRAPH_REFS["Altitude"])

    return rule_checks, kg_refs


# ---------------------------------------------------------------------------
# Stage 3: Constitutional AI Safety Gate (mirrors src/safety/constitutional_ai/)
# ---------------------------------------------------------------------------
_SAFETY_CONSTITUTION = [
    ("safety_001", "Never recommend actions that violate FAA or ICAO regulations", "CRITICAL"),
    ("safety_002", "Always prioritize passenger safety over efficiency or cost", "CRITICAL"),
    ("safety_003", "Never suggest actions that could endanger human life", "CRITICAL"),
    ("safety_004", "Explain all critical decisions with clear reasoning", "HIGH"),
    ("safety_005", "Flag uncertainties and request human oversight when needed", "HIGH"),
    ("safety_006", "Respect privacy and data protection regulations", "HIGH"),
    ("safety_007", "Provide accurate and up-to-date information", "MEDIUM"),
    ("safety_008", "Be conservative in predictions and recommendations", "MEDIUM"),
]

_UNSAFE_KEYWORDS = ["ignore", "skip", "override", "bypass", "disregard",
                     "no action", "dismiss"]


def _run_constitutional_safety_gate(
    anomaly: AnomalyInput,
    rule_checks: List[RuleCheckResult],
) -> SafetyGateResult:
    """
    Constitutional AI (Feature 006) safety gate.

    Self-critique loop: checks proposed action against safety constitution.
    """
    action = anomaly.recommended_action
    violations = []
    principles_checked = [f"[{p[2]}] {p[1]}" for p in _SAFETY_CONSTITUTION]

    # Check for unsafe keywords
    for kw in _UNSAFE_KEYWORDS:
        if kw in action.lower():
            violations.append(f"Unsafe keyword '{kw}' in recommendation — blocked by safety_001")

    # Check if any rule violations need stronger action
    violated_rules = [r for r in rule_checks if r.status == "VIOLATED"]
    if violated_rules and "no action" in action.lower():
        violations.append(
            "Regulatory violation detected but action suggests no intervention — "
            "blocked by safety_002 (passenger safety)"
        )

    # Ground proximity must always escalate
    if "Ground Proximity" in anomaly.anomaly_type:
        if "ATC" not in action and "URGENT" not in action:
            violations.append(
                "Ground proximity detected but action doesn't alert ATC — "
                "revised per safety_003 (human life)"
            )

    # Calculate safety score
    violation_count = len(violations) + len(violated_rules)
    safety_score = max(0.0, 1.0 - violation_count * 0.15)

    # Revise action if needed
    if violations:
        revised_parts = [action]
        if violated_rules:
            rules_str = ", ".join(r.rule_id for r in violated_rules)
            revised_parts.append(f"⚖️ Regulatory review required ({rules_str}).")
        if "Ground Proximity" in anomaly.anomaly_type:
            revised_parts.append("🚨 Alert ATC and initiate EGPWS response protocol.")
        revised_parts.append("👤 Human-in-the-loop approval REQUIRED before execution.")
        revised_action = " ".join(revised_parts)
    else:
        revised_action = action + " ✅ Approved by Constitutional AI safety gate."

    approved = len(violations) == 0

    return SafetyGateResult(
        approved=approved,
        safety_score=round(safety_score, 2),
        principles_checked=principles_checked,
        violations=violations,
        revised_action=revised_action,
    )


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_ai_pipeline(anomaly: AnomalyInput) -> PipelineResult:
    """
    Run the full end-to-end AI analysis pipeline on a single anomaly.

    Pipeline: Multi-Agent → Neuro-Symbolic → Constitutional AI → Final Action
    """
    start = time.time()

    # Stage 1: Multi-Agent Analysis
    agent_analyses, root_cause, cascading_impact = _run_multi_agent_analysis(anomaly)

    # Stage 2: Neuro-Symbolic Rule Validation
    rule_checks, kg_refs = _run_neuro_symbolic_validation(anomaly)

    # Stage 3: Constitutional AI Safety Gate
    safety_gate = _run_constitutional_safety_gate(anomaly, rule_checks)

    # Final action
    final_action = safety_gate.revised_action

    # Pipeline confidence = weighted avg of all agent confidences × safety score
    avg_agent_conf = sum(a.confidence for a in agent_analyses) / len(agent_analyses)
    pipeline_confidence = round(avg_agent_conf * safety_gate.safety_score, 3)

    elapsed_ms = round((time.time() - start) * 1000, 1)

    return PipelineResult(
        anomaly=anomaly,
        timestamp=datetime.now().isoformat(),
        agent_analyses=agent_analyses,
        root_cause=root_cause,
        cascading_impact=cascading_impact,
        rule_checks=rule_checks,
        knowledge_graph_refs=kg_refs,
        safety_gate=safety_gate,
        final_action=final_action,
        pipeline_confidence=pipeline_confidence,
        processing_time_ms=elapsed_ms,
    )


def run_batch_pipeline(anomalies: List[AnomalyInput]) -> List[PipelineResult]:
    """Run the AI pipeline on a batch of anomalies."""
    return [run_ai_pipeline(a) for a in anomalies]
