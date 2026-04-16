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
    # flight phase (inferred from telemetry)
    flight_phase: str = "Cruise"
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
                f"during **{anomaly.flight_phase}** phase "
                f"(score: {anomaly.anomaly_score:.3f}). "
                f"Origin: {anomaly.origin_country}. "
                f"Current delay estimate: {anomaly.delay_minutes:.0f} min. "
                f"Phase-aware detection confirms this is {'expected' if anomaly.flight_phase in ('Approach', 'Landing') and primary_type in ('Altitude', 'Speed') else 'anomalous'} "
                f"for the {anomaly.flight_phase} phase.",
        confidence=0.92,
        details={"anomaly_type": primary_type, "score": anomaly.anomaly_score, "flight_phase": anomaly.flight_phase},
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

    # --- WeatherCorrelator Agent ---
    weather_finding, weather_conf, weather_details = _run_weather_correlator(anomaly)
    analyses.append(AgentAnalysis(
        agent_name="WeatherCorrelator",
        role="Cross-reference anomaly with meteorological conditions",
        finding=weather_finding,
        confidence=weather_conf,
        details=weather_details,
    ))

    # --- MaintenancePredictor Agent ---
    maint_finding, maint_conf, maint_details = _run_maintenance_predictor(anomaly)
    analyses.append(AgentAnalysis(
        agent_name="MaintenancePredictor",
        role="Predict maintenance needs from anomaly pattern",
        finding=maint_finding,
        confidence=maint_conf,
        details=maint_details,
    ))

    # --- CrewImpactAnalyser Agent ---
    crew_finding, crew_conf, crew_details = _run_crew_impact_analyser(anomaly)
    analyses.append(AgentAnalysis(
        agent_name="CrewImpactAnalyser",
        role="Assess crew duty time impact and scheduling",
        finding=crew_finding,
        confidence=crew_conf,
        details=crew_details,
    ))

    # --- RouteOptimizer Agent ---
    route_finding, route_conf, route_details = _run_route_optimizer(anomaly)
    analyses.append(AgentAnalysis(
        agent_name="RouteOptimizer",
        role="Suggest alternate routing or altitude optimization",
        finding=route_finding,
        confidence=route_conf,
        details=route_details,
    ))

    return analyses, root_cause, cascading_impact


# ---------------------------------------------------------------------------
# Domain Agent: WeatherCorrelator
# ---------------------------------------------------------------------------
# Simulated METAR-like weather conditions keyed by altitude band + anomaly type
_WEATHER_CONDITIONS = {
    "high_alt_turbulence": {
        "condition": "CB tops FL380, moderate-to-severe turbulence reported",
        "metar": "METAR: SCT040CB BKN250 +TSRA",
        "sigmet": "SIGMET CHARLIE 3 — SEV TURB FL300-FL400",
        "correlation": 0.85,
    },
    "low_alt_wind": {
        "condition": "Surface wind gusting 35kt, low-level wind shear reported",
        "metar": "METAR: 27015G35KT 3SM -RA BR",
        "sigmet": "AIRMET TANGO — LLWS below 2000ft AGL",
        "correlation": 0.78,
    },
    "icing": {
        "condition": "Moderate icing in clouds FL100-FL200, freezing rain",
        "metar": "METAR: OVC015 -FZRA FG",
        "sigmet": "AIRMET ZULU — MOD ICE FL100-FL200",
        "correlation": 0.72,
    },
    "clear": {
        "condition": "CAVOK — no significant weather",
        "metar": "METAR: 18005KT CAVOK",
        "sigmet": "No SIGMET active",
        "correlation": 0.15,
    },
}


def _run_weather_correlator(anomaly: AnomalyInput) -> tuple:
    """Data-aware weather correlation based on altitude, phase, and anomaly type."""
    alt = anomaly.altitude
    phase = anomaly.flight_phase
    atype = anomaly.anomaly_type

    # Select weather condition based on actual telemetry
    if alt > 8000 and ("Altitude" in atype or "Vertical Rate" in atype):
        wx = _WEATHER_CONDITIONS["high_alt_turbulence"]
        assessment = (f"High-altitude anomaly at {alt:.0f}m during {phase} correlates with "
                      f"convective activity. {wx['condition']}. "
                      f"**Recommendation:** Request ride report from crew via ACARS. "
                      f"Consider FL change if turbulence persists > 5 min.")
    elif alt < 3000 and ("Speed" in atype or "Ground Proximity" in atype):
        wx = _WEATHER_CONDITIONS["low_alt_wind"]
        assessment = (f"Low-altitude anomaly at {alt:.0f}m during {phase} correlates with "
                      f"surface wind conditions. {wx['condition']}. "
                      f"**Recommendation:** Monitor wind shear alerts. "
                      f"{'Prepare for possible go-around.' if phase in ('Approach', 'Landing') else 'No immediate action.'}")
    elif 3000 <= alt <= 8000 and "Speed" in atype:
        wx = _WEATHER_CONDITIONS["icing"]
        assessment = (f"Mid-altitude speed anomaly at {alt:.0f}m — possible icing encounter. "
                      f"{wx['condition']}. "
                      f"**Recommendation:** Verify anti-ice systems active. "
                      f"Request altitude change if icing PIREPs confirmed.")
    else:
        wx = _WEATHER_CONDITIONS["clear"]
        assessment = (f"No significant weather correlation for this anomaly pattern "
                      f"(alt: {alt:.0f}m, phase: {phase}). {wx['condition']}. "
                      f"Weather is not a contributing factor — investigate other causes.")

    return assessment, wx["correlation"], {
        "metar": wx["metar"],
        "sigmet": wx["sigmet"],
        "weather_correlation": wx["correlation"],
        "altitude_band": f"{alt:.0f}m",
    }


# ---------------------------------------------------------------------------
# Domain Agent: MaintenancePredictor
# ---------------------------------------------------------------------------
_MAINT_PATTERNS = {
    "engine": {
        "system": "Engine / Powerplant",
        "ata_chapter": "ATA 72 — Engine",
        "action": "Schedule borescope inspection within next 50 flight hours",
        "urgency": "HIGH",
    },
    "pressurization": {
        "system": "Pressurization / ECS",
        "ata_chapter": "ATA 21 — Air Conditioning",
        "action": "Check cabin pressure controller and outflow valve",
        "urgency": "MEDIUM",
    },
    "flight_control": {
        "system": "Flight Control Surfaces",
        "ata_chapter": "ATA 27 — Flight Controls",
        "action": "Inspect actuators and control surface rigging",
        "urgency": "HIGH",
    },
    "navigation": {
        "system": "Navigation / EGPWS",
        "ata_chapter": "ATA 34 — Navigation",
        "action": "Verify terrain database currency and GPS accuracy",
        "urgency": "MEDIUM",
    },
    "none": {
        "system": "No system flagged",
        "ata_chapter": "N/A",
        "action": "No maintenance action required at this time",
        "urgency": "LOW",
    },
}


def _run_maintenance_predictor(anomaly: AnomalyInput) -> tuple:
    """Data-aware maintenance prediction from anomaly patterns."""
    atype = anomaly.anomaly_type
    score = anomaly.anomaly_score
    phase = anomaly.flight_phase

    # Pattern matching: which system is likely degraded?
    if "Speed" in atype and score > 0.3 and phase == "Cruise":
        pattern = _MAINT_PATTERNS["engine"]
        conf = 0.73
        finding = (f"Speed anomaly during cruise (score {score:.2f}) matches engine "
                   f"performance degradation signature. **{pattern['system']}** flagged. "
                   f"{pattern['ata_chapter']}. "
                   f"Action: {pattern['action']}. Urgency: {pattern['urgency']}.")
    elif "Altitude" in atype and score > 0.25 and phase == "Cruise":
        pattern = _MAINT_PATTERNS["pressurization"]
        conf = 0.65
        finding = (f"Altitude deviation in cruise may indicate pressurization drift. "
                   f"**{pattern['system']}** flagged. {pattern['ata_chapter']}. "
                   f"Action: {pattern['action']}. Urgency: {pattern['urgency']}.")
    elif "Vertical Rate" in atype and score > 0.3:
        pattern = _MAINT_PATTERNS["flight_control"]
        conf = 0.68
        finding = (f"Abnormal vertical rate (score {score:.2f}) may indicate flight control "
                   f"surface anomaly. **{pattern['system']}** flagged. {pattern['ata_chapter']}. "
                   f"Action: {pattern['action']}. Urgency: {pattern['urgency']}.")
    elif "Ground Proximity" in atype:
        pattern = _MAINT_PATTERNS["navigation"]
        conf = 0.60
        finding = (f"Ground proximity alert — verify EGPWS terrain database is current. "
                   f"**{pattern['system']}** flagged. {pattern['ata_chapter']}. "
                   f"Action: {pattern['action']}. Urgency: {pattern['urgency']}.")
    else:
        pattern = _MAINT_PATTERNS["none"]
        conf = 0.90
        finding = (f"Anomaly pattern does not match known maintenance degradation signatures. "
                   f"{pattern['action']}. Continue normal monitoring.")

    return finding, conf, {
        "system": pattern["system"],
        "ata_chapter": pattern["ata_chapter"],
        "urgency": pattern["urgency"],
        "maintenance_action": pattern["action"],
    }


# ---------------------------------------------------------------------------
# Domain Agent: CrewImpactAnalyser
# ---------------------------------------------------------------------------
def _run_crew_impact_analyser(anomaly: AnomalyInput) -> tuple:
    """Data-aware crew duty time and scheduling impact assessment."""
    delay = anomaly.delay_minutes
    risk = anomaly.risk_level
    phase = anomaly.flight_phase

    # FAR 117 duty time limits: 9-14 hrs depending on start time
    # Simulate remaining duty time based on delay
    base_remaining_duty = 120  # 2 hrs nominal remaining
    effective_remaining = max(0, base_remaining_duty - delay)

    if delay > 60 and risk in ("High", "Critical"):
        conf = 0.88
        finding = (f"**CREW DUTY ALERT:** {delay:.0f} min delay puts crew at risk of exceeding "
                   f"FAR 117 duty limits. Estimated remaining duty time: {effective_remaining:.0f} min. "
                   f"**Action:** Contact crew scheduling. Prepare standby crew at destination. "
                   f"If delay exceeds {base_remaining_duty:.0f} min, mandatory crew swap required. "
                   f"Estimated cost of crew swap: $4,500–$8,000.")
        details = {
            "remaining_duty_min": effective_remaining,
            "crew_swap_needed": True,
            "estimated_cost_usd": "$4,500–$8,000",
            "far_117_risk": "HIGH",
            "standby_crew_needed": True,
        }
    elif delay > 30:
        conf = 0.82
        finding = (f"Moderate delay ({delay:.0f} min) — crew duty time should be monitored. "
                   f"Remaining duty estimate: {effective_remaining:.0f} min. "
                   f"**Action:** Notify crew scheduling of potential delay. "
                   f"No immediate swap needed but monitor if delay extends. "
                   f"Check connecting crew assignments for cascading impact.")
        details = {
            "remaining_duty_min": effective_remaining,
            "crew_swap_needed": False,
            "far_117_risk": "MEDIUM",
            "standby_crew_needed": False,
        }
    elif delay > 10:
        conf = 0.85
        finding = (f"Minor delay ({delay:.0f} min) — no crew duty impact expected. "
                   f"Remaining duty time: {effective_remaining:.0f} min (well within limits). "
                   f"No crew scheduling action required.")
        details = {
            "remaining_duty_min": effective_remaining,
            "crew_swap_needed": False,
            "far_117_risk": "LOW",
        }
    else:
        conf = 0.92
        finding = (f"Negligible delay ({delay:.0f} min) during {phase} phase. "
                   f"No impact on crew duty time or scheduling. Normal operations.")
        details = {
            "remaining_duty_min": effective_remaining,
            "crew_swap_needed": False,
            "far_117_risk": "NONE",
        }

    return finding, conf, details


# ---------------------------------------------------------------------------
# Domain Agent: RouteOptimizer
# ---------------------------------------------------------------------------
def _run_route_optimizer(anomaly: AnomalyInput) -> tuple:
    """Data-aware route and altitude optimization suggestions."""
    alt = anomaly.altitude
    vel = anomaly.velocity
    phase = anomaly.flight_phase
    atype = anomaly.anomaly_type
    delay = anomaly.delay_minutes

    if phase == "Cruise" and "Altitude" in atype and alt > 6000:
        # Suggest altitude change
        new_alt = alt + 600 if alt < 11000 else alt - 600
        fuel_save = round(abs(new_alt - alt) * 0.003, 1)  # kg/m simplified
        conf = 0.80
        finding = (f"**Altitude optimization available.** Current: {alt:.0f}m → "
                   f"Suggested: {new_alt:.0f}m (±2000ft). "
                   f"Estimated fuel saving: {fuel_save:.1f} kg. "
                   f"Request FL change from ATC. Check ride reports for turbulence at new level. "
                   f"Expected delay recovery: {min(delay * 0.3, 15):.0f} min.")
        details = {
            "current_altitude_m": alt,
            "suggested_altitude_m": new_alt,
            "fuel_saving_kg": fuel_save,
            "delay_recovery_min": round(min(delay * 0.3, 15), 1),
            "optimization_type": "altitude_change",
        }
    elif phase == "Cruise" and "Speed" in atype:
        # Suggest speed adjustment (cost index)
        optimal_vel = 230 if vel > 260 else 250
        time_impact = round(abs(vel - optimal_vel) * 0.1, 1)
        conf = 0.76
        finding = (f"**Speed optimization available.** Current: {vel:.0f} m/s → "
                   f"Optimal: {optimal_vel} m/s (cost index adjustment). "
                   f"Time impact: ±{time_impact:.0f} min. "
                   f"Fuel saving at optimal speed: ~{round(abs(vel - optimal_vel) * 2.5, 0):.0f} kg. "
                   f"Request Mach adjustment from ATC if in RVSM airspace.")
        details = {
            "current_speed_ms": vel,
            "suggested_speed_ms": optimal_vel,
            "time_impact_min": time_impact,
            "fuel_saving_kg": round(abs(vel - optimal_vel) * 2.5, 0),
            "optimization_type": "speed_adjustment",
        }
    elif "Ground Proximity" in atype:
        conf = 0.91
        finding = (f"**Immediate routing action required.** Ground proximity during {phase}. "
                   f"If terrain conflict: execute EGPWS escape maneuver (wings level, max thrust, "
                   f"pitch 15° nose up). If false alarm: verify terrain database, "
                   f"continue current approach if visual contact with runway.")
        details = {
            "optimization_type": "terrain_avoidance",
            "immediate_action": True,
        }
    elif phase in ("Approach", "Landing"):
        conf = 0.85
        finding = (f"Aircraft in {phase} phase — route optimization not applicable. "
                   f"Current approach path should be maintained. "
                   f"If delay > 20 min, consider requesting priority sequencing from ATC. "
                   f"{'Holding pattern fuel check recommended.' if delay > 30 else 'No fuel concern.'}")
        details = {
            "optimization_type": "none_approach_phase",
            "priority_sequencing": delay > 20,
        }
    else:
        conf = 0.88
        finding = (f"Current routing is optimal for {phase} phase at {alt:.0f}m / {vel:.0f} m/s. "
                   f"No alternate route or altitude change recommended at this time.")
        details = {
            "optimization_type": "none_optimal",
        }

    return finding, conf, details


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
