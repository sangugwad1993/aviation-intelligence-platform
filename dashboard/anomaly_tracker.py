"""
Anomaly Resolution Tracker + Adaptive Threshold Engine
========================================================
Gap 2: Full audit trail — DETECTED → ACKNOWLEDGED → RESOLVED / ESCALATED / FALSE_POSITIVE
Gap 5: Closed-loop learning — operator feedback adjusts anomaly thresholds

Storage: data/anomaly_log.json (audit trail)
         data/adaptive_thresholds.json (learned thresholds)
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_LOG_FILE = _DATA_DIR / "anomaly_log.json"
_THRESHOLD_FILE = _DATA_DIR / "adaptive_thresholds.json"

# Valid status transitions
VALID_STATUSES = ["DETECTED", "ACKNOWLEDGED", "IN_PROGRESS", "RESOLVED", "ESCALATED", "FALSE_POSITIVE"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AnomalyEvent:
    """A single tracked anomaly event with full audit trail."""
    event_id: str
    callsign: str
    origin_country: str
    anomaly_type: str
    anomaly_score: float
    flight_phase: str
    risk_level: str
    recommended_action: str
    status: str = "DETECTED"
    assigned_to: str = ""
    resolution_notes: str = ""
    detected_at: str = ""
    acknowledged_at: str = ""
    resolved_at: str = ""
    time_to_ack_sec: float = 0.0
    feedback: str = ""  # "accept", "reject", "false_positive"


@dataclass
class ThresholdStats:
    """Feedback statistics for a single anomaly type within a flight phase."""
    total_detections: int = 0
    false_positives: int = 0
    accepted: int = 0
    rejected: int = 0
    fp_rate: float = 0.0
    # Sensitivity adjustment: positive = less sensitive, negative = more sensitive
    alt_sensitivity: float = 0.0
    vel_sensitivity: float = 0.0
    vr_sensitivity: float = 0.0
    last_updated: str = ""


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def _ensure_data_dir():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_log() -> List[dict]:
    _ensure_data_dir()
    if _LOG_FILE.exists():
        try:
            with open(_LOG_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_log(events: List[dict]):
    _ensure_data_dir()
    # Keep only last 500 events to avoid unbounded growth
    with open(_LOG_FILE, "w") as f:
        json.dump(events[-500:], f, indent=2)


def _load_thresholds() -> Dict[str, dict]:
    _ensure_data_dir()
    if _THRESHOLD_FILE.exists():
        try:
            with open(_THRESHOLD_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_thresholds(data: dict):
    _ensure_data_dir()
    with open(_THRESHOLD_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Event ID generation
# ---------------------------------------------------------------------------
def _make_event_id(callsign: str, anomaly_type: str) -> str:
    """Deterministic ID from callsign + type + current minute (deduplication window)."""
    ts_min = datetime.now().strftime("%Y%m%d%H%M")
    raw = f"{callsign}:{anomaly_type}:{ts_min}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12].upper()


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------
def log_anomaly(
    callsign: str,
    origin_country: str,
    anomaly_type: str,
    anomaly_score: float,
    flight_phase: str,
    risk_level: str,
    recommended_action: str,
    assigned_to: str = "",
) -> AnomalyEvent:
    """Log a new anomaly event (status: DETECTED). Deduplicates within same minute."""
    events = _load_log()
    event_id = _make_event_id(callsign, anomaly_type)

    # Check for duplicate
    for e in events:
        if e.get("event_id") == event_id:
            return AnomalyEvent(**e)

    event = AnomalyEvent(
        event_id=event_id,
        callsign=callsign,
        origin_country=origin_country,
        anomaly_type=anomaly_type,
        anomaly_score=anomaly_score,
        flight_phase=flight_phase,
        risk_level=risk_level,
        recommended_action=recommended_action,
        status="DETECTED",
        assigned_to=assigned_to,
        detected_at=datetime.now().isoformat(),
    )
    events.append(asdict(event))
    _save_log(events)
    return event


def acknowledge_anomaly(event_id: str, operator: str) -> Optional[dict]:
    """Move anomaly from DETECTED → ACKNOWLEDGED."""
    events = _load_log()
    for e in events:
        if e["event_id"] == event_id and e["status"] == "DETECTED":
            e["status"] = "ACKNOWLEDGED"
            e["assigned_to"] = operator
            e["acknowledged_at"] = datetime.now().isoformat()
            # Calculate time to acknowledge
            det = datetime.fromisoformat(e["detected_at"])
            ack = datetime.fromisoformat(e["acknowledged_at"])
            e["time_to_ack_sec"] = round((ack - det).total_seconds(), 1)
            _save_log(events)
            return e
    return None


def resolve_anomaly(
    event_id: str,
    resolution: str,
    notes: str = "",
    feedback: str = "accept",
) -> Optional[dict]:
    """Move anomaly to terminal state (RESOLVED / ESCALATED / FALSE_POSITIVE).

    Also triggers adaptive threshold update.
    """
    valid_resolutions = {"RESOLVED", "ESCALATED", "FALSE_POSITIVE"}
    if resolution not in valid_resolutions:
        return None

    events = _load_log()
    target = None
    for e in events:
        if e["event_id"] == event_id and e["status"] in ("DETECTED", "ACKNOWLEDGED", "IN_PROGRESS"):
            e["status"] = resolution
            e["resolution_notes"] = notes
            e["resolved_at"] = datetime.now().isoformat()
            e["feedback"] = feedback
            target = e
            break

    if target is None:
        return None

    _save_log(events)

    # --- Trigger adaptive threshold update ---
    _update_adaptive_thresholds(target)

    return target


def get_event_log(limit: int = 50) -> List[dict]:
    """Get recent anomaly events."""
    events = _load_log()
    return events[-limit:]


def get_audit_stats() -> Dict[str, Any]:
    """Get aggregate statistics for the audit dashboard."""
    events = _load_log()
    if not events:
        return {
            "total_events": 0,
            "status_counts": {},
            "avg_ack_time_sec": 0,
            "fp_rate": 0,
            "resolution_rate": 0,
        }

    status_counts = {}
    ack_times = []
    fp_count = 0
    resolved_count = 0

    for e in events:
        s = e.get("status", "DETECTED")
        status_counts[s] = status_counts.get(s, 0) + 1
        if e.get("time_to_ack_sec", 0) > 0:
            ack_times.append(e["time_to_ack_sec"])
        if s == "FALSE_POSITIVE":
            fp_count += 1
        if s in ("RESOLVED", "ESCALATED", "FALSE_POSITIVE"):
            resolved_count += 1

    total = len(events)
    return {
        "total_events": total,
        "status_counts": status_counts,
        "avg_ack_time_sec": round(sum(ack_times) / len(ack_times), 1) if ack_times else 0,
        "fp_rate": round(fp_count / total * 100, 1) if total else 0,
        "resolution_rate": round(resolved_count / total * 100, 1) if total else 0,
    }


# ---------------------------------------------------------------------------
# Gap 5: Adaptive Threshold Engine
# ---------------------------------------------------------------------------
def _update_adaptive_thresholds(event: dict):
    """Recompute adaptive thresholds from all feedback in the log.

    Logic:
    - If false_positive_rate > 30% for a phase, increase threshold sensitivity
      (= widen acceptable range = fewer alerts)
    - If reject_rate > 30%, decrease sensitivity (= tighten range = catch more)
    - Sensitivity capped at ±0.5 (50% adjustment)
    """
    events = _load_log()
    thresholds = _load_thresholds()

    # Aggregate feedback by flight_phase
    phase_stats: Dict[str, Dict[str, int]] = {}
    for e in events:
        phase = e.get("flight_phase", "Cruise")
        if phase not in phase_stats:
            phase_stats[phase] = {"total": 0, "fp": 0, "accept": 0, "reject": 0}
        phase_stats[phase]["total"] += 1
        fb = e.get("feedback", "")
        if fb == "false_positive" or e.get("status") == "FALSE_POSITIVE":
            phase_stats[phase]["fp"] += 1
        elif fb == "accept" or e.get("status") == "RESOLVED":
            phase_stats[phase]["accept"] += 1
        elif fb == "reject" or e.get("status") == "ESCALATED":
            phase_stats[phase]["reject"] += 1

    # Compute sensitivity adjustments
    for phase, stats in phase_stats.items():
        total = stats["total"]
        if total < 3:
            continue  # Need minimum feedback before adjusting

        fp_rate = stats["fp"] / total
        reject_rate = stats["reject"] / total

        # FP rate high → widen thresholds (less sensitive)
        # Reject rate high → tighten thresholds (more sensitive)
        adjustment = 0.0
        if fp_rate > 0.3:
            adjustment = min(fp_rate - 0.1, 0.5)  # Widen by up to 50%
        elif reject_rate > 0.3:
            adjustment = -min(reject_rate - 0.1, 0.3)  # Tighten by up to 30%

        thresholds[phase] = {
            "alt_sensitivity": round(adjustment, 3),
            "vel_sensitivity": round(adjustment, 3),
            "vr_sensitivity": round(adjustment, 3),
            "fp_rate": round(fp_rate * 100, 1),
            "reject_rate": round(reject_rate * 100, 1),
            "total_feedback": total,
            "last_updated": datetime.now().isoformat(),
        }

    _save_thresholds(thresholds)


def get_adaptive_thresholds() -> Dict[str, Any]:
    """Get current adaptive thresholds with stats."""
    return _load_thresholds()
