"""
Deterministic graders for each task.
Every grader returns (score: float in [0.0, 1.0], details: dict).
All graders are pure functions — given the same inputs they always
produce the same output.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple


def _norm(val: Any) -> str:
    """Normalise a value to a lowercase string for comparison."""
    if val is None:
        return ""
    return str(val).strip().lower()


# ─────────────────────────────────────────────
# EASY GRADER
# Scores: 0.5 × category_accuracy + 0.5 × priority_accuracy
# ─────────────────────────────────────────────
def grade_easy(
    expected_outcomes: Dict[str, Dict[str, Any]],
    ticket_states: Dict[str, Dict[str, Any]],
    actions_taken: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    total = len(expected_outcomes)
    cat_correct = 0
    pri_correct = 0
    details: Dict[str, Any] = {}

    for tid, exp in expected_outcomes.items():
        ts = ticket_states.get(tid, {})
        got_cat = _norm(ts.get("category"))
        got_pri = _norm(ts.get("priority"))
        exp_cat = _norm(exp.get("category", ""))
        exp_pri = _norm(exp.get("priority", ""))

        cc = got_cat == exp_cat
        pc = got_pri == exp_pri
        cat_correct += int(cc)
        pri_correct += int(pc)
        details[tid] = {
            "category_correct": cc,
            "priority_correct": pc,
            "expected": exp,
            "got": {"category": got_cat or None, "priority": got_pri or None},
        }

    cat_acc = cat_correct / total
    pri_acc = pri_correct / total
    score   = 0.5 * cat_acc + 0.5 * pri_acc

    return score, {
        "category_accuracy": round(cat_acc, 4),
        "priority_accuracy": round(pri_acc, 4),
        "tickets_classified": sum(1 for ts in ticket_states.values() if ts.get("category")),
        "ticket_details": details,
    }


# ─────────────────────────────────────────────
# MEDIUM GRADER
# Weights: classification 30%, response quality 40%, escalation 30%
# ─────────────────────────────────────────────
def _response_quality(
    response_text: str,
    customer_name: str,
    required_keywords: List[str],
) -> float:
    """Score a response 0-1 based on length, personalisation, keyword coverage."""
    text = (response_text or "").lower()
    if not text:
        return 0.0

    length_ok   = len(text) > 60
    name_used   = customer_name.split()[0].lower() in text
    kw_ratio    = (
        sum(1 for kw in required_keywords if kw.lower() in text) / len(required_keywords)
        if required_keywords else 1.0
    )
    return 0.2 * length_ok + 0.2 * name_used + 0.6 * kw_ratio


def grade_medium(
    expected_outcomes: Dict[str, Dict[str, Any]],
    ticket_states: Dict[str, Dict[str, Any]],
    actions_taken: List[Dict[str, Any]],
    tickets_by_id: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    total = len(expected_outcomes)
    cls_total = resp_total = esc_total = 0.0
    details: Dict[str, Any] = {}

    for tid, exp in expected_outcomes.items():
        ts      = ticket_states.get(tid, {})
        ticket  = tickets_by_id.get(tid, {})

        # — Classification —
        cat_ok  = _norm(ts.get("category")) == _norm(exp.get("category", ""))
        pri_ok  = _norm(ts.get("priority")) == _norm(exp.get("priority", ""))
        cls_s   = 0.5 * cat_ok + 0.5 * pri_ok
        cls_total += cls_s

        # — Response quality —
        if exp.get("response_required"):
            resp_s = _response_quality(
                ts.get("response_text") or "",
                ticket.get("customer_name", ""),
                exp.get("response_keywords", []),
            )
        else:
            resp_s = 1.0            # not required → full credit
        resp_total += resp_s

        # — Escalation —
        exp_esc = exp.get("escalation")
        got_esc = ts.get("escalation")
        if exp_esc is None:
            esc_s = 0.0 if got_esc else 1.0   # penalise unnecessary escalation
        else:
            esc_s = 1.0 if got_esc else 0.0
        esc_total += esc_s

        details[tid] = {
            "classification": round(cls_s, 4),
            "response_quality": round(resp_s, 4),
            "escalation_correct": esc_s == 1.0,
        }

    cls_acc  = cls_total  / total
    resp_acc = resp_total / total
    esc_acc  = esc_total  / total
    score    = 0.30 * cls_acc + 0.40 * resp_acc + 0.30 * esc_acc

    return score, {
        "classification_score": round(cls_acc,  4),
        "response_score":       round(resp_acc, 4),
        "escalation_score":     round(esc_acc,  4),
        "ticket_details": details,
    }


# ─────────────────────────────────────────────
# HARD GRADER
# Weights:
#   classification   20%
#   response quality 20%
#   escalation       15%
#   duplicate detect 15%
#   closure          10%
#   SLA flags        10%
#   efficiency        10%
# ─────────────────────────────────────────────
def grade_hard(
    expected_outcomes: Dict[str, Dict[str, Any]],
    ticket_states: Dict[str, Dict[str, Any]],
    actions_taken: List[Dict[str, Any]],
    tickets_by_id: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:

    # — Classification (skip tickets expected to be closed) —
    cls_items = [
        (tid, exp) for tid, exp in expected_outcomes.items()
        if exp.get("action") != "close"
    ]
    cls_scores = []
    for tid, exp in cls_items:
        ts = ticket_states.get(tid, {})
        cat_ok = _norm(ts.get("category")) == _norm(exp.get("category", ""))
        pri_ok = _norm(ts.get("priority")) == _norm(exp.get("priority", ""))
        cls_scores.append(0.5 * cat_ok + 0.5 * pri_ok)
    cls_score = sum(cls_scores) / max(len(cls_scores), 1)

    # — Response quality —
    resp_items = [(tid, exp) for tid, exp in expected_outcomes.items()
                  if exp.get("response_required")]
    resp_scores = []
    for tid, _ in resp_items:
        ts     = ticket_states.get(tid, {})
        ticket = tickets_by_id.get(tid, {})
        text   = ts.get("response_text") or ""
        length_ok = len(text) > 80
        name_used = ticket.get("customer_name", "").split()[0].lower() in text.lower()
        resp_scores.append(0.6 * length_ok + 0.4 * name_used)
    resp_score = sum(resp_scores) / max(len(resp_scores), 1)

    # — Escalation —
    esc_needed = [(tid, exp["escalation"]) for tid, exp in expected_outcomes.items()
                  if exp.get("escalation")]
    esc_hits   = sum(1 for tid, _ in esc_needed
                     if ticket_states.get(tid, {}).get("escalation"))
    esc_score  = esc_hits / max(len(esc_needed), 1)

    # — Duplicate detection: h003 ↔ h004 should each carry a related tag —
    def _has_related(ts: Dict, other_id: str) -> bool:
        tags = ts.get("tags", [])
        return (
            any(other_id in t for t in tags)
            or any("related" in t for t in tags)
            or ts.get("is_duplicate", False)
        )

    h003_ts = ticket_states.get("h003", {})
    h004_ts = ticket_states.get("h004", {})
    dup_score = float(
        _has_related(h003_ts, "h004") and _has_related(h004_ts, "h003")
    )

    # — Closure: h002 should be closed —
    closure_score = float(ticket_states.get("h002", {}).get("closed", False))

    # — SLA flags: h001 and h008 should be tagged sla-breach —
    def _sla_flagged(ts: Dict) -> bool:
        return ts.get("sla_breach_flagged", False)

    sla_score = 0.5 * _sla_flagged(ticket_states.get("h001", {})) \
              + 0.5 * _sla_flagged(ticket_states.get("h008", {}))

    # — Efficiency: optimal ~16-24 actions; penalise excess —
    n_actions   = len(actions_taken)
    efficiency  = max(0.0, 1.0 - max(0, n_actions - 24) / 10.0)

    score = (
        0.20 * cls_score
      + 0.20 * resp_score
      + 0.15 * esc_score
      + 0.15 * dup_score
      + 0.10 * closure_score
      + 0.10 * sla_score
      + 0.10 * efficiency
    )

    return score, {
        "classification_score":   round(cls_score,     4),
        "response_score":         round(resp_score,     4),
        "escalation_score":       round(esc_score,      4),
        "duplicate_detection":    round(dup_score,      4),
        "closure_score":          round(closure_score,  4),
        "sla_compliance_score":   round(sla_score,      4),
        "efficiency_score":       round(efficiency,     4),
        "total_actions":          n_actions,
    }


GRADERS = {
    "task_easy":   grade_easy,
    "task_medium": grade_medium,
    "task_hard":   grade_hard,
}
