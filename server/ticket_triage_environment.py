"""
TicketTriageEnvironment — the core logic for the Customer Support Ticket Triage env.

Implements the OpenEnv Environment ABC:
  reset()  → TicketObservation   (initial state)
  step()   → TicketObservation   (with .reward and .done set)
  state    → TicketState         (property, not method)

The reward is embedded in the returned observation, matching the real OpenEnv spec.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import TicketAction, TicketObservation, TicketState
    from ..tasks import ALL_TASKS, TaskDef
    from ..graders import GRADERS
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import TicketAction, TicketObservation, TicketState
    from tasks import ALL_TASKS, TaskDef
    from graders import GRADERS


PRIORITY_ORDER = ["low", "medium", "high", "critical"]


class TicketTriageEnvironment(Environment):
    """
    Customer Support Ticket Triage OpenEnv environment.

    Agents must classify tickets, write responses, escalate critical issues,
    detect duplicate tickets, close self-resolved tickets, and flag SLA breaches.

    Three tasks of increasing difficulty are available:
        task_easy   — classify 5 tickets
        task_medium — classify + respond + escalate 5 tickets
        task_hard   — full workflow on 8 tickets with SLA/duplicate detection
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._task: Optional[TaskDef]          = None
        self._ticket_states: Dict[str, Dict]   = {}
        self._actions_taken: List[Dict]        = []
        self._episode_id: str                  = str(uuid.uuid4())
        self._step_count: int                  = 0
        self._cumulative_reward: float         = 0.0
        self._episode_done: bool               = False

    # ─────────────────────────────────────────────────────────
    # OpenEnv API
    # ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "task_easy",
        **kwargs: Any,
    ) -> TicketObservation:
        """Reset the environment for the given task and return the initial observation."""
        task = ALL_TASKS.get(task_id)
        if task is None:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Available: {list(ALL_TASKS.keys())}"
            )

        self._task              = task
        self._episode_id        = episode_id or str(uuid.uuid4())
        self._step_count        = 0
        self._cumulative_reward = 0.0
        self._episode_done      = False
        self._actions_taken     = []
        self._ticket_states     = {
            t.id: {
                "category":          None,
                "priority":          None,
                "response_text":     None,
                "escalation":        None,
                "tags":              [],
                "closed":            False,
                "sla_breach_flagged": False,
                "is_duplicate":      False,
                "action_count":      0,
            }
            for t in task.tickets
        }

        return self._build_observation(reward=0.0, done=False, breakdown={})

    def step(self, action: TicketAction) -> TicketObservation:  # type: ignore[override]
        """Execute one action and return the resulting observation (with reward embedded)."""
        if self._episode_done:
            raise RuntimeError("Episode is done — call reset() to start a new episode.")
        if self._task is None:
            raise RuntimeError("No task loaded — call reset(task_id=...) first.")

        reward_value, breakdown, info = self._process_action(action)

        self._step_count        += 1
        self._cumulative_reward += reward_value

        # Record action for grader
        self._actions_taken.append({
            "step":         self._step_count,
            "action_type":  action.action_type,
            "ticket_id":    action.ticket_id,
            "priority":     action.priority,
            "category":     action.category,
            "has_response": bool(action.response_text),
            "escalate_to":  action.escalate_to,
            "tags":         action.tags,
        })

        # Check episode termination
        done = (
            self._step_count >= self._task.max_steps
        )

        # Final score bonus when done
        if done and not self._episode_done:
            self._episode_done = True
            final_score, _ = self._run_grader()
            bonus            = final_score * 0.5     # up to +0.5 for episode quality
            reward_value    += bonus
            breakdown["final_bonus"] = round(bonus, 4)

        reward_value = max(-1.0, min(1.0, reward_value))
        return self._build_observation(
            reward=reward_value,
            done=done,
            breakdown=breakdown,
        )

    @property
    def state(self) -> TicketState:
        """Return current internal state (exposed at GET /state)."""
        if self._task is None:
            return TicketState()
        ts = self._ticket_states
        return TicketState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task.task_id,
            task_name=self._task.name,
            difficulty=self._task.difficulty,
            max_steps=self._task.max_steps,
            cumulative_reward=round(self._cumulative_reward, 4),
            episode_done=self._episode_done,
            tickets_total=len(self._task.tickets),
            tickets_classified=sum(1 for s in ts.values() if s.get("category")),
            tickets_responded=sum(1 for s in ts.values() if s.get("response_text")),
            tickets_closed=sum(1 for s in ts.values() if s.get("closed")),
            ticket_states=ts,
        )

    # ─────────────────────────────────────────────────────────
    # Action processors
    # ─────────────────────────────────────────────────────────

    def _process_action(
        self, action: TicketAction
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """Dispatch action to the appropriate handler. Returns (reward, breakdown, info)."""
        task = self._task

        # Validate ticket ID
        valid_ids = {t.id for t in task.tickets}
        if action.ticket_id not in valid_ids:
            return -0.1, {"invalid_ticket": -0.1}, {"error": "invalid ticket_id"}

        ts   = self._ticket_states[action.ticket_id]
        exp  = task.expected_outcomes.get(action.ticket_id, {})

        # Per-ticket repetition guard
        ts["action_count"] += 1
        rep_penalty = -0.05 * max(0, ts["action_count"] - 5)

        dispatch = {
            "classify":   self._classify,
            "prioritize": self._prioritize,
            "respond":    self._respond,
            "escalate":   self._escalate,
            "close":      self._close,
            "tag":        self._tag,
        }
        handler = dispatch.get(action.action_type)
        if handler is None:
            return -0.05, {"unknown_action": -0.05}, {"error": f"Unknown action_type: {action.action_type}"}

        rv, bd, info = handler(action, ts, exp)
        if rep_penalty < 0:
            rv += rep_penalty
            bd["repetition_penalty"] = round(rep_penalty, 4)
        return rv, bd, info

    def _classify(self, action, ts, exp):
        rv, bd = 0.0, {}
        if action.category:
            ts["category"] = action.category
            exp_cat = exp.get("category", "")
            if exp_cat:
                if action.category == exp_cat:
                    rv += 0.15;  bd["correct_category"] = 0.15
                else:
                    rv -= 0.05;  bd["wrong_category"]   = -0.05
        if action.priority:
            ts["priority"] = action.priority
            exp_pri = exp.get("priority", "")
            if exp_pri:
                rv += self._priority_reward(action.priority, exp_pri, bd)
        return rv, bd, {}

    def _prioritize(self, action, ts, exp):
        bd = {}
        if action.priority:
            ts["priority"] = action.priority
            rv = self._priority_reward(action.priority, exp.get("priority", ""), bd)
        else:
            rv = 0.0
        return rv, bd, {}

    def _respond(self, action, ts, exp):
        text = (action.response_text or "").strip()
        if len(text) < 20:
            return -0.1, {"empty_response": -0.1}, {"error": "response too short"}
        ts["response_text"] = text
        priority = ts.get("priority", "")
        rv = 0.2 if priority in ("critical", "high") else (0.1 if exp.get("response_required") else 0.05)
        keywords = exp.get("response_keywords", [])
        if keywords:
            kw_hits = sum(1 for kw in keywords if kw.lower() in text.lower())
            rv += 0.10 * (kw_hits / len(keywords))
        return rv, {"response_reward": round(rv, 4)}, {}

    def _escalate(self, action, ts, exp):
        ts["escalation"] = action.escalate_to
        exp_esc = exp.get("escalation")
        if exp_esc:
            rv = 0.20;  bd = {"correct_escalation": 0.20}
        else:
            rv = -0.05; bd = {"unnecessary_escalation": -0.05}
        return rv, bd, {}

    def _close(self, action, ts, exp):
        ts["closed"] = True
        if exp.get("action") == "close":
            return 0.30, {"correct_closure": 0.30}, {}
        priority = ts.get("priority", "")
        pen = -0.30 if priority in ("critical", "high") else -0.10
        return pen, {"premature_closure": pen}, {}

    def _tag(self, action, ts, exp):
        tags = action.tags or []
        ts["tags"].extend(tags)
        rv, bd = 0.0, {}
        for t in tags:
            tl = t.lower()
            if "sla" in tl or "breach" in tl:
                if exp.get("sla_breach"):
                    ts["sla_breach_flagged"] = True
                    rv += 0.15;  bd["sla_flagged"] = 0.15
            if "related" in tl or "duplicate" in tl:
                if exp.get("is_duplicate") or exp.get("related_to"):
                    ts["is_duplicate"] = True
                    rv += 0.15;  bd["duplicate_flagged"] = 0.15
        return rv, bd, {}

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _priority_reward(self, got: str, expected: str, bd: Dict) -> float:
        if not expected:
            return 0.0
        if got == expected:
            bd["correct_priority"] = 0.15
            return 0.15
        try:
            diff = abs(PRIORITY_ORDER.index(got) - PRIORITY_ORDER.index(expected))
        except ValueError:
            diff = 2
        if diff == 1:
            bd["adjacent_priority"] = 0.05
            return 0.05
        bd["wrong_priority"] = -0.05
        return -0.05

    def _all_classified(self) -> bool:
        """True when every ticket is either classified or closed."""
        if not self._task:
            return False
        for t in self._task.tickets:
            ts = self._ticket_states.get(t.id, {})
            if not ts.get("category") and not ts.get("closed"):
                return False
        return True

    def _build_observation(
        self, reward: float, done: bool, breakdown: Dict[str, float]
    ) -> TicketObservation:
        task = self._task
        if task is None:
            return TicketObservation(
                reward=reward, done=done,
                instructions="No task loaded — call reset(task_id=...) first.",
            )

        current_ticket = task.tickets[
            min(self._step_count % len(task.tickets), len(task.tickets) - 1)
        ]
        recent = self._actions_taken[-5:]

        return TicketObservation(
            reward=reward,
            done=done,
            current_ticket=current_ticket.to_dict(),
            inbox=[t.to_dict() for t in task.tickets],
            task_id=task.task_id,
            instructions=task.description,
            step_count=self._step_count,
            max_steps=task.max_steps,
            recent_history=recent,
            reward_breakdown=breakdown,
            metadata={
                "episode_id":         self._episode_id,
                "cumulative_reward":  round(self._cumulative_reward, 4),
            },
        )

    def _run_grader(self) -> Tuple[float, Dict[str, Any]]:
        task    = self._task
        grader  = GRADERS.get(task.task_id)
        if grader is None:
            return 0.0, {"error": "no grader"}
        tickets_by_id = {t.id: t.to_dict() for t in task.tickets}
        if task.task_id == "task_easy":
            return grader(task.expected_outcomes, self._ticket_states, self._actions_taken)
        return grader(task.expected_outcomes, self._ticket_states, self._actions_taken, tickets_by_id)

    def get_final_score(self) -> Tuple[float, Dict[str, Any]]:
        """Run the grader and return final score. Safe to call at any point."""
        if self._task is None:
            return 0.0, {}
        return self._run_grader()

    def list_tasks(self) -> Dict[str, Any]:
        return {
            tid: {
                "name": t.name,
                "difficulty": t.difficulty,
                "description": t.description,
                "num_tickets": len(t.tickets),
                "max_steps":   t.max_steps,
            }
            for tid, t in ALL_TASKS.items()
        }
