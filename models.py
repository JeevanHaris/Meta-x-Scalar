"""
Data models for the Ticket Triage Environment.

Inherits from openenv.core base classes so that openenv validate,
HTTPEnvServer, and StepResponse all work out-of-the-box.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


# ─── Enums as plain strings (openenv Action forbids extra fields,
#     so we use Literal-validated str fields rather than Python Enum) ───────────

VALID_ACTION_TYPES = {"classify", "prioritize", "respond", "escalate", "close", "tag"}
VALID_CATEGORIES   = {"billing", "technical", "account", "shipping", "product", "general"}
VALID_PRIORITIES   = {"critical", "high", "medium", "low"}
VALID_ESCALATIONS  = {"tier2", "billing_team", "engineering"}


class TicketAction(Action):
    """
    An action the agent can take on a support ticket.

    action_type choices:
        classify   — set category and/or priority
        prioritize — override priority only
        respond    — write a customer-facing response
        escalate   — route ticket to a specialist team
        close      — mark a resolved ticket as closed
        tag        — attach tags (SLA-breach flag, duplicate marker, etc.)
    """

    action_type: str = Field(
        ...,
        description="One of: classify | prioritize | respond | escalate | close | tag",
    )
    ticket_id: str = Field(..., description="ID of the target ticket, e.g. 't001'")

    # optional fields — only supply what the action_type requires
    priority: Optional[str] = Field(
        default=None,
        description="critical | high | medium | low",
    )
    category: Optional[str] = Field(
        default=None,
        description="billing | technical | account | shipping | product | general",
    )
    response_text: Optional[str] = Field(
        default=None,
        description="Customer-facing response text (for 'respond' actions)",
    )
    escalate_to: Optional[str] = Field(
        default=None,
        description="tier2 | billing_team | engineering",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Tags to attach, e.g. ['sla-breach', 'related:h003']",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional agent reasoning (not scored, logged for debugging)",
    )

    model_config = {"extra": "forbid"}


class TicketObservation(Observation):
    """
    What the agent sees after each reset() or step().

    Inherits `done`, `reward`, and `metadata` from the openenv Observation base.
    `reward` is set to the step reward; `done` is True when the episode ends.
    """

    # Current ticket the agent should act on
    current_ticket: Dict[str, Any] = Field(
        default_factory=dict,
        description="The ticket currently in focus (id, subject, body, customer info)",
    )
    # Full inbox so the agent can reason about all tickets at once
    inbox: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All tickets in the current task",
    )
    task_id: str = Field(default="", description="Current task identifier")
    instructions: str = Field(
        default="",
        description="Task-specific instructions describing what the agent must do",
    )
    step_count: int = Field(default=0, description="Steps taken so far")
    max_steps: int = Field(default=10, description="Maximum steps allowed")
    recent_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Last 5 actions taken (action_type, ticket_id, reward)",
    )
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward breakdown from the last step",
    )

    model_config = {"extra": "forbid"}


class TicketState(State):
    """
    Internal environment state — exposed via GET /state.

    Inherits `episode_id` and `step_count` from the openenv State base.
    """

    task_id: str = Field(default="")
    task_name: str = Field(default="")
    difficulty: str = Field(default="")
    max_steps: int = Field(default=10)
    cumulative_reward: float = Field(default=0.0)
    final_score: float = Field(default=0.0)
    grader_details: Dict[str, Any] = Field(default_factory=dict)
    episode_done: bool = Field(default=False)
    tickets_total: int = Field(default=0)
    tickets_classified: int = Field(default=0)
    tickets_responded: int = Field(default=0)
    tickets_closed: int = Field(default=0)
    ticket_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    model_config = {"extra": "allow"}
