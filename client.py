"""
TicketTriageEnv — typed HTTP client for the Ticket Triage environment.

Usage (async):
    async with TicketTriageEnv(base_url="http://localhost:7860") as env:
        result = await env.reset(task_id="task_easy")
        result = await env.step(TicketAction(
            action_type="classify",
            ticket_id="t001",
            category="account",
            priority="high",
        ))
        print(result.observation.current_ticket)

Usage (sync):
    with TicketTriageEnv(base_url="http://localhost:7860").sync() as env:
        env.reset(task_id="task_easy")
        result = env.step(TicketAction(...))
"""

from openenv.core.http_env_client import HTTPEnvClient

try:
    from .models import TicketAction, TicketObservation, TicketState
except ImportError:
    from models import TicketAction, TicketObservation, TicketState


class TicketTriageEnv(HTTPEnvClient[TicketAction, TicketObservation]):
    """
    Typed client for the Ticket Triage OpenEnv environment.
    Connects via HTTP to a running server (local or Hugging Face Space).
    """

    def _step_payload(self, action: TicketAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_state(self, payload: dict) -> TicketState:
        return TicketState(**payload)
