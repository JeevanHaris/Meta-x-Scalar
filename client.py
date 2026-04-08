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

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import TicketAction, TicketObservation, TicketState
except ImportError:
    from models import TicketAction, TicketObservation, TicketState


class TicketTriageEnv(EnvClient[TicketAction, TicketObservation, TicketState]):
    """
    Typed client for the Ticket Triage OpenEnv environment.
    Connects via HTTP to a running server (local or Hugging Face Space).
    """

    def _step_payload(self, action: TicketAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult[TicketObservation]:
        # The payload from openenv-core 0.2.3 usually contains:
        # {"observation": {...}, "reward": ..., "done": ...}
        obs_data = payload.get("observation", payload)
        
        # Ensure we don't pass 'observation' itself into the constructor if it's still there
        if isinstance(obs_data, dict) and "observation" in obs_data and obs_data is payload:
             obs_data = obs_data["observation"]

        obs = TicketObservation(**obs_data)
        
        # Sync reward and done from the envelope payload if available
        reward = payload.get("reward", obs.reward)
        done = payload.get("done", obs.done)
        
        # Update the object as well
        obs.reward = reward
        obs.done = done
        
        return StepResult(observation=obs, reward=reward, done=done)

    def _parse_state(self, payload: dict) -> TicketState:
        return TicketState(**payload)
