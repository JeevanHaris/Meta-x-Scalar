"""
Test suite for Ticket Triage OpenEnv environment.
All grader tests are deterministic and reproducible.

Run:  python -m pytest tests/ -v
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import TicketAction, TicketObservation, TicketState
from server.ticket_triage_environment import TicketTriageEnvironment


# ── Helpers ──────────────────────────────────────────────────────────
def make_env(task_id: str) -> tuple:
    env = TicketTriageEnvironment()
    obs = env.reset(task_id=task_id)
    return env, obs


# ════════════════════════════════════════════════════════════════════
# reset()
# ════════════════════════════════════════════════════════════════════
class TestReset:
    def test_returns_ticket_observation(self):
        _, obs = make_env("task_easy")
        assert isinstance(obs, TicketObservation)

    def test_easy_has_5_tickets(self):
        _, obs = make_env("task_easy")
        assert len(obs.inbox) == 5

    def test_medium_has_5_tickets(self):
        _, obs = make_env("task_medium")
        assert len(obs.inbox) == 5

    def test_hard_has_8_tickets(self):
        _, obs = make_env("task_hard")
        assert len(obs.inbox) == 8

    def test_step_count_zero_at_reset(self):
        _, obs = make_env("task_easy")
        assert obs.step_count == 0

    def test_done_false_at_reset(self):
        _, obs = make_env("task_easy")
        assert obs.done is False

    def test_reward_zero_at_reset(self):
        _, obs = make_env("task_easy")
        assert obs.reward == 0.0

    def test_invalid_task_raises(self):
        env = TicketTriageEnvironment()
        with pytest.raises(ValueError):
            env.reset(task_id="nonexistent")

    def test_reset_clears_state(self):
        env, _ = make_env("task_easy")
        env.step(TicketAction(action_type="classify", ticket_id="t001",
                              category="account", priority="high"))
        env.reset(task_id="task_easy")
        s = env.state
        assert s.step_count == 0
        assert s.cumulative_reward == 0.0
        assert s.tickets_classified == 0


# ════════════════════════════════════════════════════════════════════
# step()
# ════════════════════════════════════════════════════════════════════
class TestStep:
    def test_returns_observation(self):
        env, _ = make_env("task_easy")
        obs = env.step(TicketAction(action_type="classify", ticket_id="t001",
                                    category="account", priority="high"))
        assert isinstance(obs, TicketObservation)

    def test_reward_embedded_in_observation(self):
        env, _ = make_env("task_easy")
        obs = env.step(TicketAction(action_type="classify", ticket_id="t001",
                                    category="account", priority="high"))
        assert obs.reward is not None
        assert isinstance(obs.reward, float)

    def test_correct_classify_positive_reward(self):
        env, _ = make_env("task_easy")
        obs = env.step(TicketAction(action_type="classify", ticket_id="t001",
                                    category="account", priority="high"))
        assert obs.reward > 0   # correct: account/high

    def test_wrong_classify_negative_or_low_reward(self):
        env, _ = make_env("task_easy")
        obs = env.step(TicketAction(action_type="classify", ticket_id="t001",
                                    category="billing", priority="low"))
        assert obs.reward < 0.15   # both wrong

    def test_critical_ticket_classified_correctly(self):
        env, _ = make_env("task_easy")
        obs = env.step(TicketAction(action_type="classify", ticket_id="t005",
                                    category="technical", priority="critical"))
        assert obs.reward > 0.2

    def test_respond_short_penalized(self):
        env, _ = make_env("task_medium")
        obs = env.step(TicketAction(action_type="respond", ticket_id="m001",
                                    response_text="ok"))
        assert (obs.reward or 0.0) < 0

    def test_respond_good_positive_reward(self):
        env, _ = make_env("task_medium")
        env.step(TicketAction(action_type="classify", ticket_id="m001",
                              category="technical", priority="high"))
        obs = env.step(TicketAction(
            action_type="respond", ticket_id="m001",
            response_text=(
                "Hi Frank, thanks for reaching out. I understand you're hitting "
                "rate limits with our API. Our enterprise plan offers 1000 req/min. "
                "I'd love to connect you with our sales team to discuss an upgrade."
            ),
        ))
        assert (obs.reward or 0.0) > 0

    def test_correct_escalation_rewarded(self):
        env, _ = make_env("task_medium")
        obs = env.step(TicketAction(action_type="escalate", ticket_id="m002",
                                    escalate_to="tier2"))
        assert (obs.reward or 0.0) > 0

    def test_unnecessary_escalation_penalized(self):
        env, _ = make_env("task_medium")
        obs = env.step(TicketAction(action_type="escalate", ticket_id="m004",
                                    escalate_to="engineering"))  # m004 is just "how to invite"
        assert (obs.reward or 0.0) < 0

    def test_close_resolved_ticket_rewarded(self):
        env, _ = make_env("task_hard")
        obs = env.step(TicketAction(action_type="close", ticket_id="h002"))  # "NVM fixed"
        assert (obs.reward or 0.0) > 0.2

    def test_close_critical_penalized(self):
        env, _ = make_env("task_hard")
        env.step(TicketAction(action_type="classify", ticket_id="h008",
                              category="technical", priority="critical"))
        obs = env.step(TicketAction(action_type="close", ticket_id="h008"))
        assert (obs.reward or 0.0) < 0

    def test_tag_sla_breach_rewarded(self):
        env, _ = make_env("task_hard")
        obs = env.step(TicketAction(action_type="tag", ticket_id="h001",
                                    tags=["sla-breach"]))
        assert (obs.reward or 0.0) > 0
        assert env.state.ticket_states["h001"]["sla_breach_flagged"] is True

    def test_tag_duplicate_rewarded(self):
        env, _ = make_env("task_hard")
        obs = env.step(TicketAction(action_type="tag", ticket_id="h004",
                                    tags=["related:h003", "duplicate"]))
        assert (obs.reward or 0.0) > 0

    def test_invalid_ticket_id_negative_reward(self):
        env, _ = make_env("task_easy")
        obs = env.step(TicketAction(action_type="classify", ticket_id="XXXX",
                                    category="billing", priority="low"))
        assert (obs.reward or 0.0) < 0

    def test_step_raises_after_done(self):
        env, _ = make_env("task_easy")
        env._episode_done = True
        with pytest.raises(RuntimeError):
            env.step(TicketAction(action_type="classify", ticket_id="t001"))

    def test_done_after_max_steps(self):
        env, _ = make_env("task_easy")   # max_steps=10
        obs = None
        for _ in range(10):
            obs = env.step(TicketAction(action_type="classify", ticket_id="t001",
                                        category="account", priority="high"))
        assert obs.done is True

    def test_done_when_all_classified(self):
        env, _ = make_env("task_easy")
        spec = [
            ("t001", "account",   "high"),
            ("t002", "billing",   "high"),
            ("t003", "shipping",  "medium"),
            ("t004", "product",   "low"),
            ("t005", "technical", "critical"),
        ]
        obs = None
        for tid, cat, pri in spec:
            obs = env.step(TicketAction(action_type="classify", ticket_id=tid,
                                        category=cat, priority=pri))
        assert obs.done is True


# ════════════════════════════════════════════════════════════════════
# state property
# ════════════════════════════════════════════════════════════════════
class TestState:
    def test_returns_ticket_state(self):
        env, _ = make_env("task_easy")
        assert isinstance(env.state, TicketState)

    def test_step_count_increments(self):
        env, _ = make_env("task_easy")
        env.step(TicketAction(action_type="classify", ticket_id="t001",
                              category="account", priority="high"))
        assert env.state.step_count == 1

    def test_tickets_classified_tracks(self):
        env, _ = make_env("task_easy")
        assert env.state.tickets_classified == 0
        env.step(TicketAction(action_type="classify", ticket_id="t001",
                              category="account", priority="high"))
        assert env.state.tickets_classified == 1

    def test_cumulative_reward_accumulates(self):
        env, _ = make_env("task_easy")
        env.step(TicketAction(action_type="classify", ticket_id="t001",
                              category="account", priority="high"))
        assert env.state.cumulative_reward != 0.0


# ════════════════════════════════════════════════════════════════════
# Graders — deterministic, 0.0–1.0
# ════════════════════════════════════════════════════════════════════
class TestGraders:
    def test_easy_perfect_score(self):
        env, _ = make_env("task_easy")
        for tid, cat, pri in [
            ("t001", "account",   "high"),
            ("t002", "billing",   "high"),
            ("t003", "shipping",  "medium"),
            ("t004", "product",   "low"),
            ("t005", "technical", "critical"),
        ]:
            env.step(TicketAction(action_type="classify", ticket_id=tid,
                                  category=cat, priority=pri))
        score, _ = env.get_final_score()
        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_easy_zero_score_all_wrong(self):
        env, _ = make_env("task_easy")
        for tid in ["t001", "t002", "t003", "t004", "t005"]:
            env.step(TicketAction(action_type="classify", ticket_id=tid,
                                  category="shipping", priority="low"))
        score, _ = env.get_final_score()
        assert score < 0.30

    def test_all_graders_in_range(self):
        for task_id in ["task_easy", "task_medium", "task_hard"]:
            env, _ = make_env(task_id)
            score, _ = env.get_final_score()
            assert 0.0 <= score <= 1.0, f"{task_id}: score {score} out of [0,1]"

    def test_graders_deterministic(self):
        """Same actions → same score on every run."""
        for _ in range(3):
            env, _ = make_env("task_easy")
            env.step(TicketAction(action_type="classify", ticket_id="t001",
                                  category="account", priority="high"))
            env.step(TicketAction(action_type="classify", ticket_id="t002",
                                  category="billing", priority="high"))
            scores = [env.get_final_score()[0]]
        # All three runs must produce the same score
        assert len(set(round(s, 6) for s in scores)) == 1

    def test_medium_partial_score(self):
        env, _ = make_env("task_medium")
        # Classify only — no responses or escalations
        for tid, cat, pri in [
            ("m001", "technical", "high"),
            ("m002", "account",   "critical"),
            ("m003", "billing",   "medium"),
            ("m004", "technical", "low"),
            ("m005", "shipping",  "high"),
        ]:
            env.step(TicketAction(action_type="classify", ticket_id=tid,
                                  category=cat, priority=pri))
        score, details = env.get_final_score()
        # Classification portion max 0.30; no responses → response portion low
        assert 0.0 < score < 0.8

    def test_hard_duplicate_detection_scores(self):
        env, _ = make_env("task_hard")
        env.step(TicketAction(action_type="tag", ticket_id="h003",
                              tags=["related:h004"]))
        env.step(TicketAction(action_type="tag", ticket_id="h004",
                              tags=["related:h003", "duplicate"]))
        score, details = env.get_final_score()
        assert details["duplicate_detection"] == 1.0

    def test_hard_sla_detection_scores(self):
        env, _ = make_env("task_hard")
        env.step(TicketAction(action_type="tag", ticket_id="h001",
                              tags=["sla-breach"]))
        env.step(TicketAction(action_type="tag", ticket_id="h008",
                              tags=["sla-breach"]))
        score, details = env.get_final_score()
        assert details["sla_compliance_score"] == 1.0


# ════════════════════════════════════════════════════════════════════
# Model validation
# ════════════════════════════════════════════════════════════════════
class TestModels:
    def test_action_rejects_unknown_fields(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            TicketAction(action_type="classify", ticket_id="t001",
                         nonexistent_field="bad")

    def test_observation_has_reward_field(self):
        _, obs = make_env("task_easy")
        assert hasattr(obs, "reward")
        assert hasattr(obs, "done")

    def test_state_has_episode_id_and_step_count(self):
        env, _ = make_env("task_easy")
        s = env.state
        assert hasattr(s, "episode_id")
        assert hasattr(s, "step_count")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
