"""
inference.py — Baseline inference script for Ticket Triage OpenEnv.

Runs an LLM agent directly against the environment (no HTTP server needed).
Uses the OpenAI client API.

Environment variables:
    API_BASE_URL   — LLM API endpoint   (default: https://api.openai.com/v1)
    MODEL_NAME     — model identifier   (default: gpt-4o-mini)
    HF_TOKEN       — API key (also accepts OPENAI_API_KEY)

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="sk-..."
    python inference.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

if not HF_TOKEN:
    print("ERROR: Set HF_TOKEN or OPENAI_API_KEY.")
    sys.exit(1)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ── Import env directly (no HTTP server required for inference) ──────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.ticket_triage_environment import TicketTriageEnvironment
from models import TicketAction


# ── Prompts ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert customer support triage agent.
Process support tickets by outputting a single JSON action object.

Available actions:
  classify   → {"action_type":"classify","ticket_id":"...","category":"billing|technical|account|shipping|product|general","priority":"critical|high|medium|low"}
  respond    → {"action_type":"respond","ticket_id":"...","response_text":"<full response to customer>"}
  escalate   → {"action_type":"escalate","ticket_id":"...","escalate_to":"tier2|billing_team|engineering"}
  close      → {"action_type":"close","ticket_id":"..."}
  tag        → {"action_type":"tag","ticket_id":"...","tags":["sla-breach","related:h003"]}

Rules:
- For EACH ticket, you must eventually CLASSIFY it (category + priority).
- Respond to all HIGH and CRITICAL tickets. Use the customer's first name.
- Escalate CRITICAL tickets and enterprise security/billing fraud cases.
- Close tickets the customer has already self-resolved (e.g. "figured it out").
- Tag SLA-breach risk: enterprise tickets created before 08:00 (SLA >4h).
- Use 'related:TICKET_ID' tags to link duplicate/related tickets (e.g. same user with same issue).
- IMPORTANT: Take multiple steps for a single ticket if needed (e.g., classify first, then respond).

Output ONLY the JSON object. Nothing else."""


def _build_user_prompt(obs_dict: Dict[str, Any]) -> str:
    inbox_lines = "\n".join(
        f"  [{t['id']}] {t['subject'][:70]}  "
        f"(customer: {t['customer_name']}, tier: {t['customer_tier']}, "
        f"created: {t['created_at']})"
        for t in obs_dict.get("inbox", [])
    )
    ct = obs_dict.get("current_ticket", {})
    history = obs_dict.get("recent_history", [])
    hist_str = ""
    if history:
        hist_str = "\n\nRecent actions:\n" + "\n".join(
            f"  step {h['step']}: {h['action_type']} on {h['ticket_id']}"
            for h in history
        )

    return f"""TASK: {obs_dict.get('instructions','')}

INBOX ({len(obs_dict.get('inbox',[]))} tickets):
{inbox_lines}

CURRENT TICKET IN FOCUS:
  ID:       {ct.get('id','')}
  Subject:  {ct.get('subject','')}
  Customer: {ct.get('customer_name','')} ({ct.get('customer_tier','')} tier)
  Created:  {ct.get('created_at','')}
  Body:     {ct.get('body','')}
{hist_str}

Step {obs_dict.get('step_count',0)}/{obs_dict.get('max_steps',10)} — output one JSON action:"""


def _parse_action(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    for marker in ("```json", "```"):
        if marker in text:
            text = text.split(marker, 1)[1].split("```")[0].strip()
            break
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _make_action(d: Dict[str, Any]) -> Optional[TicketAction]:
    try:
        return TicketAction(**d)
    except Exception as e:
        print(f"  [WARN] Bad action dict {d}: {e}")
        return None


def run_task(task_id: str) -> Dict[str, Any]:
    print(f"\n{'='*65}")
    print(f"  TASK: {task_id}")
    print(f"{'='*65}")

    env = TicketTriageEnvironment()
    obs = env.reset(task_id=task_id)
    obs_dict = obs.model_dump()

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0.0
    step = 0
    done = False
    log: List[Dict[str, Any]] = []

    while not done and step < obs_dict["max_steps"]:
        user_msg = _build_user_prompt(obs_dict)
        messages.append({"role": "user", "content": user_msg})

        # Trim history to avoid TPM limits: Keep system prompt + last 6 messages
        if len(messages) > 7:
            messages = [messages[0]] + messages[-6:]

        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=512,
                temperature=0.1,
            )
            assistant_text = resp.choices[0].message.content
            messages.append({"role": "assistant", "content": assistant_text})
        except Exception as e:
            print(f"  [ERROR] LLM call failed at step {step}: {e}")
            break

        action_dict = _parse_action(assistant_text)
        if not action_dict:
            print(f"  [WARN] Step {step+1}: could not parse JSON from response")
            step += 1
            continue

        action = _make_action(action_dict)
        if not action:
            step += 1
            continue

        # Print compact progress
        print(
            f"  step {step+1:2d}: {action.action_type:10s} {action.ticket_id}"
            + (f" | cat={action.category}" if action.category else "")
            + (f" | pri={action.priority}"  if action.priority  else "")
            + (f" | esc={action.escalate_to}" if action.escalate_to else "")
        )

        try:
            obs = env.step(action)
            obs_dict = obs.model_dump()
            step_reward = obs.reward or 0.0
            total_reward += step_reward
            done = obs.done
            log.append({
                "step": step + 1,
                "action": action_dict,
                "reward": step_reward,
            })
        except RuntimeError as e:
            print(f"  [ENV] {e}")
            break

        step += 1
        time.sleep(0.25)   # polite rate limiting

    # Final score from grader
    final_score, details = env.get_final_score()
    state = env.state
    print(f"\n  Final grader score : {final_score:.4f}")
    print(f"  Tickets classified : {state.tickets_classified}/{state.tickets_total}")
    print(f"  Tickets responded  : {state.tickets_responded}/{state.tickets_total}")

    return {
        "task_id":         task_id,
        "steps_taken":     step,
        "total_reward":    round(total_reward, 4),
        "final_score":     round(final_score,  4),
        "grader_details":  details,
        "tickets_classified": state.tickets_classified,
        "tickets_responded":  state.tickets_responded,
        "tickets_total":      state.tickets_total,
        "action_log":      log,
    }


def main() -> None:
    print("\nTicketTriage OpenEnv — Baseline Inference")
    print(f"Model    : {MODEL_NAME}")
    print(f"API base : {API_BASE_URL}")

    results = {}
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        results[task_id] = run_task(task_id)

    print(f"\n{'='*65}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"{'Task':<22} {'Score':>8}  {'Steps':>6}  Classified  Responded")
    print(f"{'-'*65}")

    scores = []
    for tid, r in results.items():
        scores.append(r["final_score"])
        print(
            f"{tid:<22} {r['final_score']:>8.4f}  {r['steps_taken']:>6}"
            f"  {r['tickets_classified']:>5}/{r['tickets_total']:<3}"
            f"  {r['tickets_responded']:>5}/{r['tickets_total']}"
        )

    print(f"\nAverage score: {sum(scores)/len(scores):.4f}")
    print(f"{'='*65}\n")

    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Full results -> baseline_results.json")


if __name__ == "__main__":
    main()
