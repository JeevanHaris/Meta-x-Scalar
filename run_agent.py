"""
LLM Agent Runner for Ticket Triage Environment
================================================
Drives Claude through all three tasks and writes baseline_results.json
in the exact format the dashboard expects.

Usage:
    pip install anthropic httpx
    python run_agent.py --url https://YOUR-SPACE.hf.space --api-key sk-ant-...

    # or via env vars:
    export ANTHROPIC_API_KEY=sk-ant-...
    export ENV_BASE_URL=https://YOUR-SPACE.hf.space
    python run_agent.py
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_URL   = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL         = "claude-opus-4-5"
TASKS         = ["task_easy", "task_medium", "task_hard"]

SYSTEM_PROMPT = """You are an expert customer support agent working inside a ticket triage system.

You will receive a JSON observation. Return ONLY a valid JSON action object — no explanation, no markdown.

Action schema:
{
  "action_type": "classify" | "prioritize" | "respond" | "escalate" | "close" | "tag",
  "ticket_id": "<ticket id>",
  "category":      "billing" | "technical" | "account" | "shipping" | "product" | "general",
  "priority":      "critical" | "high" | "medium" | "low",
  "response_text": "<customer-facing text, 80+ chars, use customer first name>",
  "escalate_to":   "tier2" | "billing_team" | "engineering",
  "tags":          ["sla-breach", "related:OTHER_ID"],
  "reasoning":     "<optional>"
}

Rules:
- classify: always set BOTH category AND priority in one action
- respond: 80+ chars, use customer first name, give concrete next steps
- escalate BEFORE responding for critical/security tickets
- tag "sla-breach": enterprise tickets created before 08:00
- tag "related:OTHER_ID": duplicate/related tickets (link both ways)
- close: only when customer says they already solved it
- Process ALL tickets before max_steps is reached
- Return ONLY JSON, nothing else
"""

# ── HTTP helpers ───────────────────────────────────────────────────────────────
async def api_post(client: httpx.AsyncClient, base: str, path: str, body: dict) -> dict:
    r = await client.post(f"{base}{path}", json=body, timeout=30)
    r.raise_for_status()
    return r.json()

async def api_get(client: httpx.AsyncClient, base: str, path: str) -> dict:
    r = await client.get(f"{base}{path}", timeout=30)
    r.raise_for_status()
    return r.json()

# ── LLM call ──────────────────────────────────────────────────────────────────
def ask_llm(
    llm: anthropic.Anthropic,
    observation: dict,
    history: List[dict],
) -> Tuple[Optional[dict], str]:
    messages = []

    if not history:
        messages.append({
            "role": "user",
            "content": f"Task started. First observation:\n{json.dumps(observation, indent=2)}"
        })
    else:
        for h in history[-8:]:
            messages.append(h)
        messages.append({
            "role": "user",
            "content": f"New observation:\n{json.dumps(observation, indent=2)}\nNext action?"
        })

    response = llm.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    raw = response.content[0].text.strip()

    # Strip markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        print(f"  ⚠  Bad JSON from LLM: {raw[:120]}")
        return None, raw

# ── Single task runner ────────────────────────────────────────────────────────
async def run_task(
    http: httpx.AsyncClient,
    llm: anthropic.Anthropic,
    base_url: str,
    task_id: str,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Task: {task_id}")
    print(f"{'='*60}")

    obs = await api_post(http, base_url, "/reset", {"task_id": task_id})
    print(f"  Instructions: {str(obs.get('instructions',''))[:120]}...")

    history: List[dict] = []
    action_log: List[dict] = []
    step = 0
    max_steps = obs.get("max_steps", 30)
    done = obs.get("done", False)

    while not done and step < max_steps:
        step += 1
        print(f"\n  Step {step}/{max_steps}")

        action_obj, raw_text = ask_llm(llm, obs, history)

        # Fallback if LLM returns bad JSON
        if action_obj is None:
            ticket_id = (obs.get("current_ticket") or {}).get("id", "t001")
            action_obj = {
                "action_type": "classify",
                "ticket_id": ticket_id,
                "category": "general",
                "priority": "medium",
            }

        print(f"  → {action_obj.get('action_type')} ticket={action_obj.get('ticket_id')} "
              f"cat={action_obj.get('category','-')} pri={action_obj.get('priority','-')}")

        history.append({"role": "assistant", "content": raw_text})

        try:
            result = await api_post(http, base_url, "/step", action_obj)
        except httpx.HTTPStatusError as e:
            print(f"  ✗ HTTP {e.response.status_code}: {e.response.text[:200]}")
            break

        reward = result.get("reward", 0.0)
        done   = result.get("done", False)
        obs    = result.get("observation", result)

        # Record in format dashboard expects
        action_log.append({
            "step":   step,
            "action": action_obj,
            "reward": round(reward, 4),
        })

        print(f"  reward={reward:.4f}  done={done}")

        history.append({
            "role": "user",
            "content": f"Action result: reward={reward}, done={done}"
        })

        if done:
            print(f"  ✓ Episode done after {step} steps")
            break

    # Fetch final state
    try:
        state = await api_get(http, base_url, "/state")
    except Exception as e:
        print(f"  ⚠ Could not fetch state: {e}")
        state = {}

    final_score    = state.get("final_score", 0.0)
    grader_details = state.get("grader_details", {})
    classified     = state.get("tickets_classified", 0)
    responded      = state.get("tickets_responded", 0)
    total_tickets  = state.get("tickets_total", 0)

    print(f"\n  Score: {final_score:.4f}  |  {classified}/{total_tickets} classified  |  {responded}/{total_tickets} responded")

    # Return in the EXACT structure dashboard expects at top level
    return {
        "final_score":        round(final_score, 4),
        "tickets_classified": classified,
        "tickets_responded":  responded,
        "tickets_total":      total_tickets,
        "steps_taken":        step,
        "action_log":         action_log,
        "grader_details":     grader_details,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
async def main(base_url: str, api_key: str, tasks: List[str], output: str):
    llm = anthropic.Anthropic(api_key=api_key)

    async with httpx.AsyncClient() as http:
        # Health check
        try:
            h = await api_get(http, base_url, "/health")
            print(f"✓ Server healthy: {h}")
        except Exception as e:
            print(f"✗ Cannot reach server at {base_url}: {e}")
            sys.exit(1)

        # Dashboard expects: { "task_easy": {...}, "task_medium": {...}, "task_hard": {...} }
        results: Dict[str, Any] = {}

        for task_id in tasks:
            try:
                results[task_id] = await run_task(http, llm, base_url, task_id)
            except Exception as e:
                print(f"  ✗ Task {task_id} failed: {e}")
                results[task_id] = {
                    "final_score": 0.0,
                    "tickets_classified": 0,
                    "tickets_responded": 0,
                    "tickets_total": 0,
                    "steps_taken": 0,
                    "action_log": [],
                    "grader_details": {"error": str(e)},
                }

    # Summary
    print(f"\n{'='*60}  SUMMARY")
    scores = [r["final_score"] for r in results.values()]
    avg = sum(scores) / max(len(scores), 1)
    for tid, r in results.items():
        print(f"  {tid:15s}  score={r['final_score']:.4f}  steps={r['steps_taken']}")
    print(f"\n  Average: {avg:.4f}")

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Written to {output}")
    print("  Commit this file to your repo root and push to HF — dashboard updates instantly.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",     default=DEFAULT_URL)
    parser.add_argument("--api-key", default=ANTHROPIC_KEY)
    parser.add_argument("--tasks",   default=",".join(TASKS))
    parser.add_argument("--output",  default="baseline_results.json")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: Set ANTHROPIC_API_KEY or pass --api-key")
        sys.exit(1)

    asyncio.run(main(
        base_url=args.url,
        api_key=args.api_key,
        tasks=args.tasks.split(","),
        output=args.output,
    ))
