"""
Task definitions for the Ticket Triage environment.
Three tasks: easy → medium → hard.
Each task specifies a set of tickets and the expected correct outcomes
used by the deterministic graders.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Ticket:
    id: str
    subject: str
    body: str
    customer_name: str
    customer_tier: str          # "free" | "pro" | "enterprise"
    created_at: str
    previous_tickets: int = 0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "body": self.body,
            "customer_name": self.customer_name,
            "customer_tier": self.customer_tier,
            "created_at": self.created_at,
            "previous_tickets": self.previous_tickets,
            "tags": list(self.tags),
        }


@dataclass
class TaskDef:
    task_id: str
    name: str
    description: str
    difficulty: str             # "easy" | "medium" | "hard"
    tickets: List[Ticket]
    expected_outcomes: Dict[str, Dict[str, Any]]
    max_steps: int = 10


# ─────────────────────────────────────────────────────────────────
# TASK 1 — EASY  (5 tickets, classify only)
# ─────────────────────────────────────────────────────────────────
TASK_EASY = TaskDef(
    task_id="task_easy",
    name="Basic Ticket Classification",
    difficulty="easy",
    max_steps=10,
    description=(
        "You are a support agent. For EACH ticket in the inbox use the 'classify' "
        "action to set its category (billing | technical | account | shipping | "
        "product | general) AND priority (critical | high | medium | low). "
        "Process all 5 tickets."
    ),
    tickets=[
        Ticket("t001", "Cannot log in to my account",
               "Hi, I've been trying to log in for the past hour but keep getting "
               "'Invalid credentials'. I haven't changed my password. Please help!",
               "Alice Johnson", "pro", "2024-01-15T09:00:00Z", previous_tickets=1),
        Ticket("t002", "Charged twice for subscription",
               "I was billed $99 twice this month. My bank shows two charges on Jan 1st. "
               "I need a refund immediately. This is unacceptable!",
               "Bob Smith", "pro", "2024-01-15T09:30:00Z"),
        Ticket("t003", "Where is my order?",
               "I ordered a laptop stand 3 weeks ago (Order #45821) and haven't received it. "
               "Tracking says 'in transit' since Jan 2nd.",
               "Carol White", "free", "2024-01-15T10:00:00Z"),
        Ticket("t004", "Question about export formats",
               "Hi, does your product support CSV export? I'd like to export my data. Thanks!",
               "Dave Lee", "free", "2024-01-15T10:15:00Z", previous_tickets=2),
        Ticket("t005", "URGENT: Production system down - all users affected",
               "Our entire production environment is down. 500 users cannot access the platform. "
               "We are losing $10k/hour. Need immediate escalation to engineering!",
               "Eve Martinez", "enterprise", "2024-01-15T10:30:00Z", previous_tickets=5),
    ],
    expected_outcomes={
        "t001": {"category": "account",    "priority": "high"},
        "t002": {"category": "billing",    "priority": "high"},
        "t003": {"category": "shipping",   "priority": "medium"},
        "t004": {"category": "product",    "priority": "low"},
        "t005": {"category": "technical",  "priority": "critical"},
    },
)


# ─────────────────────────────────────────────────────────────────
# TASK 2 — MEDIUM  (5 tickets, classify + respond + escalate)
# ─────────────────────────────────────────────────────────────────
TASK_MEDIUM = TaskDef(
    task_id="task_medium",
    name="Full Ticket Triage with Responses",
    difficulty="medium",
    max_steps=20,
    description=(
        "You are a senior support agent. For each ticket: "
        "1) Use 'classify' to set category and priority. "
        "2) Use 'respond' to write a helpful, professional response that uses the "
        "customer's name and gives concrete next steps. "
        "3) For CRITICAL or security-related tickets also use 'escalate' before responding. "
        "Process all 5 tickets."
    ),
    tickets=[
        Ticket("m001", "API rate limits too restrictive",
               "We're building an integration and keep hitting 429 errors. We need at least "
               "1000 req/min but the docs say limit is 100 req/min. Is there an enterprise plan?",
               "Frank Chen", "pro", "2024-01-15T11:00:00Z", previous_tickets=3,
               tags=["api", "integration"]),
        Ticket("m002", "Data breach concern - my account was accessed",
               "I received a login notification from an IP in Russia that wasn't me. "
               "Someone may have accessed my files. I have sensitive business documents stored. "
               "This needs immediate attention!!",
               "Grace Kim", "enterprise", "2024-01-15T11:15:00Z", previous_tickets=1),
        Ticket("m003", "Refund request - product didn't meet expectations",
               "I purchased the annual plan 2 weeks ago but the features I needed aren't "
               "available. I'd like to cancel and get a prorated refund. I'm within the 30-day window.",
               "Henry Wang", "pro", "2024-01-15T11:30:00Z"),
        Ticket("m004", "How do I invite team members?",
               "Hi, I'm trying to add my colleagues to our workspace. I went to Settings > Team "
               "but the invite button is greyed out. Am I missing something?",
               "Iris Patel", "pro", "2024-01-15T12:00:00Z", previous_tickets=4),
        Ticket("m005", "Wrong item shipped",
               "I ordered the blue version (SKU: BLU-2024) but received the red one. "
               "I have an important presentation tomorrow and specifically need the blue one. "
               "Please help ASAP.",
               "Jack Brown", "free", "2024-01-15T12:30:00Z"),
    ],
    expected_outcomes={
        "m001": {"category": "technical", "priority": "high",
                 "response_required": True,
                 "response_keywords": ["rate limit", "enterprise", "upgrade", "contact"],
                 "escalation": None},
        "m002": {"category": "account",   "priority": "critical",
                 "response_required": True,
                 "response_keywords": ["security", "password", "secure", "investigate"],
                 "escalation": "tier2"},
        "m003": {"category": "billing",   "priority": "medium",
                 "response_required": True,
                 "response_keywords": ["refund", "cancel", "30", "process"],
                 "escalation": None},
        "m004": {"category": "technical", "priority": "low",
                 "response_required": True,
                 "response_keywords": ["invite", "plan", "seats", "upgrade"],
                 "escalation": None},
        "m005": {"category": "shipping",  "priority": "high",
                 "response_required": True,
                 "response_keywords": ["apologize", "correct", "ship", "expedite"],
                 "escalation": None},
    },
)


# ─────────────────────────────────────────────────────────────────
# TASK 3 — HARD  (8 tickets, full workflow)
# Agent must: classify all, respond to high/critical, escalate appropriately,
# detect duplicate tickets, close resolved tickets, flag SLA breaches.
# ─────────────────────────────────────────────────────────────────
TASK_HARD = TaskDef(
    task_id="task_hard",
    name="Advanced Inbox Management with SLA Compliance",
    difficulty="hard",
    max_steps=30,
    description=(
        "You are a support team lead. Process 8 tickets efficiently. "
        "Requirements: "
        "(1) Classify ALL tickets (category + priority). "
        "(2) Use 'respond' for all HIGH and CRITICAL tickets. "
        "(3) Use 'escalate' for CRITICAL tickets and enterprise security/fraud cases. "
        "(4) Use 'tag' with tags=['related:TICKET_ID'] to link duplicate/related tickets. "
        "(5) Use 'close' for tickets the customer has already self-resolved. "
        "(6) Use 'tag' with tags=['sla-breach'] on tickets from enterprise customers "
        "that are over 4 hours old (created before 08:00 today). "
        "Work efficiently — you have limited steps."
    ),
    tickets=[
        Ticket("h001", "Payment failing - card declined",
               "My card is being declined when trying to upgrade. I've tried 3 different cards. "
               "Error: 'Payment processor unavailable'. This is blocking our team from working.",
               "Liam O'Brien", "enterprise", "2024-01-15T06:00:00Z", previous_tickets=8),
        Ticket("h002", "NVM - figured it out",
               "Never mind, I found the settings. Sorry for the bother!",
               "Mia Thompson", "free", "2024-01-15T09:45:00Z", previous_tickets=1),
        Ticket("h003", "Dashboard not loading - blank screen",
               "Getting a blank white screen on dashboard. Tried Chrome, Firefox, Safari. "
               "Cleared cache. Started around 10am today. Our whole team is affected.",
               "Noah Davis", "enterprise", "2024-01-15T10:00:00Z", previous_tickets=3),
        Ticket("h004", "Dashboard still down - urgent!!!",
               "Following up on my earlier ticket about the blank dashboard. It's been 2 hours "
               "and no response. We have a board meeting at 2pm and NEED this fixed.",
               "Noah Davis", "enterprise", "2024-01-15T12:00:00Z", previous_tickets=3),
        Ticket("h005", "How to export to PDF?",
               "Hi there, is there a way to export reports as PDF? "
               "I've been looking through the docs but can't find it. Thanks",
               "Olivia Garcia", "free", "2024-01-15T11:00:00Z"),
        Ticket("h006", "Suspicious invoice - possible fraud",
               "We received an invoice for $4,500 that we did not authorize. "
               "Our account shows charges for 'Enterprise Add-ons' that nobody on our team purchased. "
               "Possible unauthorized access.",
               "Peter Wilson", "enterprise", "2024-01-15T10:30:00Z", previous_tickets=2),
        Ticket("h007", "Feature request: Dark mode",
               "Love the product! One thing that would make it even better is a dark mode option. "
               "Any plans for this? Keep up the great work!",
               "Quinn Anderson", "free", "2024-01-15T09:00:00Z"),
        Ticket("h008", "CRITICAL: Data loss - all our files gone",
               "ALL our project files have disappeared from our account. We had 3 years of work "
               "stored here. This happened after the maintenance window last night. "
               "We need immediate data recovery. Legal is involved if this isn't resolved TODAY.",
               "Rachel Scott", "enterprise", "2024-01-15T08:00:00Z", previous_tickets=12),
    ],
    expected_outcomes={
        "h001": {"category": "billing",   "priority": "critical",
                 "response_required": True, "escalation": "billing_team",
                 "sla_breach": True},
        "h002": {"action": "close",       "priority": "low"},
        "h003": {"category": "technical", "priority": "critical",
                 "response_required": True, "escalation": "engineering",
                 "related_to": "h004"},
        "h004": {"category": "technical", "priority": "critical",
                 "is_duplicate": True,     "related_to": "h003"},
        "h005": {"category": "product",   "priority": "low",
                 "response_required": False},
        "h006": {"category": "billing",   "priority": "critical",
                 "response_required": True, "escalation": "billing_team",
                 "security_flag": True},
        "h007": {"category": "product",   "priority": "low"},
        "h008": {"category": "technical", "priority": "critical",
                 "response_required": True, "escalation": "engineering",
                 "sla_breach": True},
    },
)


ALL_TASKS: Dict[str, TaskDef] = {
    "task_easy":   TASK_EASY,
    "task_medium": TASK_MEDIUM,
    "task_hard":   TASK_HARD,
}
