import json
import os
import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
MODEL = "claude-sonnet-4-6"

CONVERSATIONS_DIR = Path(__file__).parent / "conversations"
QUERIES_DIR = Path(__file__).parent / "queries"

# ---------------------------------------------------------------------------
# Conversation specs — domain, context, required landmarks
# ---------------------------------------------------------------------------

CONVERSATION_SPECS = [
    {
        "id": "conv_01_project_planning",
        "domain": "Project Planning",
        "context": "A team of 4 planning a mobile app launch. Product manager, designer, backend dev, and QA lead.",
        "required_landmarks": [
            "Decision: launch date set for March 15th",
            "Commitment: backend dev will finish API by Feb 28th",
            "Action item: designer to deliver final mockups by next Monday",
            "Deadline: QA testing window is March 1-10",
        ],
        "filler_topics": ["weekend plans", "coffee preferences", "office temperature complaints"],
    },
    {
        "id": "conv_02_technical_arch",
        "domain": "Technical Architecture",
        "context": "Senior engineers debating microservices vs monolith for a new payment processing system.",
        "required_landmarks": [
            "Decision: agreed to go with microservices architecture",
            "Decision: chose PostgreSQL over DynamoDB for transaction storage",
            "Commitment: lead architect will write the RFC by Friday",
            "Action item: team needs to benchmark Redis vs Memcached",
        ],
        "filler_topics": ["past project war stories", "conference talks they watched", "new IDE plugins"],
    },
    {
        "id": "conv_03_sales_call",
        "domain": "Sales Call",
        "context": "Account executive and solutions engineer on a call with a prospect evaluating enterprise software.",
        "required_landmarks": [
            "Decision: prospect wants to proceed with a pilot program",
            "Commitment: AE will send pricing proposal by end of week",
            "Deadline: procurement needs approval by end of Q2",
            "Action item: solutions engineer to set up sandbox environment",
        ],
        "filler_topics": ["industry trends", "competitor mentions", "small talk about travel"],
    },
    {
        "id": "conv_04_customer_support",
        "domain": "Customer Support",
        "context": "Support agent helping a frustrated customer with a billing discrepancy and account access issues.",
        "required_landmarks": [
            "Decision: agreed to issue a full refund for the double charge",
            "Commitment: agent will escalate the account lock to engineering",
            "Action item: customer needs to reset password after fix",
            "Deadline: resolution expected within 24 hours",
        ],
        "filler_topics": ["apologetic pleasantries", "hold time complaints", "service plan details"],
    },
    {
        "id": "conv_05_team_retro",
        "domain": "Team Retrospective",
        "context": "Agile team running a sprint retrospective after a tough 2-week sprint with a production incident.",
        "required_landmarks": [
            "Decision: we decided to add mandatory code review for all hotfixes",
            "Commitment: SRE lead will implement automated rollback by next sprint",
            "Action item: team lead to update the on-call runbook",
            "Action item: each team member to document one process improvement",
        ],
        "filler_topics": ["praise for team members", "jokes about the incident", "snack preferences for next retro"],
    },
    {
        "id": "conv_06_product_roadmap",
        "domain": "Product Roadmap",
        "context": "Product team quarterly planning — PM, engineering lead, data analyst, and head of design.",
        "required_landmarks": [
            "Decision: AI search feature is top priority for Q3",
            "Decision: sunset the legacy dashboard by end of Q4",
            "Commitment: data analyst will deliver usage metrics report by next Tuesday",
            "Deadline: board presentation on roadmap is June 15th",
        ],
        "filler_topics": ["user feedback anecdotes", "competitor feature comparisons", "team lunch plans"],
    },
    {
        "id": "conv_07_debugging_session",
        "domain": "Debugging Session",
        "context": "Two developers pair-debugging a memory leak in a Python web service that's causing OOM crashes in production.",
        "required_landmarks": [
            "Decision: confirmed the leak is in the connection pool — not closing connections properly",
            "Commitment: senior dev will deploy the fix to staging tonight",
            "Action item: add memory profiling to CI pipeline",
            "Deadline: must be fixed before Friday's traffic spike",
        ],
        "filler_topics": ["profiler tool recommendations", "similar bugs they've seen", "deployment process gripes"],
    },
    {
        "id": "conv_08_hiring_discussion",
        "domain": "Hiring Discussion",
        "context": "Hiring panel debrief after interviewing 3 candidates for a senior ML engineer position.",
        "required_landmarks": [
            "Decision: agreed to extend offer to Candidate B (Sarah)",
            "Commitment: hiring manager will draft the offer letter by tomorrow",
            "Action item: recruiter to check salary band with HR",
            "Deadline: offer must go out by Thursday to beat competing offer",
        ],
        "filler_topics": ["general impressions", "interview format feedback", "team culture fit discussion"],
    },
    {
        "id": "conv_09_budget_planning",
        "domain": "Budget Planning",
        "context": "Engineering and finance leads planning Q3 infrastructure budget — cloud costs, tooling, and headcount.",
        "required_landmarks": [
            "Decision: approved $45K monthly AWS budget increase for GPU instances",
            "Decision: settled on Datadog over New Relic for observability",
            "Commitment: finance lead will prepare the board budget slide by Monday",
            "Action item: engineering to provide 6-month cloud cost projection",
        ],
        "filler_topics": ["past budget overruns", "vendor negotiation stories", "office space costs"],
    },
    {
        "id": "conv_10_onboarding",
        "domain": "Onboarding",
        "context": "Engineering manager onboarding a new hire — walking through codebase, team processes, and first-week tasks.",
        "required_landmarks": [
            "Decision: new hire will start on the search team, not recommendations",
            "Commitment: manager will pair-program with new hire on first PR",
            "Action item: new hire needs to complete security training by end of first week",
            "Deadline: first meaningful PR expected by end of week 2",
        ],
        "filler_topics": ["team introductions", "lunch recommendations near office", "Slack channel tour"],
    },
]


# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------

def _build_conversation_prompt(spec: dict) -> str:
    """Build the prompt for generating a single conversation."""
    landmarks_str = "\n".join(f"  - {lm}" for lm in spec["required_landmarks"])
    filler_str = ", ".join(spec["filler_topics"])

    return f"""Generate a realistic multi-turn conversation in JSON format.

DOMAIN: {spec["domain"]}
CONTEXT: {spec["context"]}

REQUIREMENTS:
- Exactly 60 messages (alternating user/assistant roles)
- Messages should feel natural — varying lengths from 1 sentence to 3-4 sentences
- Include these LANDMARK messages (decisions, commitments, action items) — embed them naturally in the conversation, don't cluster them together:
{landmarks_str}
- Include filler/low-value messages about: {filler_str}
- Mix of short casual messages ("ok", "sounds good", "yeah agreed") and substantive ones
- The conversation should flow naturally with topic shifts

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences, no preamble:
{{
  "id": "{spec["id"]}",
  "domain": "{spec["domain"]}",
  "message_count": 60,
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    ...
  ]
}}

CRITICAL: Output ONLY the JSON object. No explanation, no markdown code fences."""


def _build_queries_prompt(spec: dict, conversation: dict) -> str:
    """Build prompt for generating evaluation queries for a conversation."""
    # Take first 5 and last 5 messages as context hint
    msgs = conversation["messages"]
    sample = msgs[:5] + msgs[-5:]
    sample_str = json.dumps(sample, indent=2)

    landmarks_str = "\n".join(f"  - {lm}" for lm in spec["required_landmarks"])

    return f"""Given this conversation about {spec["domain"]}, generate 4 evaluation queries.

The conversation contains these landmark messages:
{landmarks_str}

Sample messages from the conversation:
{sample_str}

Generate 4 queries — one of each type:
1. FACTUAL: asks about a specific decision or fact from the conversation
2. ANALYTICAL: asks for a summary or comparison of discussion topics
3. PROCEDURAL: asks about steps or process discussed
4. LANDMARK: specifically targets one of the landmark decisions/commitments

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences:
{{
  "conversation_id": "{spec["id"]}",
  "queries": [
    {{
      "query": "...",
      "type": "factual",
      "expected_landmark_indices": [],
      "description": "Brief description of what a good answer should include"
    }},
    {{
      "query": "...",
      "type": "analytical",
      "expected_landmark_indices": [],
      "description": "..."
    }},
    {{
      "query": "...",
      "type": "procedural",
      "expected_landmark_indices": [],
      "description": "..."
    }},
    {{
      "query": "...",
      "type": "landmark",
      "expected_landmark_indices": [],
      "description": "..."
    }}
  ]
}}

CRITICAL: Output ONLY the JSON object. No explanation, no markdown code fences."""


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def generate_conversation(spec: dict) -> dict:
    """Generate a single conversation using Claude API."""
    prompt = _build_conversation_prompt(spec)

    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]  # remove first line
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        conversation = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error for {spec['id']}: {e}")
        print(f"  Raw output (first 200 chars): {raw[:200]}")
        raise

    # Validate basic structure
    assert "messages" in conversation, f"Missing 'messages' key in {spec['id']}"
    assert len(conversation["messages"]) >= 20, (
        f"Too few messages in {spec['id']}: {len(conversation['messages'])}"
    )

    return conversation


def generate_queries(spec: dict, conversation: dict) -> dict:
    """Generate evaluation queries for a conversation."""
    prompt = _build_queries_prompt(spec, conversation)

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        queries = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error for queries {spec['id']}: {e}")
        print(f"  Raw output (first 200 chars): {raw[:200]}")
        raise

    return queries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Generate all conversations and queries."""
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)

    all_queries = []

    print(f"Generating {len(CONVERSATION_SPECS)} conversations...\n")

    for i, spec in enumerate(CONVERSATION_SPECS):
        print(f"[{i+1}/{len(CONVERSATION_SPECS)}] {spec['domain']}...")

        # Generate conversation
        try:
            conversation = generate_conversation(spec)
            msg_count = len(conversation["messages"])
            print(f"  ✓ Generated {msg_count} messages")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

        # Save conversation
        conv_path = CONVERSATIONS_DIR / f"{spec['id']}.json"
        with open(conv_path, "w") as f:
            json.dump(conversation, f, indent=2)
        print(f"  ✓ Saved to {conv_path}")

        # Generate queries
        try:
            queries = generate_queries(spec, conversation)
            query_count = len(queries.get("queries", []))
            print(f"  ✓ Generated {query_count} queries")
            all_queries.append(queries)
        except Exception as e:
            print(f"  ✗ Query generation failed: {e}")
            continue

        print()

    # Save all queries
    queries_path = QUERIES_DIR / "queries.json"
    with open(queries_path, "w") as f:
        json.dump(all_queries, f, indent=2)
    print(f"\nAll queries saved to {queries_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"  Conversations: {len(list(CONVERSATIONS_DIR.glob('*.json')))}")
    print(f"  Query sets: {len(all_queries)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
