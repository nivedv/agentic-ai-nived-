"""
Lab 5 -- ARIA Tool Layer
Tool registration, chaining, and failure handling with raw Azure OpenAI
function calling (no framework).

Module 5 | Agentic AI Development for Innovation Teams
Engagement context: Meridian Software Ltd FY2024
"""

import json
import os
import sys
import time
import uuid
from datetime import date
from dotenv import load_dotenv
from openai import AzureOpenAI

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEPLOYMENT = "gpt-5.4"
load_dotenv()  # load .env from current working directory
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2025-01-01-preview",
)

TOKEN_BUDGET = 15_000      # hard stop per run -- Meridian cost-incident control
MAX_ITERATIONS = 8         # loop guard
MAX_RETRIES = 2            # transient tool failures only

# Phase 3 failure-injection switches. None = healthy tool.
FAILURE_CONFIG = {
    "fetch_engagement_data": "none",   # set to "timeout"  (transient)
    "analyze_variance":      "none",   # set to "corrupt"  (permanent)
    "draft_findings_memo":   "none",   # set to "reject"   (permanent)
}

# Tools exchange handles, never raw records. Raw data stays out of the
# model's context window -- this is a cost control, not a convenience.
DATA_STORE = {}

# Tracks one-shot transient failures so a retry can succeed.
_transient_fired = {"fetch_engagement_data": False}


class TransientToolError(Exception):
    """Worth retrying -- timeouts, throttling, flaky network."""


class PermanentToolError(Exception):
    """Retrying will not help -- bad input, validation failure, missing data."""


# ------------------------------------------------------------------
# Tool implementations (plain Python -- no LLM involvement)
# ------------------------------------------------------------------
def fetch_engagement_data(section: str) -> dict:
    """Fetch one section of the Meridian FY2024 ledger. Returns a handle."""
    if (FAILURE_CONFIG["fetch_engagement_data"] == "timeout"
            and not _transient_fired["fetch_engagement_data"]):
        _transient_fired["fetch_engagement_data"] = True
        raise TransientToolError("MeridianLedgerAPI timed out after 30s")

    with open("meridian_ledger.json", encoding="utf-8") as f:
        ledger = json.load(f)

    if section not in ledger:
        raise PermanentToolError(
            f"Unknown ledger section '{section}'. Valid sections: {sorted(ledger)}"
        )

    handle = f"DS-{uuid.uuid4().hex[:8]}"
    DATA_STORE[handle] = ledger[section]
    return {"handle": handle, "section": section, "record_count": len(ledger[section])}


def analyze_variance(handle: str, threshold_pct: float = 10.0) -> dict:
    """Flag accounts whose YoY variance meets or exceeds the threshold."""
    if FAILURE_CONFIG["analyze_variance"] == "corrupt":
        raise PermanentToolError(
            "Upstream payload failed schema validation (corrupt data). "
            "Do not proceed with analysis."
        )

    records = DATA_STORE.get(handle)
    if records is None:
        raise PermanentToolError(
            f"No data found for handle '{handle}'. Call fetch_engagement_data first."
        )

    flagged = [r for r in records if abs(r["variance_pct"]) >= threshold_pct]
    result_handle = f"DS-{uuid.uuid4().hex[:8]}"
    DATA_STORE[result_handle] = flagged
    return {
        "handle": result_handle,
        "threshold_pct": threshold_pct,
        "flagged_count": len(flagged),
        "flagged_accounts": [r["account"] for r in flagged],
    }


def draft_findings_memo(handle: str, analyst_summary: str) -> dict:
    """Write the findings memo. The memo STRUCTURE and DATA come from code;
    the model only supplies the narrative summary. Same principle as
    persist_precedent in Module 3: the write itself is deterministic."""
    if FAILURE_CONFIG["draft_findings_memo"] == "reject":
        raise PermanentToolError(
            "Memo rejected by compliance gate: engagement sign-off flag not set. "
            "Escalate to the engagement manager."
        )

    flagged = DATA_STORE.get(handle)
    if flagged is None:
        raise PermanentToolError(f"No analysis found for handle '{handle}'.")
    if not flagged:
        raise PermanentToolError(
            "Analysis contains zero flagged accounts -- nothing to memo. "
            "Re-run analyze_variance or lower the threshold."
        )
    if len(analyst_summary.strip()) < 20:
        raise PermanentToolError("analyst_summary too short to be a meaningful narrative.")

    # Deterministic write: every line below is code-controlled.
    lines = [
        "# Findings Memo -- Variance Review",
        "Engagement: Meridian Software Ltd FY2024",
        f"Date: {date.today().isoformat()}",
        "",
        "## Flagged accounts",
    ]
    for r in flagged:
        lines.append(
            f"- {r['account']}: FY2023 {r['fy2023']:,} -> FY2024 {r['fy2024']:,} "
            f"({r['variance_pct']:+.1f}%)"
        )
    lines += ["", "## Analyst narrative", analyst_summary.strip(), ""]

    out_path = "findings_memo_FY2024.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return {"status": "written", "path": out_path, "accounts_in_memo": len(flagged)}


# ------------------------------------------------------------------
# Tool schemas -- what the model sees
# ------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_engagement_data",
            "description": (
                "Fetch records for one section of the Meridian Software Ltd "
                "FY2024 engagement ledger. Returns a data handle and record "
                "count, never raw records."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": ["revenue", "expenses"],
                        "description": "Ledger section to fetch.",
                    }
                },
                "required": ["section"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_variance",
            "description": (
                "Flag ledger accounts whose year-on-year variance meets or "
                "exceeds a percentage threshold. Requires a handle from "
                "fetch_engagement_data. Returns a new handle to the flagged set."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "handle": {"type": "string", "description": "Handle from fetch_engagement_data."},
                    "threshold_pct": {"type": "number", "description": "Absolute variance threshold in percent. Default 10."},
                },
                "required": ["handle"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_findings_memo",
            "description": (
                "Write the findings memo file for flagged accounts. Requires a "
                "handle from analyze_variance and a short analyst narrative. "
                "The memo structure and figures are generated by code; only the "
                "narrative comes from you."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "handle": {"type": "string", "description": "Handle from analyze_variance."},
                    "analyst_summary": {"type": "string", "description": "2-4 sentence narrative on the flagged variances."},
                },
                "required": ["handle", "analyst_summary"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "fetch_engagement_data": fetch_engagement_data,
    "analyze_variance": analyze_variance,
    "draft_findings_memo": draft_findings_memo,
}


# ------------------------------------------------------------------
# Tool executor -- retry, classification, structured errors
# ------------------------------------------------------------------
def execute_tool(name: str, args: dict) -> dict:
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return {"status": "error", "retryable": False,
                "message": f"Tool '{name}' is not registered."}

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            return {"status": "ok", "result": func(**args)}
        except TransientToolError as exc:
            if attempt <= MAX_RETRIES:
                wait = 2 ** (attempt - 1)
                print(f"  [retry] {name} attempt {attempt} failed ({exc}); retrying in {wait}s")
                time.sleep(wait)
                continue
            return {"status": "error", "retryable": True,
                    "message": f"{name} failed after {MAX_RETRIES} retries: {exc}"}
        except PermanentToolError as exc:
            return {"status": "error", "retryable": False, "message": str(exc)}
        except TypeError as exc:
            return {"status": "error", "retryable": False,
                    "message": f"Bad arguments for {name}: {exc}"}


# ------------------------------------------------------------------
# Agent loop -- raw function calling, no framework
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are ARIA (Audit Research Intelligence Assistant) supporting the
Meridian Software Ltd FY2024 engagement for KPMG Deals Advisory.

Operating rules:
1. Always fetch data before analysing. Pass handles between tools; never ask for
   or repeat raw ledger records.
2. If a tool returns status "error" with retryable=false, do NOT attempt the same
   call again. State clearly what could not be completed and recommend escalation
   to the engagement manager.
3. Never invent ledger figures, account names, or memo content. If data is
   unavailable, say so.
4. When the task is complete, summarise what was done in plain language."""


def run_agent(user_request: str) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request},
    ]
    total_tokens = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            tools=TOOLS,
        )
        total_tokens += response.usage.total_tokens
        print(f"[loop {iteration}] cumulative tokens: {total_tokens}")

        if total_tokens > TOKEN_BUDGET:
            print(f"[guard] TOKEN_BUDGET of {TOKEN_BUDGET} exceeded -- aborting run.")
            return

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                print(f"  [tool] {tc.function.name}({args})")
                outcome = execute_tool(tc.function.name, args)
                print(f"  [tool] -> {outcome['status']}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(outcome),
                })
            continue

        print("\n=== ARIA final response ===")
        print(msg.content)
        return

    print(f"[guard] MAX_ITERATIONS of {MAX_ITERATIONS} reached -- aborting run.")


# ------------------------------------------------------------------
# Phase runner
# ------------------------------------------------------------------
REQUESTS = {
    "1": "How many records are in the expenses section of the Meridian ledger?",
    "2": ("Review the FY2024 revenue ledger for Meridian Software Ltd, flag "
          "variances of 15% or more, and draft a findings memo summarising "
          "the flagged accounts."),
}

if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "1"
    run_agent(REQUESTS.get(phase, REQUESTS["1"]))
