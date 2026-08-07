"""Tool registry and the HR record tools the copilot can call.

Contract for every tool
-----------------------
A tool returns a ``ToolResult`` with three separate payloads:

* ``summary`` — compact text injected into the LLM prompt. It is written for a
  model, not a human: dense, unambiguous, no markdown tables.
* ``data``    — structured JSON the frontend renders as a card, so the user
  sees the *record* and not just the model's retelling of it.
* ``policy_hint`` — a query string used to pull the governing policy passage
  alongside the record. This is what turns "you have 8.5 days left" into
  "you have 8.5 days left, and unused days above 5 expire on 31 March [2]".

Access control is enforced at call time from the server-side identity, never
from an argument the model produced. A model that hallucinates
``employee_id=EMP-1099`` cannot read that employee's balance.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from .hr_records import AccessDenied, HRRecords

LEAVE_TYPE_ALIASES = {
    "annual": "ANNUAL", "pto": "ANNUAL", "vacation": "ANNUAL", "holiday": "ANNUAL",
    "leave": "ANNUAL", "sick": "SICK", "illness": "SICK", "wellbeing": "WELLBEING",
    "wellness": "WELLBEING", "volunteer": "VOLUNTEER", "volunteering": "VOLUNTEER",
}


@dataclass
class ToolResult:
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    policy_hint: str = ""
    error: str | None = None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, str]
    handler: Callable[..., ToolResult]
    reads_personal_data: bool = True

    def describe(self) -> str:
        args = ", ".join(f"{key}: {value}" for key, value in self.parameters.items()) or "no arguments"
        return f"- {self.name}({args}) — {self.description}"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def tool_get_leave_balance(
    records: HRRecords,
    requester_id: str,
    *,
    leave_type: str | None = None,
    **_ignored: Any,
) -> ToolResult:
    employee = records.employee(requester_id)
    if not employee:
        return ToolResult(False, "", error=f"Unknown employee id {requester_id}")
    normalised = LEAVE_TYPE_ALIASES.get((leave_type or "").strip().lower(), leave_type)
    rows = records.balances(requester_id, normalised)
    if not rows:
        return ToolResult(
            False,
            "",
            error=f"No {leave_type or 'leave'} balance on record for {requester_id}",
        )
    lines = [
        f"{row['leave_type_label']}: {row['available_days']} days available "
        f"(entitled {row['entitled_days']}, carried over {row['carried_over_days']}, "
        f"taken {row['taken_days']}, pending approval {row['pending_days']}) "
        f"as of {row['as_of_date']}"
        for row in rows
    ]
    return ToolResult(
        ok=True,
        summary=(
            f"Leave balances for {employee['full_name']} ({employee['employment_type']}, "
            f"{employee['location']}):\n" + "\n".join(f"- {line}" for line in lines)
        ),
        data={
            "kind": "leave_balance",
            "employee": {
                "employee_id": employee["employee_id"],
                "full_name": employee["full_name"],
                "employment_type": employee["employment_type"],
            },
            "balances": rows,
        },
        policy_hint="annual leave entitlement accrual carryover expiry sick leave",
    )


def tool_get_upcoming_holidays(
    records: HRRecords,
    requester_id: str,
    *,
    limit: int = 5,
    include_optional: bool = True,
    today: date | None = None,
    **_ignored: Any,
) -> ToolResult:
    employee = records.employee(requester_id)
    if not employee:
        return ToolResult(False, "", error=f"Unknown employee id {requester_id}")
    reference = today or date.today()
    try:
        limit = max(1, min(int(limit), 15))
    except (TypeError, ValueError):
        limit = 5
    rows = records.upcoming_holidays(
        employee["country"],
        on_or_after=reference,
        limit=limit,
        include_optional=include_optional,
    )
    if not rows:
        return ToolResult(
            False, "", error=f"No remaining {reference.year} holidays for {employee['country']}"
        )
    lines = [f"{row['holiday_date']} — {row['holiday_name']} ({row['type']})" for row in rows]
    return ToolResult(
        ok=True,
        summary=(
            f"Next {len(rows)} company holidays for {employee['location']} "
            f"(calendar {employee['country']}, from {reference.isoformat()}):\n"
            + "\n".join(f"- {line}" for line in lines)
        ),
        data={
            "kind": "holidays",
            "country": employee["country"],
            "location": employee["location"],
            "from_date": reference.isoformat(),
            "holidays": rows,
        },
        policy_hint="public holidays swap religious significance leave year",
    )


def tool_get_my_leave_requests(
    records: HRRecords,
    requester_id: str,
    *,
    status: str | None = None,
    **_ignored: Any,
) -> ToolResult:
    rows = records.requests(requester_id, status)
    if not rows:
        return ToolResult(
            False,
            "",
            error=f"No {status or ''} leave requests on record for {requester_id}".replace("  ", " "),
        )
    lines = [
        f"{row['request_id']}: {row['leave_type']} {row['start_date']} to {row['end_date']} "
        f"({row['working_days']} working days) — {row['status']}"
        for row in rows
    ]
    return ToolResult(
        ok=True,
        summary="Leave requests on record:\n" + "\n".join(f"- {line}" for line in lines),
        data={"kind": "leave_requests", "requests": rows},
        policy_hint="leave request approval notice period manager response",
    )


def tool_lookup_employee_directory(
    records: HRRecords,
    requester_id: str,
    *,
    query: str = "",
    **_ignored: Any,
) -> ToolResult:
    rows = records.search_directory(query, limit=5)
    if not rows:
        return ToolResult(False, "", error=f"No directory match for '{query}'")
    enriched: list[dict[str, str]] = []
    for row in rows:
        manager = records.directory_entry(row["manager_id"]) if row["manager_id"] else None
        enriched.append({**row, "manager_name": manager["full_name"] if manager else "—"})
    lines = [
        f"{row['full_name']} — {row['job_title']}, {row['department']}, {row['location']} "
        f"(manager: {row['manager_name']}, {row['work_email']})"
        for row in enriched
    ]
    return ToolResult(
        ok=True,
        summary=(
            "Company directory matches (public fields only — no pay, leave or "
            "performance data is available for other employees):\n"
            + "\n".join(f"- {line}" for line in lines)
        ),
        data={"kind": "directory", "query": query, "people": enriched},
        policy_hint="",
    )


def tool_get_my_profile(
    records: HRRecords,
    requester_id: str,
    **_ignored: Any,
) -> ToolResult:
    employee = records.employee(requester_id)
    if not employee:
        return ToolResult(False, "", error=f"Unknown employee id {requester_id}")
    manager = records.directory_entry(employee["manager_id"]) if employee["manager_id"] else None
    summary = (
        f"Requester profile: {employee['full_name']} ({employee['employee_id']}), "
        f"{employee['job_title']} at level {employee['level']} in {employee['department']}, "
        f"based in {employee['location']} ({employee['timezone']}), "
        f"{employee['employment_type']}, working pattern {employee['work_pattern']}, "
        f"hired {employee['hire_date']}, manager "
        f"{manager['full_name'] if manager else 'not recorded'}."
    )
    return ToolResult(
        ok=True,
        summary=summary,
        data={
            "kind": "profile",
            "employee": employee,
            "manager": manager,
        },
        policy_hint="",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
POLICY_SEARCH = "search_policies"


class ToolRegistry:
    """Holds the callable tools and enforces the identity boundary."""

    def __init__(self, records: HRRecords) -> None:
        self.records = records
        self._specs: dict[str, ToolSpec] = {}
        self.register(
            ToolSpec(
                name="get_leave_balance",
                description=(
                    "Current leave balances for the requester (annual/PTO, sick, "
                    "wellbeing, volunteering), including carried-over and pending days."
                ),
                parameters={"leave_type": "optional: annual|sick|wellbeing|volunteer"},
                handler=tool_get_leave_balance,
            )
        )
        self.register(
            ToolSpec(
                name="get_upcoming_holidays",
                description="Upcoming public holidays for the requester's work location.",
                parameters={"limit": "optional integer, default 5"},
                handler=tool_get_upcoming_holidays,
            )
        )
        self.register(
            ToolSpec(
                name="get_my_leave_requests",
                description="The requester's own leave requests and their approval status.",
                parameters={"status": "optional: approved|pending|declined"},
                handler=tool_get_my_leave_requests,
            )
        )
        self.register(
            ToolSpec(
                name="lookup_employee_directory",
                description=(
                    "Find a colleague in the company directory by name, team or role. "
                    "Returns public directory fields only."
                ),
                parameters={"query": "name, email, team or job title"},
                handler=tool_lookup_employee_directory,
                reads_personal_data=False,
            )
        )
        self.register(
            ToolSpec(
                name="get_my_profile",
                description=(
                    "The requester's own employment record: role, level, location, "
                    "working pattern, manager, hire date."
                ),
                parameters={},
                handler=tool_get_my_profile,
            )
        )

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    def describe(self) -> str:
        lines = [spec.describe() for spec in self._specs.values()]
        lines.append(
            f"- {POLICY_SEARCH}(query: what to look up) — search the policy "
            "knowledge base. Use this for anything the documents can answer."
        )
        return "\n".join(lines)

    def call(self, name: str, requester_id: str, arguments: dict[str, Any] | None = None) -> tuple[ToolResult, int]:
        """Invoke a tool. Returns (result, latency_ms). Never raises for bad input."""
        started = time.perf_counter()
        spec = self._specs.get(name)
        if spec is None:
            return (
                ToolResult(False, "", error=f"Unknown tool '{name}'"),
                int((time.perf_counter() - started) * 1000),
            )
        arguments = dict(arguments or {})
        # The identity is server-side. Any employee_id the model produced is
        # discarded here — this is the enforcement point, not the prompt.
        supplied_id = str(arguments.pop("employee_id", "") or "")
        if supplied_id and supplied_id.upper() != requester_id.upper() and spec.reads_personal_data:
            return (
                ToolResult(
                    False,
                    "",
                    error=(
                        "Access denied: the copilot can only return the requester's own "
                        "records. Ask the colleague directly, or contact People Operations."
                    ),
                ),
                int((time.perf_counter() - started) * 1000),
            )
        try:
            result = spec.handler(self.records, requester_id, **arguments)
        except AccessDenied as exc:
            result = ToolResult(False, "", error=str(exc))
        except TypeError as exc:
            result = ToolResult(False, "", error=f"Invalid arguments for {name}: {exc}")
        except Exception as exc:  # defensive: a tool must never 500 the request
            result = ToolResult(False, "", error=f"{name} failed: {exc.__class__.__name__}")
        return result, int((time.perf_counter() - started) * 1000)


# ---------------------------------------------------------------------------
# Deterministic router — the fallback when no planner LLM is available
# ---------------------------------------------------------------------------
ROUTING_RULES: list[tuple[re.Pattern[str], str, dict[str, Any]]] = [
    (re.compile(r"\b(balance|days? (do i have|left|remaining)|how much (leave|pto|vacation)"
                r"|remaining (leave|pto|holiday|vacation))\b", re.I),
     "get_leave_balance", {}),
    (re.compile(r"\b(sick (days?|leave) (do i|left|remaining|balance))\b", re.I),
     "get_leave_balance", {"leave_type": "sick"}),
    (re.compile(r"\b(next|upcoming|remaining|list of)\b.{0,24}\b(holidays?|bank holidays?|days? off)\b", re.I),
     "get_upcoming_holidays", {}),
    (re.compile(r"\b(public|company) holidays?\b.{0,30}\b(coming|next|upcoming|this year|left|remaining)\b", re.I),
     "get_upcoming_holidays", {}),
    (re.compile(r"\bwhen is the next (public |company )?holiday\b", re.I),
     "get_upcoming_holidays", {}),
    (re.compile(r"\b(my|pending|approved|declined) leave requests?\b", re.I),
     "get_my_leave_requests", {}),
    (re.compile(r"\b(who is|find|contact details for|email of|which team is)\b.{0,40}"
                r"\b(manager|colleague|person|engineer|lead|partner)\b", re.I),
     "lookup_employee_directory", {}),
    (re.compile(r"\b(who is my manager|what is my (role|level|location|working pattern)"
                r"|when did i (join|start))\b", re.I),
     "get_my_profile", {}),
]


def route_deterministically(question: str) -> tuple[str, dict[str, Any]] | None:
    """Regex router used when no LLM planner is available, or as a cross-check.

    Kept deliberately conservative: it fires only on unambiguous phrasings, and
    anything it does not recognise falls through to document retrieval.
    """
    for pattern, tool, arguments in ROUTING_RULES:
        if pattern.search(question):
            args = dict(arguments)
            if tool == "lookup_employee_directory":
                name = _extract_person(question)
                if not name:
                    continue
                args["query"] = name
            if tool == "get_leave_balance" and "leave_type" not in args:
                for alias, code in LEAVE_TYPE_ALIASES.items():
                    if re.search(rf"\b{alias}\b", question, re.I):
                        args["leave_type"] = code
                        break
            return tool, args
    return None


PERSON_RE = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b")


def _extract_person(question: str) -> str:
    match = PERSON_RE.search(question)
    if match:
        return match.group(1)
    tail = re.search(r"\bfor ([a-z ]{3,30})$", question.strip(), re.I)
    return tail.group(1).strip() if tail else ""
