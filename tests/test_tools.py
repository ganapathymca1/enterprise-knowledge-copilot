"""Tool layer: routing, execution and the access-control boundary."""

from __future__ import annotations

import pytest

from backend.app.tools import route_deterministically

REQUESTER = "EMP-1002"


@pytest.mark.parametrize(
    "question,expected_tool",
    [
        ("How many annual leave days do I have left?", "get_leave_balance"),
        ("What is my remaining PTO balance?", "get_leave_balance"),
        ("What are my next public holidays?", "get_upcoming_holidays"),
        ("When is the next company holiday?", "get_upcoming_holidays"),
        ("Do I have any pending leave requests?", "get_my_leave_requests"),
        ("Who is my manager?", "get_my_profile"),
    ],
)
def test_router_recognises_record_questions(question, expected_tool):
    decision = route_deterministically(question)
    assert decision is not None, f"no route for {question!r}"
    assert decision[0] == expected_tool


@pytest.mark.parametrize(
    "question",
    [
        "What is the leave policy?",
        "How much notice do I need to give?",
        "Summarize our leave policy",
        "What is the escalation process for incidents?",
    ],
)
def test_router_leaves_policy_questions_to_retrieval(question):
    """The router is deliberately conservative: unknown phrasing means retrieval."""
    assert route_deterministically(question) is None


def test_leave_balance_returns_records_and_a_policy_hint(tools):
    result, latency_ms = tools.call("get_leave_balance", REQUESTER, {"leave_type": "annual"})
    assert result.ok
    assert "Annual leave" in result.summary
    assert result.data["kind"] == "leave_balance"
    assert result.data["balances"][0]["leave_type"] == "ANNUAL"
    # The hint is what lets a number be answered together with its policy.
    assert "carryover" in result.policy_hint or "carry" in result.policy_hint
    assert latency_ms >= 0


def test_part_time_entitlement_is_prorated(tools):
    result, _ = tools.call("get_leave_balance", "EMP-1001", {"leave_type": "annual"})
    assert result.ok
    entitled = float(result.data["balances"][0]["entitled_days"])
    assert entitled < 24.0, "a 0.6 FTE employee must not show the full-time entitlement"


def test_holidays_are_scoped_to_the_employee_location(tools):
    india, _ = tools.call("get_upcoming_holidays", "EMP-1002", {"limit": 3})
    us, _ = tools.call("get_upcoming_holidays", "EMP-1001", {"limit": 3})
    assert india.data["country"] == "IN"
    assert us.data["country"] == "US"
    assert india.data["holidays"] != us.data["holidays"]


def test_directory_exposes_only_public_fields(tools):
    result, _ = tools.call("lookup_employee_directory", REQUESTER, {"query": "Sofia"})
    assert result.ok
    person = result.data["people"][0]
    for forbidden in ("level", "hire_date", "employment_type", "available_days"):
        assert forbidden not in person


def test_another_employees_records_are_denied(tools):
    """The identity is server-side: a hallucinated employee_id must not work."""
    result, _ = tools.call("get_leave_balance", REQUESTER, {"employee_id": "EMP-1005"})
    assert not result.ok
    assert "Access denied" in result.error
    assert "own records" in result.error


def test_own_employee_id_is_accepted(tools):
    result, _ = tools.call("get_leave_balance", REQUESTER, {"employee_id": REQUESTER})
    assert result.ok


def test_unknown_tool_and_bad_arguments_fail_softly(tools):
    unknown, _ = tools.call("delete_everything", REQUESTER, {})
    assert not unknown.ok and "Unknown tool" in unknown.error

    bad, _ = tools.call("get_upcoming_holidays", REQUESTER, {"limit": "not-a-number"})
    assert bad.ok, "an unparseable limit should fall back to the default, not raise"


def test_registry_description_lists_the_policy_search_option(tools):
    described = tools.describe()
    assert "search_policies" in described
    for name in tools.names:
        assert name in described
