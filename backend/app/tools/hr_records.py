"""Read-only access to the synthetic HR record store.

In a real deployment this class is the seam where the HRIS (Workday,
SuccessFactors, PeopleHub…) is called over an authenticated service account.
Keeping every CSV read behind one narrow interface means swapping in that
client changes one file and no tool logic.

Access control lives here rather than in the tools, so that no tool can
accidentally reach a field it should not see: the store exposes *own-record*
methods that require the requester id, and a separate *directory* method that
returns only the fields published in the company directory.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

# Fields any employee may see about any colleague. Everything else — salary,
# balances, requests, level, hire date — is own-record only.
DIRECTORY_FIELDS = (
    "employee_id",
    "full_name",
    "work_email",
    "department",
    "job_title",
    "location",
    "timezone",
    "work_pattern",
    "manager_id",
)


class AccessDenied(PermissionError):
    """Raised when a tool asks for personal data belonging to someone else."""


@dataclass
class HRRecords:
    employees: list[dict[str, str]]
    leave_balances: list[dict[str, str]]
    leave_requests: list[dict[str, str]]
    holidays: list[dict[str, str]]

    # -- loading ----------------------------------------------------------
    @classmethod
    def from_directory(cls, records_dir: Path) -> "HRRecords":
        def read(name: str) -> list[dict[str, str]]:
            path = records_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Missing HR record file: {path}")
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))

        return cls(
            employees=read("employees.csv"),
            leave_balances=read("leave_balances.csv"),
            leave_requests=read("leave_requests.csv"),
            holidays=read("holidays.csv"),
        )

    # -- lookups ----------------------------------------------------------
    def employee(self, employee_id: str) -> dict[str, str] | None:
        for row in self.employees:
            if row["employee_id"].upper() == (employee_id or "").upper():
                return row
        return None

    def directory_entry(self, employee_id: str) -> dict[str, str] | None:
        row = self.employee(employee_id)
        return {key: row[key] for key in DIRECTORY_FIELDS} if row else None

    def search_directory(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        needle = (query or "").strip().lower()
        if not needle:
            return []
        hits: list[tuple[int, dict[str, str]]] = []
        for row in self.employees:
            haystack = " ".join(
                (row["full_name"], row["work_email"], row["department"], row["job_title"])
            ).lower()
            if needle in haystack:
                # Exact name matches rank above substring hits in other fields.
                rank = 0 if needle in row["full_name"].lower() else 1
                hits.append((rank, {key: row[key] for key in DIRECTORY_FIELDS}))
        hits.sort(key=lambda item: (item[0], item[1]["full_name"]))
        return [row for _rank, row in hits[:limit]]

    def balances(self, employee_id: str, leave_type: str | None = None) -> list[dict[str, str]]:
        rows = [
            row
            for row in self.leave_balances
            if row["employee_id"].upper() == employee_id.upper()
        ]
        if leave_type:
            wanted = leave_type.strip().upper()
            rows = [row for row in rows if row["leave_type"] == wanted]
        return rows

    def requests(self, employee_id: str, status: str | None = None) -> list[dict[str, str]]:
        rows = [
            row
            for row in self.leave_requests
            if row["employee_id"].upper() == employee_id.upper()
        ]
        if status:
            wanted = status.strip().lower()
            rows = [row for row in rows if row["status"].lower() == wanted]
        return sorted(rows, key=lambda row: row["start_date"])

    def upcoming_holidays(
        self,
        country: str,
        *,
        on_or_after: date,
        limit: int = 5,
        include_optional: bool = True,
    ) -> list[dict[str, str]]:
        rows = [
            row
            for row in self.holidays
            if row["country"].upper() == country.upper()
            and datetime.strptime(row["holiday_date"], "%Y-%m-%d").date() >= on_or_after
            and (include_optional or row["type"] == "public")
        ]
        rows.sort(key=lambda row: row["holiday_date"])
        return rows[:limit]

    def team_of(self, manager_id: str) -> list[dict[str, str]]:
        return [
            {key: row[key] for key in DIRECTORY_FIELDS}
            for row in self.employees
            if row["manager_id"].upper() == manager_id.upper()
        ]


@lru_cache(maxsize=4)
def load_records(records_dir: Path) -> HRRecords:
    """Cached loader — the CSVs are immutable for the lifetime of the process."""
    return HRRecords.from_directory(records_dir)
