"""Generate the synthetic HR record CSVs used by the copilot's tools.

The generator is seeded so that re-running it reproduces byte-identical files.
It is committed alongside the data so the provenance of every row is auditable —
one of the governance requirements in `docs/RESPONSIBLE_AI.md`.

Usage:
    python scripts/generate_hr_records.py
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 20260801
AS_OF = date(2026, 8, 1)
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "hr_records"

FIRST_NAMES = [
    "Amara", "Priya", "Diego", "Mei", "Tomas", "Nadia", "Kwame", "Sofia",
    "Rahul", "Elena", "Yusuf", "Hana", "Lucas", "Ingrid", "Omar", "Aiko",
    "Farida", "Niko", "Beatriz", "Sanjay", "Clara", "Idris", "Lena", "Ravi",
]
LAST_NAMES = [
    "Okafor", "Venkatesh", "Ramirez", "Chen", "Novak", "Haddad", "Mensah",
    "Rossi", "Iyer", "Petrova", "Demir", "Sato", "Almeida", "Larsen", "Farouk",
    "Tanaka", "Nasser", "Berg", "Lima", "Kulkarni", "Dubois", "Bello",
    "Sorensen", "Menon",
]

# (department, job_title, level, location, timezone)
ROLES = [
    ("Engineering", "Senior Software Engineer", "L5", "Austin, US", "America/Chicago"),
    ("Engineering", "Software Engineer", "L3", "Bengaluru, IN", "Asia/Kolkata"),
    ("Engineering", "Staff Software Engineer", "L6", "Austin, US", "America/Chicago"),
    ("Engineering", "Site Reliability Engineer", "L4", "London, UK", "Europe/London"),
    ("Engineering", "Engineering Manager", "L6", "London, UK", "Europe/London"),
    ("Data & AI", "Machine Learning Engineer", "L4", "Bengaluru, IN", "Asia/Kolkata"),
    ("Data & AI", "Data Engineer", "L4", "Austin, US", "America/Chicago"),
    ("Data & AI", "Head of Data & AI", "L7", "Austin, US", "America/Chicago"),
    ("Product", "Product Manager", "L5", "London, UK", "Europe/London"),
    ("Product", "Product Designer", "L4", "Bengaluru, IN", "Asia/Kolkata"),
    ("People Operations", "HR Business Partner", "L4", "London, UK", "Europe/London"),
    ("People Operations", "Head of People Operations", "L7", "London, UK", "Europe/London"),
    ("Finance", "Finance Analyst", "L3", "Austin, US", "America/Chicago"),
    ("Finance", "Finance Operations Lead", "L5", "Austin, US", "America/Chicago"),
    ("Customer Support", "Support Engineer", "L3", "Bengaluru, IN", "Asia/Kolkata"),
    ("Customer Support", "Support Team Lead", "L5", "Bengaluru, IN", "Asia/Kolkata"),
    ("Information Security", "Security Engineer", "L5", "London, UK", "Europe/London"),
    ("Information Security", "Security Duty Officer", "L4", "Austin, US", "America/Chicago"),
]

LEAVE_TYPES = [
    # (code, label, entitlement for a full-time employee)
    ("ANNUAL", "Annual leave (PTO)", 24.0),
    ("SICK", "Sick leave", 12.0),
    ("WELLBEING", "Wellbeing days", 2.0),
    ("VOLUNTEER", "Volunteering day", 1.0),
]

HOLIDAYS = {
    "US": [
        ("2026-01-01", "New Year's Day", "public"),
        ("2026-01-19", "Martin Luther King Jr. Day", "public"),
        ("2026-02-16", "Presidents' Day", "optional"),
        ("2026-05-25", "Memorial Day", "public"),
        ("2026-06-19", "Juneteenth", "public"),
        ("2026-07-03", "Independence Day (observed)", "public"),
        ("2026-09-07", "Labor Day", "public"),
        ("2026-10-12", "Indigenous Peoples' Day", "optional"),
        ("2026-11-11", "Veterans Day", "optional"),
        ("2026-11-26", "Thanksgiving Day", "public"),
        ("2026-11-27", "Day after Thanksgiving", "public"),
        ("2026-12-24", "Christmas Eve", "optional"),
        ("2026-12-25", "Christmas Day", "public"),
        ("2026-12-31", "New Year's Eve", "optional"),
    ],
    "IN": [
        ("2026-01-01", "New Year's Day", "public"),
        ("2026-01-26", "Republic Day", "public"),
        ("2026-03-04", "Holi", "public"),
        ("2026-04-03", "Good Friday", "optional"),
        ("2026-05-01", "Labour Day", "optional"),
        ("2026-08-15", "Independence Day", "public"),
        ("2026-08-26", "Ganesh Chaturthi", "optional"),
        ("2026-10-02", "Gandhi Jayanti", "public"),
        ("2026-10-20", "Dussehra", "public"),
        ("2026-11-08", "Diwali", "public"),
        ("2026-11-09", "Diwali (second day)", "optional"),
        ("2026-12-25", "Christmas Day", "public"),
        ("2026-12-31", "New Year's Eve", "optional"),
        ("2026-04-14", "Regional New Year", "optional"),
    ],
    "UK": [
        ("2026-01-01", "New Year's Day", "public"),
        ("2026-04-03", "Good Friday", "public"),
        ("2026-04-06", "Easter Monday", "public"),
        ("2026-05-04", "Early May Bank Holiday", "public"),
        ("2026-05-25", "Spring Bank Holiday", "public"),
        ("2026-08-31", "Summer Bank Holiday", "public"),
        ("2026-12-25", "Christmas Day", "public"),
        ("2026-12-28", "Boxing Day (observed)", "public"),
        ("2026-01-02", "New Year Holiday (Scotland)", "optional"),
        ("2026-03-17", "St Patrick's Day (NI)", "optional"),
        ("2026-07-13", "Battle of the Boyne (NI, observed)", "optional"),
        ("2026-11-30", "St Andrew's Day (Scotland)", "optional"),
        ("2026-12-24", "Christmas Eve", "optional"),
        ("2026-12-31", "New Year's Eve", "optional"),
    ],
}

COUNTRY_BY_LOCATION = {
    "Austin, US": "US",
    "Bengaluru, IN": "IN",
    "London, UK": "UK",
}


def build_employees(rng: random.Random) -> list[dict]:
    employees: list[dict] = []
    for i in range(24):
        dept, title, level, location, tz = ROLES[i % len(ROLES)]
        first = FIRST_NAMES[i]
        last = LAST_NAMES[i]
        emp_id = f"EMP-{1001 + i}"
        hire = date(2019, 1, 1) + timedelta(days=rng.randint(0, 2400))
        employees.append(
            {
                "employee_id": emp_id,
                "full_name": f"{first} {last}",
                "work_email": f"{first.lower()}.{last.lower()}@northwind-example.com",
                "department": dept,
                "job_title": title,
                "level": level,
                "location": location,
                "country": COUNTRY_BY_LOCATION[location],
                "timezone": tz,
                "employment_type": "Full-time" if i % 9 else "Part-time (0.6 FTE)",
                "work_pattern": ["Hybrid", "Hybrid", "Fully remote", "Office-based"][i % 4],
                "hire_date": hire.isoformat(),
                "manager_id": "",
                "status": "Active",
            }
        )

    # Assign managers: heads report to nobody, everyone else to a manager in
    # their own department where one exists, otherwise to the Head of People Ops.
    leaders = {e["department"]: e["employee_id"] for e in employees if e["level"] in ("L6", "L7")}
    ceo = employees[7]["employee_id"]  # Head of Data & AI acts as the top node
    for emp in employees:
        if emp["employee_id"] == ceo:
            continue
        lead = leaders.get(emp["department"], ceo)
        emp["manager_id"] = ceo if lead == emp["employee_id"] else lead
    return employees


def build_leave_balances(employees: list[dict], rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    for emp in employees:
        fte = 0.6 if emp["employment_type"].startswith("Part-time") else 1.0
        for code, label, full_entitlement in LEAVE_TYPES:
            entitled = round(full_entitlement * fte, 1)
            if code == "ANNUAL":
                taken = round(rng.uniform(2, entitled - 3), 1)
                pending = round(rng.choice([0, 0, 1, 2, 3, 5]) * fte, 1)
                carried = round(rng.choice([0, 0, 0, 2, 5]) * fte, 1)
            elif code == "SICK":
                taken = float(rng.randint(0, 6))
                pending, carried = 0.0, 0.0
            else:
                taken = float(rng.randint(0, int(entitled)))
                pending, carried = 0.0, 0.0
            available = round(entitled + carried - taken - pending, 1)
            rows.append(
                {
                    "employee_id": emp["employee_id"],
                    "leave_type": code,
                    "leave_type_label": label,
                    "leave_year": "2026",
                    "entitled_days": entitled,
                    "carried_over_days": carried,
                    "taken_days": taken,
                    "pending_days": pending,
                    "available_days": available,
                    "carryover_expires_on": "2027-03-31" if code == "ANNUAL" else "",
                    "as_of_date": AS_OF.isoformat(),
                }
            )
    return rows


def build_leave_requests(employees: list[dict], rng: random.Random) -> list[dict]:
    statuses = ["Approved", "Approved", "Approved", "Pending", "Declined"]
    rows: list[dict] = []
    for i in range(30):
        emp = employees[rng.randrange(len(employees))]
        start = AS_OF + timedelta(days=rng.randint(-120, 120))
        days = rng.choice([1, 1, 2, 3, 5, 5, 10])
        status = statuses[i % len(statuses)]
        rows.append(
            {
                "request_id": f"LR-2026-{2001 + i}",
                "employee_id": emp["employee_id"],
                "leave_type": rng.choice(["ANNUAL", "ANNUAL", "SICK", "WELLBEING"]),
                "start_date": start.isoformat(),
                "end_date": (start + timedelta(days=days - 1)).isoformat(),
                "working_days": days,
                "status": status,
                "submitted_on": (start - timedelta(days=rng.randint(3, 25))).isoformat(),
                "approver_id": emp["manager_id"] or emp["employee_id"],
                "decided_on": "" if status == "Pending" else (start - timedelta(days=rng.randint(1, 3))).isoformat(),
            }
        )
    rows.sort(key=lambda r: r["request_id"])
    return rows


def build_holidays() -> list[dict]:
    rows: list[dict] = []
    for country, entries in HOLIDAYS.items():
        for day, name, kind in sorted(entries):
            rows.append(
                {
                    "country": country,
                    "holiday_date": day,
                    "holiday_name": name,
                    "type": kind,
                    "leave_year": "2026",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows):>3} rows -> {path.relative_to(path.parents[2])}")


def main() -> None:
    rng = random.Random(SEED)
    employees = build_employees(rng)
    write_csv(OUT_DIR / "employees.csv", employees)
    write_csv(OUT_DIR / "leave_balances.csv", build_leave_balances(employees, rng))
    write_csv(OUT_DIR / "leave_requests.csv", build_leave_requests(employees, rng))
    write_csv(OUT_DIR / "holidays.csv", build_holidays())


if __name__ == "__main__":
    main()
