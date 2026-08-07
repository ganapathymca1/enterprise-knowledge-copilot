---
doc_id: OPS-POL-004
title: Incident Management and Escalation Procedure
category: Operations
owner: Site Reliability Engineering
version: "5.1"
effective_date: 2026-03-01
review_date: 2026-09-30
audience: Engineering, Support, Duty Managers
---

# Incident Management and Escalation Procedure

## 1. What counts as an incident

An incident is any unplanned interruption or degradation of a production
service, or an event that puts customer data at risk. Anyone in the company may
declare an incident — declaring one is never penalised, and a false alarm is
always preferable to a silent outage.

## 2. Severity levels

| Severity | Definition | Response target | Update cadence |
|---|---|---|---|
| **SEV1** | Complete outage or confirmed data breach affecting many customers | 15 minutes | Every 30 minutes |
| **SEV2** | Major feature unavailable or severe degradation; no workaround | 30 minutes | Every 60 minutes |
| **SEV3** | Partial degradation with a workaround; limited customer impact | 4 business hours | Daily |
| **SEV4** | Minor issue, cosmetic defect, or single-user impact | 2 business days | On change |

Severity is set by the Incident Commander and can be revised at any time. When
in doubt, start at the higher severity and downgrade later.

## 3. Declaring an incident

1. Post in the `#incident-response` channel using `/incident declare`.
2. The bot creates a numbered incident channel and a ServiceDesk record.
3. Page the on-call engineer for the affected service through the paging tool.
4. The first responder becomes **Incident Commander (IC)** until formally
   handed over.

## 4. Roles during an incident

- **Incident Commander** — owns coordination and decisions. Does not debug.
- **Operations Lead** — performs the technical investigation and mitigation.
- **Communications Lead** — writes customer-facing and internal updates.
- **Scribe** — records the timeline in the incident channel.

For SEV3 and SEV4 one person may hold all roles. SEV1 requires a dedicated IC
who is not also the Operations Lead.

## 5. Escalation matrix

Escalate when a response target is missed, when the current responder is
blocked, or when impact grows.

| Step | Who | When | Channel |
|---|---|---|---|
| 1 | Service on-call engineer | Immediately at declaration | Paging tool |
| 2 | Secondary on-call / team lead | No acknowledgement within 10 minutes | Paging tool + phone |
| 3 | Engineering Manager for the service | 30 minutes with no mitigation, or any SEV1 | Phone |
| 4 | Duty Director (24×7 rota) | 60 minutes on SEV1, or customer commitment at risk | Phone |
| 5 | VP Engineering and Head of Support | 2 hours on SEV1, or regulatory exposure | Phone + email |
| 6 | Executive team and Legal | Confirmed data breach, or media/regulator contact | Phone bridge |

Security incidents follow the same matrix but **must additionally page the
Security Duty Officer at step 1** and follow the Information Security Policy
(SEC-POL-009). Suspected personal-data breaches have a regulatory notification
clock of **72 hours** and must reach Legal at step 6 without delay.

## 6. Communication

- Internal updates go in the incident channel at the cadence in Section 2.
- Customer-facing updates are published on the status page by the
  Communications Lead only. No one else posts externally.
- Employees who receive customer questions during an incident should link to
  the status page rather than speculate.

## 7. Resolution and post-incident review

An incident is resolved when customer impact has ended and monitoring is clean
for 30 minutes. A **blameless post-incident review** is mandatory for every
SEV1 and SEV2 and must be published within **5 working days**. It records the
timeline, contributing factors, customer impact, and action items with named
owners and due dates. Action items are tracked to closure in the engineering
backlog and reviewed monthly by the SRE lead.

## 8. On-call expectations

- On-call shifts run for one week, handing over on Tuesday at 10:00 local time.
- Acknowledge a page within **10 minutes**, 24×7.
- On-call compensation is described in the Compensation Policy (HR-POL-007).
- After a night page lasting more than 2 hours, the engineer may take the
  following morning off; this is not deducted from leave.

## 9. Non-production and workplace incidents

Workplace safety incidents, harassment reports, and facilities emergencies are
**not** covered here — see the Code of Conduct and Grievance Procedure
(HR-POL-010) and contact `people-ops@northwind-example.com`.
