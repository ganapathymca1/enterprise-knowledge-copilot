---
doc_id: SEC-POL-011
title: Information Security and Acceptable Use Policy
category: Security
owner: Information Security
version: "7.1"
effective_date: 2026-02-01
review_date: 2026-08-31
audience: All employees, contractors
---

# Information Security and Acceptable Use Policy

## 1. Data classification

| Class | Examples | Handling |
|---|---|---|
| **Public** | Marketing site, published docs | No restriction |
| **Internal** | Wiki pages, team plans, this policy | Employees and contractors only |
| **Confidential** | Customer data, contracts, source code, salary data | Need-to-know; encrypted at rest and in transit |
| **Restricted** | Credentials, encryption keys, security findings, personal data of customers | Named individuals only; access logged and reviewed quarterly |

Personal data of employees or customers is always at least Confidential.

## 2. Accounts and authentication

- Single sign-on with **multi-factor authentication is mandatory** for all
  company systems. Hardware keys are issued to engineering and admin roles.
- Never share credentials. Use the company password manager for anything that
  cannot use SSO.
- Service accounts must have named owners and credentials rotated at least
  every 90 days.
- Access is reviewed quarterly and revoked immediately on role change or exit.

## 3. Devices

Company laptops and mobile devices must run the MDM agent, full-disk
encryption, an automatic screen lock of 5 minutes or less, and current
operating system patches. Lost or stolen devices must be reported to
`security@northwind-example.com` **within 1 hour** of discovery.

## 4. Acceptable use

Company systems are for business use, with reasonable personal use permitted.
Employees must not use company systems to store personal media libraries, run
unauthorised commercial activity, access unlawful material, or bypass security
controls. Company systems may be monitored for security purposes in line with
local law; monitoring is proportionate and never targets an individual without
a documented investigation.

## 5. Use of generative AI tools

- **Approved tools only.** The current list is maintained on the security wiki.
  The internal Enterprise Knowledge Copilot is approved for Internal and
  Confidential content.
- **Never paste Restricted data**, credentials, personal data of customers, or
  unreleased financial information into any external AI service.
- AI-generated code must be reviewed by a human before merge and scanned by the
  standard pipeline.
- AI output must not be presented as a final authority for legal, HR,
  compensation, medical or security decisions. Always verify against the
  source document.

## 6. Phishing and social engineering

Report suspected phishing with the **Report Phish** button in the mail client.
Never approve an MFA prompt you did not initiate. The IT team will never ask
for a password or an MFA code. Requests to change bank details always require
out-of-band verification by phone with a known contact.

## 7. Software and third parties

New software or SaaS handling company data requires a security review and a
signed data processing agreement before use. Open-source components must have a
compatible licence and pass the dependency scanner.

## 8. Reporting a security incident

Report anything suspicious to `security@northwind-example.com` or page the
Security Duty Officer. Security incidents follow the Incident Management and
Escalation Procedure (OPS-POL-004), with the Security Duty Officer paged at
step 1 and confirmed personal-data breaches escalated to Legal within the
**72-hour** regulatory notification window.

## 9. Training and compliance

Security and privacy training is mandatory within the first 10 working days and
annually thereafter. Simulated phishing exercises run quarterly. Repeated
policy breaches are handled under the Code of Conduct (HR-POL-010).
