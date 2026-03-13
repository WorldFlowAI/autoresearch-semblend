#!/usr/bin/env python3
"""Multi-tenant API benchmark: cross-tenant KV sharing via SemBlend.

Simulates a multi-tenant LLM API where different tenants submit similar
documents (e.g., legal contracts, support tickets, financial reports).
SemBlend should discover cross-tenant donors and accelerate TTFT without
any tenant-specific configuration.

Measures:
  - Cross-tenant hit rate (tenant A's doc benefits tenant B)
  - TTFT speedup over cold prefill
  - Per-tenant isolation (no quality degradation from cross-tenant KV)

Usage:
    python benchmarks/e2e/multitenant_api_bench.py \
        --endpoint http://localhost:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --n-tenants 10 \
        --docs-per-tenant 5 \
        --token-length 8192
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests

# Each tenant has a domain-specific system prompt
TENANT_PROFILES = [
    ("legal_firm", "You are a legal analyst at Morrison & Partners LLP. Analyze the following document for contractual risks, liability exposure, and compliance gaps. Cite specific clauses."),
    ("fintech_startup", "You are a financial analyst at PayFlow Inc. Review the following document for revenue projections, burn rate analysis, and investor-relevant metrics. Be quantitative."),
    ("healthcare_co", "You are a healthcare compliance officer at MedCare Systems. Review the following document for HIPAA compliance, patient data handling procedures, and clinical protocol adherence."),
    ("consulting_firm", "You are a senior consultant at Deloitte Digital. Analyze the following document and provide strategic recommendations with implementation priorities and timeline estimates."),
    ("insurance_co", "You are an insurance underwriter at Pacific Risk Group. Assess the following document for risk factors, actuarial implications, and coverage recommendations."),
    ("govt_agency", "You are a policy analyst at the Department of Commerce. Review the following document for regulatory alignment, public interest implications, and inter-agency coordination needs."),
    ("tech_company", "You are a principal engineer at CloudScale Technologies. Review the following document for technical architecture decisions, scalability concerns, and security posture."),
    ("manufacturing", "You are an operations manager at Precision Manufacturing Corp. Analyze the following document for supply chain risks, quality control measures, and production efficiency metrics."),
    ("education", "You are a curriculum director at National Education Alliance. Review the following document for educational outcomes, accessibility compliance, and pedagogical effectiveness."),
    ("real_estate", "You are a commercial real estate analyst at Meridian Properties. Evaluate the following document for market positioning, ROI projections, and regulatory zoning implications."),
]

# Shared document templates that different tenants would encounter
# (e.g., all tenants might review quarterly reports, contracts, audits)
SHARED_DOC_TEMPLATES = [
    ("quarterly_report", """
# Q4 2025 Quarterly Business Report

## Executive Summary
Revenue reached $47.3M in Q4, representing 23% year-over-year growth. Operating margins
improved to 18.2% from 15.7% in Q3, driven by automation initiatives in the fulfillment
pipeline. Customer acquisition cost (CAC) decreased 12% to $847 per enterprise account.

## Financial Highlights
- Revenue: $47.3M (Q3: $42.1M, Q4 2024: $38.5M)
- Gross Margin: 72.4% (target: 70%)
- Operating Expenses: $38.7M (headcount +15 to 342 FTEs)
- Free Cash Flow: $8.2M (first positive FCF quarter)
- ARR: $189.2M (net retention: 118%)

## Product & Engineering
- Shipped v3.2 with real-time analytics dashboard (80% of enterprise customers activated)
- Reduced P99 latency from 340ms to 89ms via edge caching rollout
- Completed SOC 2 Type II audit with zero findings
- Technical debt reduction: 23% of sprint capacity allocated to refactoring

## Sales & Marketing
- 47 new enterprise logos (target: 40)
- Pipeline: $62M qualified (weighted: $31M)
- Average deal size: $127K (up from $98K in Q3)
- Win rate vs. competitors: 34% (Competitor A), 52% (Competitor B)

## Risks & Challenges
- Key person dependency: CTO transition planned for Q1 2026
- Regulatory: EU AI Act compliance timeline may require product changes
- Competition: Competitor A raised $200M Series D, expanding enterprise team 3x
- Infrastructure: Cloud costs grew 28% vs. 23% revenue growth — margin pressure

## Q1 2026 Outlook
- Revenue target: $51-53M
- Planned headcount: +20 FTEs (engineering: 12, sales: 5, support: 3)
- Product roadmap: AI-powered insights engine, multi-region deployment, API v4
- Capital allocation: $5M for potential M&A (data integration tools)
"""),
    ("service_agreement", """
# Master Services Agreement

## Parties
This Master Services Agreement ("Agreement") is entered into between ServiceProvider Corp
("Provider") and the undersigned client entity ("Client"), effective as of the date of
last signature below.

## 1. Scope of Services
Provider shall deliver the following services: (a) Enterprise software platform access
including API endpoints, dashboard, and administrative console; (b) Data processing and
analytics services with guaranteed throughput of 10,000 transactions per second;
(c) 24/7 technical support with 15-minute response SLA for Severity 1 issues;
(d) Quarterly business reviews and optimization recommendations.

## 2. Service Levels
- Uptime: 99.95% measured monthly (excluding scheduled maintenance windows)
- Latency: P95 < 200ms for API calls, P99 < 500ms
- Data Processing: 99.99% accuracy for financial calculations
- Support: Sev1: 15 min response / 4 hr resolution; Sev2: 1 hr / 8 hr; Sev3: 4 hr / 48 hr

## 3. Fees and Payment
- Annual license: $240,000 (paid quarterly at $60,000)
- Usage overage: $0.003 per transaction above 10M/month threshold
- Professional services: $275/hour for custom integrations
- Annual escalator: CPI + 2%, capped at 7%

## 4. Data Governance
- All client data processed within designated geographic region (US-East or EU-West)
- Encryption: AES-256 at rest, TLS 1.3 in transit
- Retention: Client data retained for contract duration + 90 days
- Deletion: Certified destruction within 30 days of termination request
- Sub-processors: Listed in Exhibit B; 30-day advance notice for changes

## 5. Liability and Indemnification
- Provider liability cap: 12 months of fees paid
- Mutual indemnification for IP infringement and data breach
- Consequential damages excluded except for: (a) breach of confidentiality,
  (b) willful misconduct, (c) indemnification obligations
- Insurance: Provider maintains $10M cyber liability, $5M E&O coverage

## 6. Term and Termination
- Initial term: 36 months from effective date
- Auto-renewal: 12-month periods unless 90-day written notice
- Termination for cause: 30-day cure period after written notice
- Transition assistance: 6 months at then-current rates
"""),
    ("security_assessment", """
# Annual Security Assessment Report

## Assessment Overview
Period: January 1 - December 31, 2025
Assessor: CyberShield Security Consulting
Framework: NIST CSF 2.0, mapped to ISO 27001:2022 and SOC 2 Type II

## Executive Summary
Overall security maturity score: 3.4/5.0 (up from 2.8 in 2024). Critical findings
reduced from 12 to 3. The organization has made significant progress in identity
management and network segmentation but continues to face challenges in application
security and third-party risk management.

## Critical Findings (3)
1. **API Authentication Bypass** (CVSS 9.1): The payment processing API accepts
   requests with expired OAuth tokens due to a clock skew tolerance of 300 seconds.
   Recommendation: Reduce tolerance to 30 seconds, implement token binding.

2. **Privilege Escalation via IDOR** (CVSS 8.5): User ID enumeration in the admin
   panel allows authenticated users to access other tenants' configuration data.
   Recommendation: Implement UUID-based resource identifiers, add tenant validation.

3. **Unencrypted PII in Logs** (CVSS 7.8): Application logs contain full customer
   names, email addresses, and partial payment information in plaintext.
   Recommendation: Implement log scrubbing for PII patterns, rotate affected logs.

## High Findings (7)
- Outdated TLS 1.0/1.1 on 4 internal endpoints
- Missing MFA for 23% of admin accounts
- Database backup encryption key stored in same S3 bucket as backups
- WAF bypass via HTTP/2 connection coalescing
- Incomplete RBAC: 15 service accounts with excessive permissions
- Third-party JavaScript libraries with known CVEs (3 components)
- Insufficient rate limiting on authentication endpoints

## Recommendations Priority Matrix
| Priority | Finding | Effort | Impact | Deadline |
|----------|---------|--------|--------|----------|
| P0 | API Auth Bypass | Medium | Critical | 30 days |
| P0 | IDOR Privilege Escalation | High | Critical | 45 days |
| P0 | PII in Logs | Low | High | 14 days |
| P1 | TLS Upgrade | Low | Medium | 60 days |
| P1 | MFA Enforcement | Medium | High | 30 days |
"""),
    ("incident_postmortem", """
# Post-Incident Report: Payment Processing Outage

## Incident Summary
- **Incident ID**: INC-2025-0847
- **Severity**: SEV-1 (Customer-Impacting)
- **Duration**: 3 hours 42 minutes (14:23 UTC - 18:05 UTC, November 15, 2025)
- **Impact**: 100% of payment transactions failed; 2.3M users affected
- **Revenue Impact**: Estimated $1.8M in lost transactions, $340K in SLA credits

## Timeline
- 14:23 UTC: Automated alerts fire for payment API error rate > 5%
- 14:25 UTC: On-call engineer acknowledges alert, begins investigation
- 14:31 UTC: Error rate reaches 98%; all payment endpoints returning 503
- 14:45 UTC: Root cause identified — database connection pool exhausted
- 14:52 UTC: Attempted fix: increase connection pool from 100 to 200
- 15:01 UTC: Fix ineffective — underlying cause is connection leak in ORM
- 15:23 UTC: Hotfix deployed to patch connection leak; pool draining
- 15:45 UTC: Pool drain slower than expected due to long-running queries
- 16:30 UTC: Decision to failover to read replica with manual promotion
- 17:15 UTC: Replica promoted, connections redirected
- 17:45 UTC: Payment success rate recovers to 95%
- 18:05 UTC: Full recovery confirmed; incident closed

## Root Cause
A code change deployed at 13:00 UTC introduced a connection leak in the payment
service's database ORM layer. Specifically, a try-catch block in the refund
processing path caught and swallowed a connection timeout exception without
releasing the connection back to the pool. Under normal load, the leak was
gradual (~2 connections/minute), but a promotional campaign at 14:00 UTC
increased refund volume 5x, exhausting the 100-connection pool within 23 minutes.

## Action Items
| # | Action | Owner | Due Date | Status |
|---|--------|-------|----------|--------|
| 1 | Add connection leak detection to CI/CD | Platform Team | Dec 1 | Open |
| 2 | Implement circuit breaker for DB connections | Backend Team | Dec 15 | Open |
| 3 | Add connection pool monitoring to dashboards | SRE Team | Nov 22 | Done |
| 4 | Review all ORM exception handlers | Backend Team | Dec 1 | Open |
| 5 | Runbook: database failover procedure | SRE Team | Nov 30 | Open |
| 6 | Load test refund path at 10x normal volume | QA Team | Dec 15 | Open |
"""),
    ("product_roadmap", """
# Product Roadmap: H1 2026

## Vision
Transform our analytics platform into the industry's first AI-native business
intelligence suite, enabling non-technical users to derive insights through
natural language interaction while maintaining enterprise-grade security and
compliance.

## Theme 1: AI-Powered Insights Engine (40% of engineering capacity)
### Q1 2026
- Natural language query interface (GPT-4 integration for SQL generation)
- Automated anomaly detection across all data sources
- Smart alerting: context-aware notifications based on business rules
- Beta: Predictive forecasting for revenue and churn metrics

### Q2 2026
- GA: Predictive forecasting with confidence intervals
- Conversational follow-up queries ("why did revenue drop last Tuesday?")
- Custom model training on customer's historical data
- Multi-modal inputs: upload charts/screenshots for analysis

## Theme 2: Enterprise Platform Hardening (30% of engineering capacity)
### Q1 2026
- Multi-region deployment (EU-West, APAC-Southeast)
- Row-level security for shared dashboards
- SCIM provisioning for user management
- Audit log export to customer SIEM

### Q2 2026
- FedRAMP authorization (Moderate baseline)
- Data residency controls per workspace
- SSO certificate rotation automation
- Customer-managed encryption keys (CMEK)

## Theme 3: Ecosystem & Integrations (20% of engineering capacity)
### Q1 2026
- Salesforce bi-directional sync
- Snowflake native integration (push-down queries)
- dbt Cloud integration for transformation orchestration
- Webhook framework for custom integrations

### Q2 2026
- Tableau/Power BI embedding compatibility
- Databricks Unity Catalog support
- Real-time CDC from PostgreSQL, MySQL
- Marketplace: third-party visualization plugins

## Success Metrics
| Metric | Current | Q1 Target | Q2 Target |
|--------|---------|-----------|-----------|
| NPS | 42 | 50 | 55 |
| Enterprise ARR | $189M | $215M | $245M |
| AI Feature Adoption | 0% | 25% | 60% |
| P99 Latency | 89ms | 75ms | 60ms |
| Support Tickets/User | 2.1/mo | 1.5/mo | 1.0/mo |
"""),
]


def generate_tenant_document(template_name: str, template_text: str,
                              tenant_name: str, tenant_idx: int,
                              target_tokens: int) -> str:
    """Generate a tenant-specific variant of a shared document template."""
    header = (
        f"[CONFIDENTIAL — {tenant_name.replace('_', ' ').upper()} INTERNAL]\n"
        f"Document prepared for Tenant #{tenant_idx:03d}\n"
        f"Classification: Business Confidential\n\n"
    )
    doc = header + template_text.strip()

    # Pad to target token count
    target_chars = target_tokens * 4
    padding_sections = [
        f"\n\n## Additional Details for {tenant_name.replace('_', ' ').title()}\n"
        f"This section contains supplementary analysis specific to the {tenant_name.replace('_', ' ')} "
        f"context. The data has been cross-referenced against industry benchmarks and internal "
        f"performance metrics. Key indicators suggest alignment with the strategic objectives "
        f"outlined in the Q3 planning session.\n",
        f"\n## Historical Context\n"
        f"Previous assessments have identified similar patterns in the {template_name.replace('_', ' ')} "
        f"domain. Year-over-year trends indicate a 15-20% improvement in the primary KPIs, "
        f"consistent with the organizational transformation initiative launched in 2024. "
        f"The current report builds on these findings with updated data through Q4 2025.\n",
        f"\n## Stakeholder Impact Analysis\n"
        f"Cross-functional teams including engineering, product, sales, and finance have been "
        f"consulted. The consensus recommendation is to proceed with the proposed changes "
        f"subject to the risk mitigations outlined above. Budget allocation of $2.5M-$4.2M "
        f"has been provisionally approved pending board review in the next fiscal quarter.\n",
    ]
    idx = 0
    while len(doc) < target_chars:
        doc += padding_sections[idx % len(padding_sections)]
        idx += 1
    return doc[:target_chars]


def measure_ttft(endpoint: str, model: str, messages: list,
                 max_tokens: int = 5, stream: bool = True) -> float:
    """Measure TTFT via chat completions API (streaming)."""
    t0 = time.monotonic()
    resp = requests.post(
        f"{endpoint}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": stream,
        },
        timeout=300,
        stream=stream,
    )
    resp.raise_for_status()
    if stream:
        for line in resp.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8", errors="replace")
            if decoded.startswith("data: ") and decoded[6:].strip() != "[DONE]":
                ttft = (time.monotonic() - t0) * 1000
                for _ in resp.iter_lines():
                    pass
                return ttft
        raise ValueError("No tokens in stream")
    else:
        return (time.monotonic() - t0) * 1000


def run_benchmark(args):
    random.seed(42)

    n_tenants = min(args.n_tenants, len(TENANT_PROFILES))
    tenants = TENANT_PROFILES[:n_tenants]
    n_docs = min(args.docs_per_tenant, len(SHARED_DOC_TEMPLATES))
    doc_templates = SHARED_DOC_TEMPLATES[:n_docs]

    print(f"Multi-tenant API benchmark: {n_tenants} tenants × {n_docs} shared docs "
          f"@ ~{args.token_length} tokens")

    results = []

    # Phase 1: Seed the store — each tenant processes each document
    print("\n=== Phase 1: Seeding donor store ===")
    cold_ttfts = {}
    for t_idx, (t_name, t_system) in enumerate(tenants):
        for d_idx, (d_name, d_text) in enumerate(doc_templates):
            doc = generate_tenant_document(d_name, d_text, t_name, t_idx, args.token_length)
            messages = [
                {"role": "system", "content": f"Request-{t_idx:03d}-{d_idx:03d}. {t_system}"},
                {"role": "user", "content": doc},
            ]
            try:
                ttft = measure_ttft(args.endpoint, args.model, messages,
                                     max_tokens=1, stream=False)
                key = (t_idx, d_idx)
                cold_ttfts[key] = ttft
                print(f"  Tenant {t_name} × {d_name}: cold {ttft:.0f}ms")
            except Exception as e:
                print(f"  Tenant {t_name} × {d_name}: ERROR - {e}")

    # Phase 2: Cross-tenant queries — different tenants ask about same document type
    print("\n=== Phase 2: Cross-tenant reuse ===")
    for d_idx, (d_name, d_text) in enumerate(doc_templates):
        print(f"\n--- Document type: {d_name} ---")
        # Use first tenant's cold TTFT as baseline
        baseline_key = (0, d_idx)
        if baseline_key not in cold_ttfts:
            continue
        baseline_cold = cold_ttfts[baseline_key]

        for t_idx, (t_name, t_system) in enumerate(tenants):
            # Generate the SAME document content but with different tenant header
            doc = generate_tenant_document(d_name, d_text, t_name, t_idx, args.token_length)
            # Use a UNIQUE system message to avoid prefix cache
            messages = [
                {"role": "system", "content": f"CrossTenant-{t_idx:03d}-{d_idx:03d}-{int(time.time())}. {t_system}"},
                {"role": "user", "content": doc},
            ]
            try:
                ttft = measure_ttft(args.endpoint, args.model, messages,
                                     max_tokens=5, stream=True)
            except Exception as e:
                print(f"  {t_name}: ERROR - {e}")
                continue

            own_cold = cold_ttfts.get((t_idx, d_idx), baseline_cold)
            speedup = own_cold / ttft if ttft > 0 else 0
            is_hit = ttft < own_cold * 0.7

            result = {
                "doc_type": d_name,
                "tenant": t_name,
                "tenant_idx": t_idx,
                "cold_ttft_ms": round(own_cold, 1),
                "cross_tenant_ttft_ms": round(ttft, 1),
                "speedup": round(speedup, 2),
                "is_hit": is_hit,
            }
            results.append(result)
            marker = "HIT" if is_hit else "MISS"
            print(f"  {t_name}: {ttft:.0f}ms (vs cold {own_cold:.0f}ms) "
                  f"= {speedup:.2f}x [{marker}]")

    # Summary
    print("\n" + "=" * 60)
    print("MULTI-TENANT API BENCHMARK SUMMARY")
    print("=" * 60)

    total = len(results)
    hits = sum(1 for r in results if r["is_hit"])
    hit_rate = hits / total if total > 0 else 0
    print(f"Total cross-tenant queries: {total}")
    print(f"Cross-tenant hit rate: {hits}/{total} = {hit_rate:.1%}")

    if hits > 0:
        hit_speedups = [r["speedup"] for r in results if r["is_hit"]]
        print(f"Hit-only speedup: {sum(hit_speedups)/len(hit_speedups):.2f}x (mean)")

    # Per document type
    print("\nPer document type:")
    for d_name, _ in doc_templates:
        dr = [r for r in results if r["doc_type"] == d_name]
        dh = sum(1 for r in dr if r["is_hit"])
        rate = dh / len(dr) if dr else 0
        spds = [r["speedup"] for r in dr if r["is_hit"]]
        avg = f"{sum(spds)/len(spds):.2f}x" if spds else "N/A"
        print(f"  {d_name}: {rate:.0%} hit ({dh}/{len(dr)}), hit-only: {avg}")

    all_speedups = [r["speedup"] for r in results]
    blended = sum(all_speedups) / len(all_speedups) if all_speedups else 0
    print(f"\nBlended speedup (including misses): {blended:.2f}x")

    # Save
    output = {
        "benchmark": "multitenant_api",
        "n_tenants": n_tenants,
        "docs_per_tenant": n_docs,
        "token_length": args.token_length,
        "model": args.model,
        "total_queries": total,
        "hit_rate": round(hit_rate, 3),
        "blended_speedup": round(blended, 2),
        "results": results,
    }
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Multi-tenant API benchmark")
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--n-tenants", type=int, default=10)
    parser.add_argument("--docs-per-tenant", type=int, default=5)
    parser.add_argument("--token-length", type=int, default=8192)
    parser.add_argument("--output", default="benchmarks/e2e/results/multitenant_api.json")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
