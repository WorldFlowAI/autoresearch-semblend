#!/usr/bin/env python3
"""Customer support ticket benchmark: sequential similar tickets.

Simulates a customer support system where agents process sequential tickets
about similar issues. SemBlend should discover that ticket N+1 about
"password reset" is similar to ticket N about the same topic, reusing KV.

Uses real-world support ticket patterns:
  - Same issue type, different customer details
  - Similar product complaints with varying specifics
  - Sequential escalation threads

Usage:
    python benchmarks/e2e/customer_support_bench.py \
        --endpoint http://localhost:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --tickets-per-category 10 \
        --token-length 4096
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import requests

SUPPORT_SYSTEM_PROMPT = (
    "You are a Tier 2 customer support specialist for CloudScale Technologies. "
    "Analyze the following support ticket and provide: (1) Root cause assessment, "
    "(2) Recommended resolution steps, (3) Customer communication draft, "
    "(4) Escalation recommendation if needed. Follow the company's support playbook."
)

# Each category has a template with variable slots
TICKET_CATEGORIES = {
    "password_reset": {
        "subject": "Unable to reset password — {customer}",
        "template": """
Support Ticket #{ticket_id}
Priority: {priority}
Customer: {customer} ({email})
Account: Enterprise Tier — {company}
Product: CloudScale Analytics Platform v3.2
Category: Authentication / Password Reset

## Issue Description
Customer reports inability to reset their password through the standard self-service
flow. When clicking the "Reset Password" link in the email, the page shows
"{error_msg}". Customer has tried {attempts} times over the past {days} days.

## Customer Environment
- Browser: {browser}
- OS: {os}
- SSO Provider: {sso}
- Last successful login: {last_login}
- MFA enabled: {mfa}

## Previous Interactions
{prev_interactions}

## Account Notes
- Account created: {account_date}
- License tier: Enterprise ({seats} seats)
- Annual contract value: ${acv}
- Renewal date: {renewal}
- Account health score: {health}/100
- Customer satisfaction (last survey): {csat}/5

## Technical Details
- Error code: AUTH-{error_code}
- Session ID: {session_id}
- IP geolocation: {geo}
- Rate limit status: {rate_limit}
""",
        "variants": [
            {"customer": "Sarah Chen", "email": "s.chen@techcorp.io", "company": "TechCorp Industries",
             "error_msg": "Token expired. Please request a new reset link.", "attempts": "3", "days": "2",
             "browser": "Chrome 120", "os": "macOS 14.2", "sso": "Okta", "last_login": "2025-11-28",
             "mfa": "Yes (TOTP)", "account_date": "2023-06-15", "seats": "250", "acv": "180,000",
             "renewal": "2026-06-15", "health": "82", "csat": "3.8", "error_code": "4012",
             "session_id": "sess_7f3a2b1c", "geo": "San Francisco, CA", "rate_limit": "3/5 attempts used"},
            {"customer": "Marcus Johnson", "email": "m.johnson@globalfin.com", "company": "Global Finance Ltd",
             "error_msg": "Invalid token format. Contact support.", "attempts": "5", "days": "4",
             "browser": "Firefox 121", "os": "Windows 11", "sso": "Azure AD", "last_login": "2025-12-01",
             "mfa": "Yes (WebAuthn)", "account_date": "2022-01-10", "seats": "500", "acv": "340,000",
             "renewal": "2026-01-10", "health": "91", "csat": "4.2", "error_code": "4015",
             "session_id": "sess_8e4b3c2d", "geo": "London, UK", "rate_limit": "5/5 attempts used (locked)"},
            {"customer": "Priya Patel", "email": "priya@medisys.co", "company": "MediSys Healthcare",
             "error_msg": "Password does not meet complexity requirements.", "attempts": "7", "days": "3",
             "browser": "Edge 120", "os": "Windows 10", "sso": "OneLogin", "last_login": "2025-11-15",
             "mfa": "No", "account_date": "2024-03-20", "seats": "75", "acv": "52,000",
             "renewal": "2025-03-20", "health": "65", "csat": "2.9", "error_code": "4023",
             "session_id": "sess_9f5c4d3e", "geo": "Mumbai, India", "rate_limit": "2/5 attempts used"},
            {"customer": "James O'Brien", "email": "jobrien@retailmax.com", "company": "RetailMax Corp",
             "error_msg": "Account locked due to too many attempts.", "attempts": "10", "days": "7",
             "browser": "Safari 17", "os": "macOS 14.1", "sso": "Ping Identity", "last_login": "2025-10-30",
             "mfa": "Yes (SMS)", "account_date": "2023-09-01", "seats": "150", "acv": "110,000",
             "renewal": "2026-09-01", "health": "73", "csat": "3.1", "error_code": "4031",
             "session_id": "sess_af6d5e4f", "geo": "Dublin, Ireland", "rate_limit": "LOCKED"},
            {"customer": "Lisa Wang", "email": "l.wang@dataflow.ai", "company": "DataFlow AI",
             "error_msg": "SSO redirect loop detected. Clear cookies and retry.", "attempts": "4", "days": "1",
             "browser": "Chrome 121", "os": "Ubuntu 22.04", "sso": "Okta", "last_login": "2025-12-10",
             "mfa": "Yes (TOTP)", "account_date": "2024-07-01", "seats": "30", "acv": "22,000",
             "renewal": "2025-07-01", "health": "88", "csat": "4.5", "error_code": "4018",
             "session_id": "sess_bg7e6f5g", "geo": "Toronto, Canada", "rate_limit": "1/5 attempts used"},
        ],
    },
    "api_error": {
        "subject": "API returning 500 errors — {customer}",
        "template": """
Support Ticket #{ticket_id}
Priority: {priority}
Customer: {customer} ({email})
Account: Enterprise Tier — {company}
Product: CloudScale Analytics API v4.1
Category: API / Server Error

## Issue Description
Customer's production integration is receiving intermittent HTTP 500 errors from the
Analytics API. The errors started approximately {hours_ago} hours ago and affect
{affected_pct}% of their API calls. Customer's application serves {end_users} end users
and this is causing {impact}.

## Error Details
- Endpoint: {endpoint}
- HTTP Status: 500 Internal Server Error
- Error body: {{"error": "{error_body}", "request_id": "{request_id}"}}
- Rate of errors: ~{error_rate} per minute
- Pattern: {pattern}

## Customer Integration
- SDK: {sdk}
- Authentication: API Key (key_id: {key_id})
- Rate limit tier: {rate_tier} ({rate_limit} req/min)
- Webhook endpoint: {webhook}
- Retry configuration: {retry_config}

## Timeline
{timeline}

## Account Context
- Integration since: {integration_date}
- Monthly API volume: {monthly_volume}
- Contract value: ${acv}/year
- Technical contact: {tech_contact}
- Last API version upgrade: {last_upgrade}
""",
        "variants": [
            {"customer": "Alex Rivera", "email": "a.rivera@ecomm.io", "company": "E-Commerce Plus",
             "hours_ago": "6", "affected_pct": "15", "end_users": "450,000",
             "impact": "checkout flow failures for 15% of users",
             "endpoint": "POST /v4/analytics/events/batch", "error_body": "Internal processing error",
             "request_id": "req_a1b2c3", "error_rate": "42", "pattern": "Correlated with batch size > 500 events",
             "sdk": "Python SDK 2.3.1", "key_id": "key_prod_7x8y9z", "rate_tier": "Enterprise",
             "rate_limit": "10,000", "webhook": "https://ecomm.io/webhooks/analytics",
             "retry_config": "3 retries, exponential backoff (1s, 2s, 4s)",
             "integration_date": "2024-01-15", "monthly_volume": "45M requests",
             "acv": "280,000", "tech_contact": "CTO", "last_upgrade": "2025-10-01"},
            {"customer": "Diana Foster", "email": "dfoster@healthtrak.com", "company": "HealthTrak Systems",
             "hours_ago": "2", "affected_pct": "100", "end_users": "12,000 clinicians",
             "impact": "complete loss of real-time patient analytics dashboard",
             "endpoint": "GET /v4/analytics/dashboards/{id}/data", "error_body": "Query execution timeout",
             "request_id": "req_d4e5f6", "error_rate": "200+", "pattern": "All requests failing, not intermittent",
             "sdk": "JavaScript SDK 3.0.0", "key_id": "key_prod_3m4n5o", "rate_tier": "Enterprise Plus",
             "rate_limit": "50,000", "webhook": "https://api.healthtrak.com/hooks/cs-events",
             "retry_config": "5 retries, 500ms fixed interval",
             "integration_date": "2023-03-01", "monthly_volume": "120M requests",
             "acv": "520,000", "tech_contact": "VP Engineering", "last_upgrade": "2025-11-15"},
            {"customer": "Robert Kim", "email": "rkim@logisticspro.net", "company": "LogisticsPro Global",
             "hours_ago": "12", "affected_pct": "5", "end_users": "8,000 warehouse operators",
             "impact": "delayed shipment tracking updates for some facilities",
             "endpoint": "PUT /v4/analytics/custom-metrics", "error_body": "Serialization error: invalid field type",
             "request_id": "req_g7h8i9", "error_rate": "8", "pattern": "Only for metrics with nested object values",
             "sdk": "Go SDK 1.8.0", "key_id": "key_prod_6p7q8r", "rate_tier": "Standard",
             "rate_limit": "5,000", "webhook": "N/A",
             "retry_config": "No retries configured",
             "integration_date": "2024-09-01", "monthly_volume": "8M requests",
             "acv": "95,000", "tech_contact": "Senior Developer", "last_upgrade": "2025-06-01"},
        ],
    },
    "billing_dispute": {
        "subject": "Billing discrepancy — {customer}",
        "template": """
Support Ticket #{ticket_id}
Priority: {priority}
Customer: {customer} ({email})
Account: {tier} Tier — {company}
Product: CloudScale Platform
Category: Billing / Invoice Dispute

## Issue Description
Customer is disputing their most recent invoice (#{invoice_id}), citing a discrepancy
of ${discrepancy} between expected and actual charges. The primary concern is
{concern}. Customer has requested {resolution_request}.

## Invoice Details
- Invoice #: {invoice_id}
- Billing period: {billing_period}
- Total charged: ${total_charged}
- Expected amount: ${expected_amount}
- Discrepancy: ${discrepancy}

## Charge Breakdown
{charge_breakdown}

## Usage Data
- API calls: {api_calls} (included: {api_included})
- Storage: {storage} (included: {storage_included})
- Compute hours: {compute} (included: {compute_included})
- Data transfer: {transfer} (included: {transfer_included})

## Account History
- Payment method: {payment_method}
- Payment history: {payment_history}
- Previous disputes: {prev_disputes}
- Account standing: {standing}
- Contract terms: {contract_terms}
""",
        "variants": [
            {"customer": "Emma Thompson", "email": "ethompson@fintechco.com", "company": "FinTech Innovations",
             "tier": "Enterprise", "invoice_id": "INV-2025-12847", "discrepancy": "4,200",
             "total_charged": "24,200", "expected_amount": "20,000",
             "concern": "unexpected overage charges for API calls during a load test",
             "resolution_request": "credit for the overage amount",
             "billing_period": "November 1-30, 2025",
             "charge_breakdown": "Base license: $15,000\nAPI overage (2.1M calls × $0.002): $4,200\nStorage: $3,500\nSupport: $1,500",
             "api_calls": "12.1M", "api_included": "10M", "storage": "3.5TB", "storage_included": "5TB",
             "compute": "450 hrs", "compute_included": "500 hrs", "transfer": "8TB", "transfer_included": "10TB",
             "payment_method": "ACH (Wells Fargo ****4521)", "payment_history": "24/24 on-time payments",
             "prev_disputes": "None", "standing": "Excellent", "contract_terms": "3-year commitment, 10M API calls/month included"},
            {"customer": "David Park", "email": "dpark@retailchain.com", "company": "National Retail Chain",
             "tier": "Enterprise Plus", "invoice_id": "INV-2025-12903", "discrepancy": "12,500",
             "total_charged": "52,500", "expected_amount": "40,000",
             "concern": "being charged for a premium feature they never activated",
             "resolution_request": "full refund of the feature charge and audit of their feature flags",
             "billing_period": "November 1-30, 2025",
             "charge_breakdown": "Base license: $30,000\nAdvanced Analytics: $12,500 (DISPUTED)\nStorage: $5,000\nSupport Premium: $5,000",
             "api_calls": "8.5M", "api_included": "20M", "storage": "12TB", "storage_included": "15TB",
             "compute": "800 hrs", "compute_included": "1000 hrs", "transfer": "20TB", "transfer_included": "25TB",
             "payment_method": "Corporate Card (Amex ****8832)", "payment_history": "18/18 on-time",
             "prev_disputes": "1 resolved in customer's favor (June 2025)", "standing": "Good",
             "contract_terms": "2-year commitment, Advanced Analytics was in trial period"},
        ],
    },
    "performance_degradation": {
        "subject": "Dashboard load times increased 5x — {customer}",
        "template": """
Support Ticket #{ticket_id}
Priority: {priority}
Customer: {customer} ({email})
Account: Enterprise Tier — {company}
Product: CloudScale Analytics Dashboard
Category: Performance / Latency

## Issue Description
Customer reports that their main analytics dashboard has experienced a {severity}
degradation in load times starting {start_time}. Dashboard queries that previously
completed in {old_latency} are now taking {new_latency}. This is affecting
{affected_users} users across {affected_teams} teams.

## Performance Metrics (customer-provided)
- Dashboard load time (before): {old_latency}
- Dashboard load time (current): {new_latency}
- Widget count on dashboard: {widget_count}
- Data source connections: {data_sources}
- Date range filter: {date_range}
- Refresh interval: {refresh}

## Environment Changes
{env_changes}

## System Health (our side)
- Region: {region}
- Cluster utilization: {cluster_util}
- Query queue depth: {queue_depth}
- Cache hit rate: {cache_rate}
- Recent deployments: {deployments}

## Customer Business Impact
{business_impact}
""",
        "variants": [
            {"customer": "Jennifer Walsh", "email": "jwalsh@mediagroup.com", "company": "National Media Group",
             "severity": "5x", "start_time": "Monday morning (Dec 9, 2025)",
             "old_latency": "2-3 seconds", "new_latency": "12-15 seconds",
             "affected_users": "150", "affected_teams": "5 (editorial, sales, finance, marketing, executive)",
             "widget_count": "24", "data_sources": "6 (PostgreSQL, Snowflake, Google Analytics, Salesforce, HubSpot, internal API)",
             "date_range": "Last 90 days", "refresh": "Every 5 minutes",
             "env_changes": "- Customer added 2 new Snowflake data sources on Dec 6\n- Upgraded SDK from v2.9 to v3.0 on Dec 8\n- No infrastructure changes on our side",
             "region": "us-east-1", "cluster_util": "67%", "queue_depth": "Normal (avg 12 queries)",
             "cache_rate": "45% (down from 78%)", "deployments": "Platform v3.2.1 deployed Dec 7 (cache invalidation fix)",
             "business_impact": "- Executive team cannot access real-time revenue dashboard for board meeting prep\n- Sales team missing hourly pipeline updates\n- Advertising team delayed on campaign performance reporting\n- Customer considering fallback to Tableau for critical dashboards"},
            {"customer": "Tom Rodriguez", "email": "trodriguez@logistics.co", "company": "Pacific Logistics",
             "severity": "3x", "start_time": "Thursday afternoon (Dec 5, 2025)",
             "old_latency": "4-5 seconds", "new_latency": "15-20 seconds",
             "affected_users": "45", "affected_teams": "2 (operations, executive)",
             "widget_count": "18", "data_sources": "3 (MySQL, internal REST API, CSV uploads)",
             "date_range": "Last 30 days", "refresh": "Every 15 minutes",
             "env_changes": "- No changes on customer side\n- Customer suspects our recent platform update",
             "region": "us-west-2", "cluster_util": "82%", "queue_depth": "Elevated (avg 45 queries)",
             "cache_rate": "62%", "deployments": "Platform v3.2.1 deployed Dec 4 (cache invalidation fix)",
             "business_impact": "- Operations team cannot track real-time shipment status\n- Fleet dispatchers reverting to manual spreadsheets\n- Customer threatening to invoke SLA penalty clause ($5,000/day for P99 > 10s)"},
        ],
    },
}


def generate_ticket(category: str, template: str, variant: dict,
                    ticket_id: int, target_tokens: int) -> str:
    """Generate a support ticket from template + variant."""
    # Fill in common fields
    filled = template.format(
        ticket_id=f"CS-2025-{ticket_id:05d}",
        priority=random.choice(["P1 - Critical", "P2 - High", "P3 - Medium"]),
        prev_interactions="- Nov 28: Initial email from customer (auto-response sent)\n"
                          "- Nov 29: Tier 1 agent attempted basic troubleshooting\n"
                          "- Dec 1: Escalated to Tier 2 due to persistence of issue\n"
                          "- Dec 2: Customer called in, frustrated with lack of resolution",
        timeline="- T+0h: Customer reports issue via support portal\n"
                 "- T+1h: Auto-acknowledged, assigned to Tier 1 queue\n"
                 "- T+3h: Tier 1 confirms reproduction of the issue\n"
                 "- T+4h: Escalated to Tier 2 engineering support\n"
                 "- T+6h: Engineering confirms server-side component involved",
        **variant,
    )

    # Pad to target length
    target_chars = target_tokens * 4
    padding = (
        f"\n\n## Internal Notes (Tier 1)\n"
        f"Attempted standard troubleshooting steps per KB article #{random.randint(1000,9999)}. "
        f"Customer has been cooperative but increasingly frustrated. Previous similar tickets "
        f"for this account type were resolved by {random.choice(['cache clear', 'token refresh', 'config reset', 'DNS flush'])}. "
        f"Recommended escalation to Tier 2 based on ticket age and customer tier.\n"
        f"\n## Similar Resolved Tickets\n"
        f"- CS-2025-{random.randint(10000,99999)}: Similar {category.replace('_', ' ')} issue "
        f"resolved in 48 hours via configuration change\n"
        f"- CS-2025-{random.randint(10000,99999)}: Related issue traced to upstream provider\n"
    )
    while len(filled) < target_chars:
        filled += padding
    return filled[:target_chars]


def measure_ttft(endpoint: str, model: str, messages: list,
                 max_tokens: int = 5, stream: bool = True) -> float:
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
    n_per_cat = args.tickets_per_category

    print(f"Customer support benchmark: {len(TICKET_CATEGORIES)} categories × "
          f"{n_per_cat} tickets @ ~{args.token_length} tokens")

    results = []
    ticket_counter = 1000

    for cat_name, cat_data in TICKET_CATEGORIES.items():
        template = cat_data["template"]
        variants = cat_data["variants"]
        print(f"\n=== Category: {cat_name} ({len(variants)} variants available) ===")

        cold_ttft = None
        for i in range(min(n_per_cat, len(variants))):
            variant = variants[i % len(variants)]
            ticket_counter += 1
            ticket_text = generate_ticket(cat_name, template, variant,
                                           ticket_counter, args.token_length)

            # Unique system message per request to avoid prefix cache
            sys_msg = f"Ticket-{ticket_counter:05d}-{int(time.time())}. {SUPPORT_SYSTEM_PROMPT}"
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": ticket_text},
            ]

            try:
                ttft = measure_ttft(args.endpoint, args.model, messages,
                                     max_tokens=5, stream=True)
            except Exception as e:
                print(f"  Ticket {i+1}: ERROR - {e}")
                continue

            if cold_ttft is None:
                cold_ttft = ttft
                print(f"  Ticket {i+1} ({variant['customer']}): {ttft:.0f}ms [FIRST/COLD]")
                results.append({
                    "category": cat_name,
                    "ticket_idx": i,
                    "customer": variant["customer"],
                    "ttft_ms": round(ttft, 1),
                    "cold_ttft_ms": round(ttft, 1),
                    "speedup": 1.0,
                    "is_hit": False,
                    "is_cold": True,
                })
            else:
                speedup = cold_ttft / ttft if ttft > 0 else 0
                is_hit = ttft < cold_ttft * 0.7
                marker = "HIT" if is_hit else "MISS"
                print(f"  Ticket {i+1} ({variant['customer']}): {ttft:.0f}ms "
                      f"({speedup:.2f}x vs cold {cold_ttft:.0f}ms) [{marker}]")
                results.append({
                    "category": cat_name,
                    "ticket_idx": i,
                    "customer": variant["customer"],
                    "ttft_ms": round(ttft, 1),
                    "cold_ttft_ms": round(cold_ttft, 1),
                    "speedup": round(speedup, 2),
                    "is_hit": is_hit,
                    "is_cold": False,
                })

    # Summary
    print("\n" + "=" * 60)
    print("CUSTOMER SUPPORT BENCHMARK SUMMARY")
    print("=" * 60)

    non_cold = [r for r in results if not r.get("is_cold")]
    total = len(non_cold)
    hits = sum(1 for r in non_cold if r["is_hit"])
    hit_rate = hits / total if total > 0 else 0
    print(f"Total non-cold queries: {total}")
    print(f"Sequential hit rate: {hits}/{total} = {hit_rate:.1%}")

    if hits > 0:
        hit_speedups = [r["speedup"] for r in non_cold if r["is_hit"]]
        print(f"Hit-only speedup: {sum(hit_speedups)/len(hit_speedups):.2f}x (mean)")

    print("\nPer category:")
    for cat_name in TICKET_CATEGORIES:
        cr = [r for r in non_cold if r["category"] == cat_name]
        ch = sum(1 for r in cr if r["is_hit"])
        rate = ch / len(cr) if cr else 0
        spds = [r["speedup"] for r in cr if r["is_hit"]]
        avg = f"{sum(spds)/len(spds):.2f}x" if spds else "N/A"
        print(f"  {cat_name}: {rate:.0%} hit ({ch}/{len(cr)}), hit-only: {avg}")

    all_speedups = [r["speedup"] for r in non_cold]
    blended = sum(all_speedups) / len(all_speedups) if all_speedups else 0
    print(f"\nBlended speedup: {blended:.2f}x")

    output = {
        "benchmark": "customer_support",
        "categories": len(TICKET_CATEGORIES),
        "tickets_per_category": n_per_cat,
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
    parser = argparse.ArgumentParser(description="Customer support benchmark")
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--tickets-per-category", type=int, default=5)
    parser.add_argument("--token-length", type=int, default=4096)
    parser.add_argument("--output", default="benchmarks/e2e/results/customer_support.json")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
