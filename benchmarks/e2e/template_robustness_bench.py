#!/usr/bin/env python3
"""Template robustness benchmark: variable-length field insertion.

Tests SemBlend with prompts that share a large template but have different
variable-length fields inserted. Common in enterprise:
  - Email templates with different recipient names/content
  - Report templates with different data sections
  - Form-based prompts with variable fields

Key question: Does SemBlend correctly reuse KV for the shared template
portions while handling the variable fields?

Usage:
    python benchmarks/e2e/template_robustness_bench.py \
        --endpoint http://localhost:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --n-variants 20 \
        --token-length 8192
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import requests

SYSTEM_PROMPT = (
    "You are an automated report generation system. Given the following template "
    "with populated data fields, generate a professional executive summary. "
    "Focus on key metrics, trends, and actionable recommendations."
)

# Large template with insertion points marked by {field_N}
REPORT_TEMPLATE = """
# Monthly Operations Report — {company_name}
# Report Period: {report_period}
# Generated: {gen_date}
# Classification: Internal — Management Review

## 1. Executive Dashboard

### 1.1 Key Performance Indicators
| KPI | Target | Actual | Variance | Trend |
|-----|--------|--------|----------|-------|
| Revenue | {rev_target} | {rev_actual} | {rev_var} | {rev_trend} |
| Gross Margin | {gm_target} | {gm_actual} | {gm_var} | {gm_trend} |
| Customer Satisfaction | {csat_target} | {csat_actual} | {csat_var} | {csat_trend} |
| Employee NPS | {enps_target} | {enps_actual} | {enps_var} | {enps_trend} |
| System Uptime | {uptime_target} | {uptime_actual} | {uptime_var} | {uptime_trend} |

### 1.2 Financial Summary
Total revenue for {report_period} was {rev_actual}, representing a {rev_yoy}%
year-over-year change. Operating expenses totaled {opex}, with the largest
contributors being personnel costs ({personnel_pct}%), infrastructure ({infra_pct}%),
and sales & marketing ({sm_pct}%). EBITDA margin reached {ebitda_margin}%.

Cash position: {cash_position}. Runway at current burn rate: {runway} months.
Accounts receivable: {ar_total} ({ar_aging} average aging). Accounts payable:
{ap_total} ({ap_aging} average aging).

### 1.3 Headcount
Total FTEs: {total_fte} (change: {fte_change} from prior period)
- Engineering: {eng_fte} ({eng_pct}%)
- Sales: {sales_fte} ({sales_pct}%)
- Operations: {ops_fte} ({ops_pct}%)
- G&A: {ga_fte} ({ga_pct}%)

Open requisitions: {open_reqs}. Average time-to-fill: {ttf} days.
Voluntary attrition rate: {attrition}% (trailing 12-month).

## 2. Product & Engineering

### 2.1 Release Summary
{release_summary}

### 2.2 Technical Metrics
- Deployment frequency: {deploy_freq} per {deploy_period}
- Lead time for changes: {lead_time}
- Mean time to recovery: {mttr}
- Change failure rate: {cfr}%

### 2.3 Infrastructure
- Cloud spend: {cloud_spend} ({cloud_change}% vs. prior period)
- Compute utilization: {compute_util}%
- Storage: {storage_used} / {storage_total}
- CDN bandwidth: {cdn_bw}
- Database query P99: {db_p99}

## 3. Sales & Marketing

### 3.1 Pipeline
{pipeline_summary}

### 3.2 Marketing Metrics
- MQLs generated: {mqls}
- SQL conversion rate: {sql_rate}%
- CAC: {cac}
- LTV/CAC ratio: {ltv_cac}
- Content engagement: {content_eng} sessions

## 4. Customer Success

### 4.1 Health Scores
{health_summary}

### 4.2 Support Metrics
- Total tickets: {total_tickets}
- Resolution rate: {resolution_rate}%
- Average resolution time: {avg_resolution}
- CSAT score: {support_csat}/5
- Escalation rate: {escalation_rate}%

## 5. Risk Register

### Active Risks
{risk_register}

### Mitigation Status
{mitigation_status}

## 6. Strategic Initiatives

{strategic_initiatives}

## 7. Recommendations

Based on the data presented in this report, the following actions are recommended:

{recommendations}

---
Report prepared by: {preparer}
Distribution: {distribution}
Next review: {next_review}
"""

# Variable field sets — each creates a different-looking but structurally identical report
COMPANIES = [
    "Apex Digital Solutions", "Meridian Analytics Corp", "NovaTech Industries",
    "Pinnacle Systems Group", "Vanguard Data Services", "Eclipse Software Inc",
    "Quantum Leap Technologies", "Horizon Cloud Platforms", "Atlas Computing Group",
    "Zenith AI Solutions", "Frontier Digital Labs", "Catalyst Innovation Corp",
    "Summit Enterprise Tech", "Velocity Cloud Systems", "Nexus Data Corp",
    "Prism Analytics Inc", "Forge Software Group", "Beacon Technology Partners",
    "Vertex Cloud Solutions", "Synergy Digital Systems",
]


def generate_variant(idx: int, target_tokens: int) -> dict:
    """Generate a set of variable fields for the template."""
    random.seed(42 + idx)
    company = COMPANIES[idx % len(COMPANIES)]
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    month = months[idx % 12]
    year = 2025

    rev = random.uniform(3, 80)
    gm = random.uniform(55, 85)
    fte = random.randint(50, 800)

    fields = {
        "company_name": company,
        "report_period": f"{month} {year}",
        "gen_date": f"December {random.randint(1,28)}, 2025",
        "rev_target": f"${rev:.1f}M", "rev_actual": f"${rev * random.uniform(0.9,1.15):.1f}M",
        "rev_var": f"{random.uniform(-10,15):.1f}%", "rev_trend": random.choice(["↑", "↓", "→"]),
        "gm_target": f"{gm:.0f}%", "gm_actual": f"{gm + random.uniform(-3,3):.1f}%",
        "gm_var": f"{random.uniform(-3,3):.1f}pp", "gm_trend": random.choice(["↑", "↓", "→"]),
        "csat_target": "4.5/5", "csat_actual": f"{random.uniform(3.8,4.8):.1f}/5",
        "csat_var": f"{random.uniform(-0.3,0.3):.1f}", "csat_trend": random.choice(["↑", "→"]),
        "enps_target": "40", "enps_actual": str(random.randint(25, 55)),
        "enps_var": f"{random.randint(-10,10)}", "enps_trend": random.choice(["↑", "↓", "→"]),
        "uptime_target": "99.95%", "uptime_actual": f"{random.uniform(99.9,99.99):.2f}%",
        "uptime_var": f"{random.uniform(-0.05,0.04):.2f}pp", "uptime_trend": "→",
        "rev_yoy": f"{random.uniform(-5,35):.1f}",
        "opex": f"${rev * random.uniform(0.6,0.9):.1f}M",
        "personnel_pct": str(random.randint(45, 65)),
        "infra_pct": str(random.randint(15, 30)),
        "sm_pct": str(random.randint(10, 25)),
        "ebitda_margin": f"{random.uniform(5,30):.1f}%",
        "cash_position": f"${random.uniform(5,200):.1f}M",
        "runway": str(random.randint(12, 48)),
        "ar_total": f"${random.uniform(1,20):.1f}M", "ar_aging": f"{random.randint(25,55)} days",
        "ap_total": f"${random.uniform(0.5,10):.1f}M", "ap_aging": f"{random.randint(20,45)} days",
        "total_fte": str(fte), "fte_change": f"+{random.randint(0,20)}",
        "eng_fte": str(int(fte * 0.4)), "eng_pct": "40",
        "sales_fte": str(int(fte * 0.25)), "sales_pct": "25",
        "ops_fte": str(int(fte * 0.2)), "ops_pct": "20",
        "ga_fte": str(int(fte * 0.15)), "ga_pct": "15",
        "open_reqs": str(random.randint(5, 50)),
        "ttf": str(random.randint(25, 65)),
        "attrition": f"{random.uniform(8,20):.1f}",
        "release_summary": (
            f"- v{random.randint(3,8)}.{random.randint(0,9)}.{random.randint(0,9)}: "
            f"{random.choice(['Performance optimization', 'New dashboard features', 'API v2 endpoints', 'Security patch'])}\n"
            f"- v{random.randint(3,8)}.{random.randint(0,9)}.{random.randint(0,9)}: "
            f"{random.choice(['Bug fixes and stability', 'Mobile app update', 'Analytics engine upgrade', 'Integration improvements'])}\n"
            f"- Planned: {random.choice(['AI features beta', 'Multi-region deployment', 'Real-time streaming', 'Enterprise SSO'])}"
        ),
        "deploy_freq": str(random.randint(8, 40)), "deploy_period": "week",
        "lead_time": f"{random.randint(1, 72)} hours",
        "mttr": f"{random.randint(5, 120)} minutes",
        "cfr": f"{random.uniform(0.5, 8):.1f}",
        "cloud_spend": f"${random.uniform(20, 500):.0f}K",
        "cloud_change": f"{random.uniform(-5, 25):.1f}",
        "compute_util": str(random.randint(45, 85)),
        "storage_used": f"{random.uniform(1, 50):.1f}TB",
        "storage_total": f"{random.uniform(50, 200):.0f}TB",
        "cdn_bw": f"{random.uniform(5, 100):.1f}TB",
        "db_p99": f"{random.randint(15, 200)}ms",
        "pipeline_summary": (
            f"- Total pipeline: ${random.uniform(5, 100):.1f}M\n"
            f"- Qualified opportunities: {random.randint(20, 200)}\n"
            f"- Average deal size: ${random.randint(20, 500)}K\n"
            f"- Win rate: {random.randint(20, 50)}%\n"
            f"- Sales cycle: {random.randint(30, 120)} days average"
        ),
        "mqls": str(random.randint(200, 5000)),
        "sql_rate": str(random.randint(15, 40)),
        "cac": f"${random.randint(500, 5000)}",
        "ltv_cac": f"{random.uniform(2, 8):.1f}x",
        "content_eng": f"{random.randint(5000, 100000):,}",
        "health_summary": (
            f"- Green (healthy): {random.randint(60, 80)}%\n"
            f"- Yellow (at risk): {random.randint(10, 25)}%\n"
            f"- Red (critical): {random.randint(2, 10)}%\n"
            f"- NRR: {random.randint(105, 130)}%"
        ),
        "total_tickets": str(random.randint(200, 3000)),
        "resolution_rate": str(random.randint(85, 98)),
        "avg_resolution": f"{random.randint(2, 48)} hours",
        "support_csat": f"{random.uniform(3.5, 4.8):.1f}",
        "escalation_rate": f"{random.uniform(2, 12):.1f}",
        "risk_register": (
            f"1. [{random.choice(['HIGH', 'MEDIUM'])}] {random.choice(['Key person dependency', 'Competitor threat', 'Regulatory change', 'Supply chain disruption'])}\n"
            f"2. [{random.choice(['HIGH', 'MEDIUM'])}] {random.choice(['Technical debt', 'Data breach risk', 'Vendor lock-in', 'Market downturn'])}\n"
            f"3. [{random.choice(['MEDIUM', 'LOW'])}] {random.choice(['Talent retention', 'Currency exposure', 'IP litigation', 'Integration complexity'])}"
        ),
        "mitigation_status": (
            f"- Risk 1: {random.choice(['In progress — 60% complete', 'Planned for Q1', 'Monitoring', 'Mitigated'])}\n"
            f"- Risk 2: {random.choice(['Assessment phase', 'Budget approved', 'Vendor evaluation', 'Deferred to Q2'])}\n"
            f"- Risk 3: {random.choice(['Accepted', 'Under review', 'Insurance obtained', 'Policy updated'])}"
        ),
        "strategic_initiatives": (
            f"### Initiative A: {random.choice(['Digital Transformation', 'Market Expansion', 'Product Innovation', 'Operational Excellence'])}\n"
            f"Status: {random.choice(['On track', 'Behind schedule', 'Ahead of plan'])}\n"
            f"Budget utilization: {random.randint(40, 90)}%\n"
            f"Key milestone: {random.choice(['Platform launch', 'Beta release', 'Pilot completion', 'Full rollout'])} — "
            f"{random.choice(['Completed', 'On schedule for Q1', 'Delayed 2 weeks', 'In review'])}\n\n"
            f"### Initiative B: {random.choice(['Customer Success 2.0', 'AI Integration', 'Global Expansion', 'Cost Optimization'])}\n"
            f"Status: {random.choice(['On track', 'At risk', 'Completed'])}\n"
            f"Budget utilization: {random.randint(30, 85)}%\n"
        ),
        "recommendations": (
            f"1. {random.choice(['Increase engineering investment in AI features', 'Accelerate multi-region deployment', 'Implement automated testing pipeline', 'Expand sales team in EMEA'])}\n"
            f"2. {random.choice(['Review vendor contracts for cost optimization', 'Launch customer advisory board', 'Hire VP of Data Science', 'Implement OKR framework'])}\n"
            f"3. {random.choice(['Address technical debt in core platform', 'Develop partnership program', 'Upgrade security infrastructure', 'Launch employee development program'])}\n"
        ),
        "preparer": random.choice(["Chief of Staff", "VP Operations", "Director of Strategy", "COO Office"]),
        "distribution": "Executive Team, Board of Directors, Department Heads",
        "next_review": f"{random.choice(['January', 'February', 'March'])} {year + 1}",
    }
    return fields


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
    n = args.n_variants

    print(f"Template robustness benchmark: {n} variants @ ~{args.token_length} tokens")

    results = []
    cold_ttft = None

    for i in range(n):
        fields = generate_variant(i, args.token_length)
        report = REPORT_TEMPLATE.format(**fields)

        # Pad to target length
        target_chars = args.token_length * 4
        while len(report) < target_chars:
            report += (
                f"\n## Appendix — Supplementary Data ({fields['company_name']})\n"
                f"Additional operational metrics, financial details, and trend analysis "
                f"for the period {fields['report_period']}. This data supports the "
                f"findings presented in the main report sections above.\n"
            )
        report = report[:target_chars]

        # Unique system message to avoid prefix cache
        sys_msg = f"Report-{i:04d}-{int(time.time())}. {SYSTEM_PROMPT}"
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": report},
        ]

        try:
            ttft = measure_ttft(args.endpoint, args.model, messages,
                                 max_tokens=5, stream=True)
        except Exception as e:
            print(f"  Variant {i+1} ({fields['company_name']}): ERROR - {e}")
            continue

        if cold_ttft is None:
            cold_ttft = ttft
            print(f"  Variant {i+1} ({fields['company_name']}): {ttft:.0f}ms [FIRST/COLD]")
            results.append({
                "variant_idx": i,
                "company": fields["company_name"],
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
            print(f"  Variant {i+1} ({fields['company_name']}): {ttft:.0f}ms "
                  f"({speedup:.2f}x vs cold {cold_ttft:.0f}ms) [{marker}]")
            results.append({
                "variant_idx": i,
                "company": fields["company_name"],
                "ttft_ms": round(ttft, 1),
                "cold_ttft_ms": round(cold_ttft, 1),
                "speedup": round(speedup, 2),
                "is_hit": is_hit,
                "is_cold": False,
            })

    # Summary
    print("\n" + "=" * 60)
    print("TEMPLATE ROBUSTNESS BENCHMARK SUMMARY")
    print("=" * 60)

    non_cold = [r for r in results if not r.get("is_cold")]
    total = len(non_cold)
    hits = sum(1 for r in non_cold if r["is_hit"])
    hit_rate = hits / total if total > 0 else 0
    print(f"Total non-cold queries: {total}")
    print(f"Hit rate: {hits}/{total} = {hit_rate:.1%}")

    if hits > 0:
        hit_speedups = [r["speedup"] for r in non_cold if r["is_hit"]]
        print(f"Hit-only speedup: {sum(hit_speedups)/len(hit_speedups):.2f}x (mean)")
        print(f"  Min: {min(hit_speedups):.2f}x")
        print(f"  Max: {max(hit_speedups):.2f}x")
        sorted_spds = sorted(hit_speedups)
        print(f"  P50: {sorted_spds[len(sorted_spds)//2]:.2f}x")

    all_speedups = [r["speedup"] for r in non_cold]
    blended = sum(all_speedups) / len(all_speedups) if all_speedups else 0
    print(f"\nBlended speedup: {blended:.2f}x")

    output = {
        "benchmark": "template_robustness",
        "n_variants": n,
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
    parser = argparse.ArgumentParser(description="Template robustness benchmark")
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--n-variants", type=int, default=20)
    parser.add_argument("--token-length", type=int, default=8192)
    parser.add_argument("--output", default="benchmarks/e2e/results/template_robustness.json")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
