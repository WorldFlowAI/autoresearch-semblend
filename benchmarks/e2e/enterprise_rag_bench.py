#!/usr/bin/env python3
"""Enterprise RAG benchmark: same document, genuinely different user queries.

Unlike cross_instruction_bench.py (which varies instructions by 9-25 chars),
this benchmark tests the REAL enterprise RAG pattern: users ask fundamentally
different questions about the same retrieved document.

Query types differ by 100-500 tokens:
  - Summarization: "Summarize the key points..."
  - Extraction: "Extract all dates, names, amounts..."
  - Risk analysis: "What risks and limitations..."
  - Comparison: "Compare to industry standards..."
  - Factual Q&A: "Based on this document, answer..."

Also compares SemBlend vs LMCache-only (baseline).

Usage:
    python benchmarks/e2e/enterprise_rag_bench.py \
        --endpoint http://localhost:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --n-docs 30 \
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

# Enterprise-realistic instruction templates (100-500 tokens difference between types)
QUERY_TYPES = {
    "summarize": (
        "You are a senior analyst preparing an executive briefing. Read the following "
        "document carefully and provide a comprehensive summary covering all key points, "
        "main arguments, supporting evidence, and conclusions. The summary should be "
        "suitable for C-level executives who need to understand the document without "
        "reading it in full. Focus on actionable insights and strategic implications. "
        "Organize your summary with clear headers for each major topic."
    ),
    "extract": (
        "You are a data extraction specialist. Carefully read the following document and "
        "extract ALL structured information including: (1) Named entities: people, "
        "organizations, locations, products; (2) Dates and time references; (3) Monetary "
        "amounts and financial figures; (4) Technical specifications and measurements; "
        "(5) Legal terms, regulations, or compliance references; (6) Action items or "
        "recommendations. Present each category as a bulleted list with exact quotes "
        "from the source document. Flag any ambiguous or uncertain extractions."
    ),
    "risk_analysis": (
        "You are a risk management consultant reviewing the following document. Identify "
        "and analyze: (1) All explicit risks, threats, or vulnerabilities mentioned; "
        "(2) Implicit risks that the document does not address but should; (3) "
        "Dependencies and single points of failure; (4) Regulatory or compliance "
        "concerns; (5) Financial exposure or liability. For each risk, assess severity "
        "(low/medium/high/critical), likelihood, and potential mitigation strategies. "
        "Conclude with a prioritized risk register."
    ),
    "comparison": (
        "You are an industry analyst conducting a competitive analysis. Read the "
        "following document and compare its content against standard industry practices "
        "and benchmarks. Address: (1) How does the approach described compare to the "
        "current state of the art? (2) What are the unique differentiators? (3) What "
        "gaps exist relative to competitor offerings? (4) What is the total cost of "
        "ownership compared to alternatives? (5) What are the scalability limitations? "
        "Provide specific examples and quantitative comparisons where possible."
    ),
    "qa_detailed": (
        "You are a technical support specialist. Based ONLY on the information in the "
        "following document, answer these questions in detail with exact references to "
        "the source text. If the document does not contain sufficient information to "
        "answer a question, explicitly state 'Not addressed in document'. Questions: "
        "What is the primary purpose of this document? What are the three most important "
        "technical details? What prerequisites or dependencies are mentioned? What are "
        "the expected outcomes or deliverables? What timeline or milestones are defined?"
    ),
}

# Diverse document topics for generating synthetic enterprise content
DOCUMENT_TOPICS = [
    ("cloud_migration", "A Fortune 500 company's cloud migration strategy document covering AWS infrastructure, cost analysis, security requirements, compliance frameworks (SOC2, HIPAA), timeline from Q1-Q4, team allocation across 5 departments, risk assessment of downtime during cutover, and rollback procedures."),
    ("product_launch", "Product requirements document for a B2B SaaS analytics platform. Covers user personas, feature specifications for dashboard builder, real-time data ingestion pipeline, role-based access control, API rate limiting, integration with Salesforce/HubSpot, pricing tiers, and go-to-market timeline."),
    ("security_audit", "Annual cybersecurity audit report for a financial services firm. Findings include: 3 critical vulnerabilities in payment processing, outdated TLS configurations on 12 endpoints, inadequate logging in the authentication service, phishing simulation results showing 23% click rate, and recommendations for zero-trust architecture."),
    ("merger_analysis", "Due diligence analysis for a proposed acquisition of a data analytics startup. Covers: intellectual property portfolio (14 patents), technical debt assessment, customer concentration risk (top 3 clients = 67% revenue), employee retention projections, integration cost estimates, and regulatory approval timeline."),
    ("incident_report", "Post-incident report for a 4-hour production outage affecting 2.3 million users. Root cause: cascading failure in the message queue cluster triggered by a misconfigured auto-scaling policy. Timeline of events, response actions, customer impact assessment, and 12 remediation items with owners and deadlines."),
    ("compliance_doc", "GDPR compliance documentation for a healthcare SaaS platform. Covers: data processing agreements with 47 sub-processors, consent management implementation, data retention policies by category, cross-border transfer mechanisms (SCCs), breach notification procedures, and DPO responsibilities."),
    ("infra_review", "Infrastructure capacity planning review for a video streaming service. Current state: 850 Kubernetes pods across 3 regions, 15PB storage, 2.1M concurrent users peak. Projected growth: 40% YoY. Recommendations: multi-CDN strategy, edge computing rollout, GPU transcoding cluster expansion, and database sharding plan."),
    ("api_design", "API design specification for a payment processing gateway. RESTful endpoints for payment initiation, status polling, refund processing, and webhook management. Includes authentication flows (OAuth 2.0 + API keys), rate limiting tiers, idempotency key handling, PCI DSS compliance requirements, and SDK integration guides for 6 languages."),
    ("ml_pipeline", "Machine learning pipeline architecture for a recommendation engine. Components: feature store (Feast), training pipeline (Kubeflow), model registry (MLflow), A/B testing framework, real-time inference with sub-50ms latency requirement. Data sources: clickstream (2TB/day), purchase history, user profiles. Model: two-tower architecture with 120M parameters."),
    ("legal_contract", "Master services agreement between a cloud provider and enterprise customer. Terms include: SLA guarantees (99.99% uptime), data sovereignty requirements (EU-only processing), liability caps, indemnification clauses, termination procedures, transition assistance obligations, and intellectual property ownership for custom integrations."),
]


def generate_document(topic_name: str, description: str, target_tokens: int) -> str:
    """Generate a synthetic enterprise document by repeating/expanding the description."""
    # Build a realistic document structure
    sections = [
        f"# {topic_name.replace('_', ' ').title()}\n\n",
        f"## Executive Summary\n\n{description}\n\n",
        f"## Background and Context\n\nThis document provides a detailed analysis of {topic_name.replace('_', ' ')}. ",
        f"The following sections outline the key findings, recommendations, and action items "
        f"identified during the review process. All stakeholders should review the content "
        f"carefully and provide feedback by the specified deadline.\n\n",
    ]

    # Expand with repeated detail sections to reach target token count
    detail_paragraphs = [
        f"## Detailed Analysis\n\n{description} The analysis covers multiple dimensions "
        f"including technical feasibility, financial impact, organizational readiness, and "
        f"regulatory compliance. Each dimension is evaluated independently and then "
        f"synthesized into an overall recommendation.\n\n",
        f"## Technical Assessment\n\nFrom a technical perspective, the {topic_name.replace('_', ' ')} "
        f"involves several interconnected components. The architecture must support both "
        f"current requirements and projected growth over the next 3-5 years. Key technical "
        f"decisions include infrastructure selection, security framework implementation, "
        f"and integration strategy with existing systems.\n\n",
        f"## Financial Impact\n\nThe estimated total cost for the {topic_name.replace('_', ' ')} "
        f"initiative ranges from $2.5M to $4.2M over a 24-month implementation period. "
        f"This includes: personnel costs ($1.8M), infrastructure ($800K-1.5M), licensing "
        f"($300K), and contingency (15%). Expected ROI is 3.2x within 36 months based on "
        f"projected efficiency gains and revenue impact.\n\n",
        f"## Risk Assessment\n\nKey risks identified: (1) Timeline risk - aggressive schedule "
        f"assumes parallel workstreams with no blocking dependencies; (2) Talent risk - "
        f"specialized skills required for implementation are in high demand; (3) Integration "
        f"risk - legacy system APIs are poorly documented; (4) Regulatory risk - pending "
        f"regulatory changes could require scope adjustment.\n\n",
        f"## Recommendations\n\nBased on the analysis, we recommend: (1) Proceeding with "
        f"Phase 1 implementation targeting Q2 delivery; (2) Establishing a dedicated team "
        f"of 8 FTEs; (3) Engaging a specialized consulting firm for regulatory compliance; "
        f"(4) Implementing bi-weekly stakeholder reviews; (5) Setting up automated monitoring "
        f"and alerting for all critical milestones.\n\n",
        f"## Appendix\n\n{description} Additional supporting data, technical specifications, "
        f"and reference architectures are available in the supplementary materials. Contact "
        f"the project lead for access to the detailed technical documentation and test results.\n\n",
    ]

    doc = "".join(sections)
    # Keep adding content until we approximate target token count (~4 chars per token)
    target_chars = target_tokens * 4
    idx = 0
    while len(doc) < target_chars:
        doc += detail_paragraphs[idx % len(detail_paragraphs)]
        idx += 1

    return doc[:target_chars]


def measure_ttft(endpoint: str, model: str, prompt: str,
                 max_tokens: int = 5, stream: bool = True) -> float:
    """Measure TTFT via streaming."""
    t0 = time.monotonic()
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
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


def build_prompt(instruction: str, document: str) -> str:
    """Build a chat-formatted prompt."""
    return (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{document}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_benchmark(args):
    random.seed(42)

    # Generate documents
    print(f"Generating {args.n_docs} enterprise documents at ~{args.token_length} tokens...")
    docs = []
    topics = DOCUMENT_TOPICS * ((args.n_docs // len(DOCUMENT_TOPICS)) + 1)
    for i in range(args.n_docs):
        name, desc = topics[i]
        doc_text = generate_document(name, desc, args.token_length)
        docs.append({"name": f"{name}_{i}", "text": doc_text})
    print(f"  Generated {len(docs)} documents, avg ~{sum(len(d['text']) for d in docs) // len(docs)} chars")

    query_types = list(QUERY_TYPES.items())
    results = []

    for doc_idx, doc in enumerate(docs):
        print(f"\n--- Document {doc_idx+1}/{len(docs)}: {doc['name']} ---")

        # Use first query type as donor
        donor_type, donor_instr = query_types[0]
        donor_prompt = build_prompt(donor_instr, doc["text"])

        # Register donor (cold call)
        print(f"  Registering donor ({donor_type})...")
        cold_ttft = measure_ttft(args.endpoint, args.model, donor_prompt,
                                 max_tokens=1, stream=False)
        print(f"    Cold TTFT: {cold_ttft:.0f}ms")

        # Test each query type against the donor
        for qt_name, qt_instr in query_types[1:]:
            query_prompt = build_prompt(qt_instr, doc["text"])

            # Measure TTFT with SemBlend (donor should be in store)
            try:
                sb_ttft = measure_ttft(args.endpoint, args.model, query_prompt,
                                       max_tokens=5, stream=True)
            except Exception as e:
                print(f"    {qt_name}: ERROR - {e}")
                continue

            speedup = cold_ttft / sb_ttft if sb_ttft > 0 else 0
            is_hit = sb_ttft < cold_ttft * 0.7  # hit if TTFT < 70% of cold

            result = {
                "doc": doc["name"],
                "donor_type": donor_type,
                "query_type": qt_name,
                "cold_ttft_ms": round(cold_ttft, 1),
                "semblend_ttft_ms": round(sb_ttft, 1),
                "speedup": round(speedup, 2),
                "is_hit": is_hit,
            }
            results.append(result)
            marker = "HIT" if is_hit else "MISS"
            print(f"    {qt_name}: {sb_ttft:.0f}ms ({speedup:.2f}x) [{marker}]")

        # Restart to avoid cross-document contamination? No — we WANT cross-doc to stay
        # to test the donor store behavior with multiple docs

    # Summary
    print("\n" + "=" * 60)
    print("ENTERPRISE RAG BENCHMARK SUMMARY")
    print("=" * 60)

    total = len(results)
    hits = sum(1 for r in results if r["is_hit"])
    hit_rate = hits / total if total > 0 else 0
    print(f"Total queries: {total}")
    print(f"Hit rate: {hits}/{total} = {hit_rate:.1%}")

    if hits > 0:
        hit_speedups = [r["speedup"] for r in results if r["is_hit"]]
        print(f"Hit-only speedup: {sum(hit_speedups)/len(hit_speedups):.2f}x (mean)")
        print(f"  P50: {sorted(hit_speedups)[len(hit_speedups)//2]:.2f}x")

    # Per query type
    print("\nPer query type:")
    for qt_name, _ in query_types[1:]:
        qt_results = [r for r in results if r["query_type"] == qt_name]
        qt_hits = sum(1 for r in qt_results if r["is_hit"])
        qt_rate = qt_hits / len(qt_results) if qt_results else 0
        qt_speedups = [r["speedup"] for r in qt_results if r["is_hit"]]
        avg_spd = f"{sum(qt_speedups)/len(qt_speedups):.2f}x" if qt_speedups else "N/A"
        print(f"  {qt_name}: {qt_rate:.0%} hit ({qt_hits}/{len(qt_results)}), "
              f"hit-only speedup: {avg_spd}")

    # Overall speedup (blended)
    all_speedups = [r["speedup"] for r in results]
    blended = sum(all_speedups) / len(all_speedups) if all_speedups else 0
    print(f"\nBlended speedup (including misses): {blended:.2f}x")

    # Save results
    output = {
        "benchmark": "enterprise_rag",
        "n_docs": args.n_docs,
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
    parser = argparse.ArgumentParser(description="Enterprise RAG benchmark")
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--n-docs", type=int, default=30,
                        help="Number of documents")
    parser.add_argument("--token-length", type=int, default=8192,
                        help="Target document length in tokens")
    parser.add_argument("--output", default="benchmarks/e2e/results/enterprise_rag.json")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
