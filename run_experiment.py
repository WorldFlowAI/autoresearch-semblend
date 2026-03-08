"""
Experiment harness for SemBlend autoresearch.
Three tiers: unit tests, component microbenchmarks, full E2E deploy+benchmark.

Usage:
    python run_experiment.py --tier 1                    # pytest
    python run_experiment.py --tier 2                    # component microbenchmarks
    python run_experiment.py --tier 3 --bench ttft       # full E2E TTFT benchmark
    python run_experiment.py --tier 3 --bench quality    # full E2E quality benchmark
    python run_experiment.py --tier 3 --bench all        # all benchmarks
"""

import argparse
import os
import re
import subprocess
import sys
import time

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
INFRA_DIR = os.path.join(REPO_DIR, "infra")
BENCHMARKS_DIR = os.path.join(REPO_DIR, "benchmarks", "e2e")
SYNAPSE_DIR = os.path.expanduser("~/dev/worldflowai/ONECONTEXT/synapse")
NAMESPACE = "autoresearch"

VALID_BENCHMARKS = [
    "ttft",
    "quality",
    "ablation",
    "memory",
    "scale",
    "all",
]


# ---------------------------------------------------------------------------
# Tier 1: Unit tests
# ---------------------------------------------------------------------------


def run_tier1():
    """Run pytest on synapse_kv_connector tests."""
    test_dir = os.path.join(REPO_DIR, "synapse_kv_connector", "tests")
    if not os.path.isdir(test_dir):
        print("ERROR: test directory not found. Run prepare.py first.")
        return 1

    print("=== Tier 1: Unit Tests ===")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_dir, "-v", "--tb=short"],
        cwd=REPO_DIR,
    )

    status = "pass" if result.returncode == 0 else "fail"
    print(f"\n---")
    print(f"tier:     1")
    print(f"status:   {status}")
    print(f"---")
    return result.returncode


# ---------------------------------------------------------------------------
# Tier 2: Component microbenchmarks
# ---------------------------------------------------------------------------


def run_tier2():
    """Run component-level microbenchmarks (no deployment needed)."""
    print("=== Tier 2: Component Microbenchmarks ===")

    microbench = os.path.join(BENCHMARKS_DIR, "microbench.py")
    component_bench = os.path.join(BENCHMARKS_DIR, "component_bench.py")

    bench_script = microbench if os.path.exists(microbench) else component_bench
    if not os.path.exists(bench_script):
        print("ERROR: no microbenchmark script found in benchmarks/e2e/")
        return 1

    result = subprocess.run(
        [sys.executable, bench_script],
        cwd=BENCHMARKS_DIR,
    )

    status = "pass" if result.returncode == 0 else "fail"
    print(f"\n---")
    print(f"tier:     2")
    print(f"status:   {status}")
    print(f"---")
    return result.returncode


# ---------------------------------------------------------------------------
# Tier 3: Full E2E (deploy + benchmark)
# ---------------------------------------------------------------------------


def ensure_gpu():
    """Run ensure_gpu.sh — provision A10G if not running."""
    script = os.path.join(INFRA_DIR, "ensure_gpu.sh")
    print("--- Ensuring GPU node ---")
    result = subprocess.run(["bash", script], timeout=360)
    if result.returncode != 0:
        print("ERROR: GPU provisioning failed")
        return False
    return True


def deploy():
    """Run deploy.sh — build, push, helm upgrade."""
    script = os.path.join(INFRA_DIR, "deploy.sh")
    print("--- Deploying to autoresearch namespace ---")
    result = subprocess.run(["bash", script], timeout=900)
    if result.returncode != 0:
        print("ERROR: Deploy failed")
        return False
    return True


def wait_for_vllm(timeout=600):
    """Wait for vLLM pod to be ready and responding."""
    print("--- Waiting for vLLM readiness ---")
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "pods",
                    "-n",
                    NAMESPACE,
                    "-l",
                    "app.kubernetes.io/component=vllm",
                    "-o",
                    "jsonpath={.items[0].status.phase}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.stdout.strip() == "Running":
                # Check if model is loaded by hitting health endpoint
                health = subprocess.run(
                    ["curl", "-sf", "http://localhost:8100/health"],
                    capture_output=True,
                    timeout=5,
                )
                if health.returncode == 0:
                    elapsed = time.time() - start
                    print(f"vLLM ready ({elapsed:.0f}s)")
                    return True
        except (subprocess.TimeoutExpired, Exception):
            pass
        time.sleep(10)
    print(f"ERROR: vLLM not ready after {timeout}s")
    return False


def run_benchmark(bench_name):
    """Run a specific benchmark script against the deployed vLLM."""
    bench_map = {
        "ttft": "semblend_ttft_scaling.py",
        "quality": "semblend_quality_bench.py",
        "ablation": "embedding_ablation_bench.py",
        "memory": "semblend_memory_bench.py",
        "scale": "semblend_scale_bench.py",
    }

    script_name = bench_map.get(bench_name)
    if not script_name:
        print(f"ERROR: unknown benchmark: {bench_name}")
        return None

    script_path = os.path.join(BENCHMARKS_DIR, script_name)
    if not os.path.exists(script_path):
        # Try alternative names
        alt_names = [
            f"semblend_{bench_name}_bench.py",
            f"semblend_{bench_name}.py",
            f"{bench_name}_bench.py",
        ]
        for alt in alt_names:
            alt_path = os.path.join(BENCHMARKS_DIR, alt)
            if os.path.exists(alt_path):
                script_path = alt_path
                break
        else:
            print(f"ERROR: benchmark script not found: {script_name}")
            return None

    print(f"--- Running benchmark: {bench_name} ({os.path.basename(script_path)}) ---")
    start = time.time()

    result = subprocess.run(
        [
            sys.executable,
            script_path,
            "--base-url",
            "http://localhost:8100/v1",
        ],
        capture_output=True,
        text=True,
        cwd=BENCHMARKS_DIR,
        timeout=2700,  # 45 min max
    )

    duration = time.time() - start

    if result.returncode != 0:
        print(f"Benchmark failed (exit {result.returncode})")
        if result.stderr:
            print(result.stderr[-500:])
        return {
            "benchmark": bench_name,
            "status": "fail",
            "duration_seconds": int(duration),
        }

    output = result.stdout
    print(output[-2000:])  # Print last 2000 chars of output

    # Parse metrics from output
    metrics = parse_benchmark_output(output, bench_name)
    metrics["duration_seconds"] = int(duration)
    return metrics


def parse_benchmark_output(output, bench_name):
    """Extract metrics from benchmark output."""
    metrics = {"benchmark": bench_name, "status": "pass"}

    # Look for common metric patterns
    patterns = {
        "ttft_speedup_2k": r"(?:2k|2048).*?speedup.*?(\d+\.\d+)",
        "ttft_speedup_5k": r"(?:5k|5120).*?speedup.*?(\d+\.\d+)",
        "ttft_speedup_8k": r"(?:8k|8192).*?speedup.*?(\d+\.\d+)",
        "ttft_speedup_16k": r"(?:16k|16384).*?speedup.*?(\d+\.\d+)",
        "rouge_l_avg": r"rouge.?l.*?(\d+\.\d+)",
        "ppl_ratio_avg": r"ppl.?ratio.*?(\d+\.\d+)",
        "hit_rate_avg": r"hit.?rate.*?(\d+\.\d+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except ValueError:
                pass

    return metrics


def run_tier3(bench_name):
    """Full E2E: ensure GPU, deploy, wait, benchmark."""
    print("=== Tier 3: Full E2E Benchmark ===")
    start = time.time()

    # Step 1: Ensure GPU
    if not ensure_gpu():
        return 1

    # Step 2: Deploy
    if not deploy():
        return 1

    # Step 3: Wait for vLLM
    if not wait_for_vllm():
        return 1

    # Step 4: Run benchmark(s)
    benchmarks = VALID_BENCHMARKS[:-1] if bench_name == "all" else [bench_name]
    all_metrics = {}

    for bname in benchmarks:
        metrics = run_benchmark(bname)
        if metrics:
            all_metrics.update(metrics)

    total_duration = int(time.time() - start)

    # Print parseable output
    print("\n---")
    print(f"tier:               3")
    for key, value in sorted(all_metrics.items()):
        if key not in ("status", "benchmark"):
            if isinstance(value, float):
                print(f"{key + ':':20s}{value:.3f}")
            else:
                print(f"{key + ':':20s}{value}")
    print(f"{'duration_seconds:':20s}{total_duration}")
    print("---")

    return 0 if all_metrics.get("status") != "fail" else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SemBlend experiments"
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Experiment tier: 1=unit tests, 2=microbenchmarks, 3=full E2E",
    )
    parser.add_argument(
        "--bench",
        type=str,
        default="ttft",
        choices=VALID_BENCHMARKS,
        help="Benchmark to run (tier 3 only)",
    )
    args = parser.parse_args()

    if args.tier == 1:
        sys.exit(run_tier1())
    elif args.tier == 2:
        sys.exit(run_tier2())
    elif args.tier == 3:
        sys.exit(run_tier3(args.bench))
