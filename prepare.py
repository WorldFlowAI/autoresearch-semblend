"""
One-time setup for autoresearch-semblend experiments.
Copies SemBlend source code, benchmarks, and paper from the synapse repo
into this working directory for autonomous experimentation.

Usage:
    python prepare.py                    # copy code + verify
    python prepare.py --build-datasets   # also build benchmark dataset clusters
    python prepare.py --provision-gpu    # also provision A10G node
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SYNAPSE_DIR = os.path.expanduser("~/dev/worldflowai/ONECONTEXT/synapse")

SOURCE_PATHS = {
    "synapse_kv_connector": os.path.join(
        SYNAPSE_DIR, "services", "vllm", "synapse_kv_connector"
    ),
    "benchmarks/e2e": os.path.join(SYNAPSE_DIR, "benchmarks", "e2e"),
    "paper": os.path.join(SYNAPSE_DIR, "paper"),
}

DEST_PATHS = {
    "synapse_kv_connector": os.path.join(REPO_DIR, "synapse_kv_connector"),
    "benchmarks/e2e": os.path.join(REPO_DIR, "benchmarks", "e2e"),
    "paper": os.path.join(REPO_DIR, "paper"),
}

HASH_FILE = os.path.join(REPO_DIR, ".source_hash")

# ---------------------------------------------------------------------------
# Copy logic
# ---------------------------------------------------------------------------


def compute_dir_hash(path):
    """Compute a hash of all files in a directory tree for change detection."""
    h = hashlib.sha256()
    for root, _dirs, files in sorted(os.walk(path)):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, path)
            h.update(rel.encode())
            try:
                h.update(open(fpath, "rb").read())
            except (OSError, PermissionError):
                pass
    return h.hexdigest()


def copy_source(name, src, dst, force=False):
    """Copy a source directory, skipping if unchanged (unless forced)."""
    if not os.path.isdir(src):
        print(f"  ERROR: source not found: {src}")
        return False

    if os.path.isdir(dst) and not force:
        src_hash = compute_dir_hash(src)
        dst_hash = compute_dir_hash(dst)
        if src_hash == dst_hash:
            file_count = sum(1 for _, _, files in os.walk(dst) for _ in files)
            print(f"  {name}: up to date ({file_count} files)")
            return True

    # Remove old copy and replace
    if os.path.exists(dst):
        shutil.rmtree(dst)

    # Exclude __pycache__, .pyc, egg-info
    def ignore(directory, contents):
        return [
            c
            for c in contents
            if c == "__pycache__"
            or c.endswith(".pyc")
            or c.endswith(".egg-info")
        ]

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(src, dst, ignore=ignore)
    file_count = sum(1 for _, _, files in os.walk(dst) for _ in files)
    print(f"  {name}: copied {file_count} files")
    return True


def copy_analysis_to_paper():
    """Copy ANALYSIS.md into the paper/ directory if not already there."""
    analysis_src = os.path.join(SYNAPSE_DIR, "paper", "ANALYSIS.md")
    analysis_dst = os.path.join(REPO_DIR, "paper", "ANALYSIS.md")
    if os.path.exists(analysis_dst):
        print("  ANALYSIS.md: already in paper/")
        return
    if os.path.exists(analysis_src):
        shutil.copy2(analysis_src, analysis_dst)
        print("  ANALYSIS.md: copied to paper/")


# ---------------------------------------------------------------------------
# Literature seeding
# ---------------------------------------------------------------------------


def seed_literature():
    """Create initial literature directory with reading list and notes."""
    lit_dir = os.path.join(REPO_DIR, "literature")
    os.makedirs(lit_dir, exist_ok=True)

    reading_list = os.path.join(lit_dir, "reading_list.md")
    if not os.path.exists(reading_list):
        with open(reading_list, "w") as f:
            f.write("""# Reading List

## Priority 1: Direct Competitors
- [ ] **SemShareKV** — Semantic KV Cache Sharing for Large Language Models (2025)
  - Token-level LSH matching, RoPE-aware E-cache, 3 models x 9 datasets
  - Key comparison target for SemBlend paper
- [ ] **CacheBlend** — Fast Large Language Model Serving with Cached Knowledge Fusion (EuroSys'25)
  - Selective recomputation for KV cache blending
  - Potential baseline comparison

## Priority 2: Related Systems
- [ ] **LMCache** — KV cache management for vLLM
  - SemBlend's underlying KV transport layer
  - Understand chunk storage, retrieval, connector API
- [ ] **vLLM** — PagedAttention and prefix caching
  - Production baseline for prefix-match KV reuse

## Priority 3: Theoretical Foundations
- [ ] **RoFormer** — Enhanced Transformer with Rotary Position Embedding (Su et al., 2024)
  - Theoretical basis for RoPE delta correction
  - Exact position correction proof
- [ ] **SnapKV** — LLM Knows What You Are Looking For Before Generation
  - KV cache compression baseline (SemShareKV compares against this)
- [ ] **PyramidKV** — Dynamic KV Cache Compression
  - Another baseline from SemShareKV evaluation
""")
        print("  reading_list.md: created")

    semshare_notes = os.path.join(lit_dir, "semshareKV_notes.md")
    if not os.path.exists(semshare_notes):
        with open(semshare_notes, "w") as f:
            f.write("""# SemShareKV Notes

## Key Architecture Differences vs SemBlend

| Aspect | SemShareKV | SemBlend |
|--------|-----------|----------|
| Matching | Token-level LSH, O(T) | Embedding-level, O(1) |
| Position | First-layer recompute (heuristic) | Exact RoPE delta correction |
| Framework | HuggingFace Transformers (custom) | vLLM + LMCache (production) |
| Donors | Single reference per target | Multi-donor store (10K+) |
| Max sequence | 5K tokens | 16K+ tokens |

## Evaluation Methodology
- 3 models: Mistral-7B, LLaMA-3.1-8B, MPT-7B
- 9 datasets: MultiNews, WikiHow, Qasper, SAMSum, PubMed, BookSum, BigPatent, LCC, MMLU
- 4 baselines: Full Recompute, SnapKV, PyramidKV, H2O
- 3 ablation studies: fuzzy+full cache, zero ablation, random ablation
- Quality metric: ROUGE-L on full generation outputs
- 42% KV cache memory reduction reported

## What We Must Match
1. Multi-model testing (at least 2 models)
2. Real-world datasets (at least 3)
3. Baseline comparisons (at least prefix caching + cold)
4. Ablation studies (at least threshold sweep + embedder comparison)
5. Quality metrics on meaningful output (max_tokens=256)

## Where We're Already Better
1. Production vLLM integration (vs HuggingFace hack)
2. Exact RoPE correction (vs brute-force first-layer recompute)
3. Longer sequences (16K vs 5K)
4. Honest negative results (0.72x on diverse workloads)
5. Statistical methodology (n=10, per-length restarts)
""")
        print("  semshareKV_notes.md: created")

    lmcache_notes = os.path.join(lit_dir, "lmcache_notes.md")
    if not os.path.exists(lmcache_notes):
        with open(lmcache_notes, "w") as f:
            f.write("""# LMCache Notes

## Architecture
- KV cache chunk storage and retrieval for vLLM
- SemBlend wraps LMCacheConnectorV1 with semantic donor discovery
- Chunk-swap injection: contiguous prefix replacement (delta=0)
- CacheBlend mode: selective recomputation of mismatched tokens

## Key Integration Points
- `LMCacheConnectorV1`: vLLM's KV connector interface
- `SemBlendConnectorV1`: wraps LMCache, adds semantic matching
- Chunk storage: CPU DRAM (warm), can offload to disk (cold)

## TODO
- [ ] Read LMCache source to understand chunk format
- [ ] Understand CacheBlend selective recomputation API
- [ ] Map connector API for multi-model support
""")
        print("  lmcache_notes.md: created")


# ---------------------------------------------------------------------------
# Infrastructure verification
# ---------------------------------------------------------------------------


def verify_kubectl():
    """Check kubectl connectivity."""
    try:
        result = subprocess.run(
            ["kubectl", "cluster-info", "--request-timeout=5s"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("  kubectl: connected")
            return True
        print(f"  kubectl: connection failed ({result.stderr.strip()})")
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  kubectl: not available or timed out")
        return False


def check_gpu_nodes():
    """Check if A10G GPU nodes are running."""
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "nodes",
                "-l",
                "gpu-type=a10g",
                "--no-headers",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        ready_count = result.stdout.strip().count("Ready")
        if ready_count > 0:
            print(f"  GPU nodes: {ready_count} A10G node(s) Ready")
            return True
        print("  GPU nodes: none running")
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  GPU nodes: check failed")
        return False


def provision_gpu():
    """Run ensure_gpu.sh to provision A10G node."""
    script = os.path.join(REPO_DIR, "infra", "ensure_gpu.sh")
    if not os.path.exists(script):
        print("  GPU provision: ensure_gpu.sh not found")
        return False
    print("  Provisioning GPU node...")
    result = subprocess.run(
        ["bash", script],
        capture_output=False,
        timeout=360,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Results file
# ---------------------------------------------------------------------------


def init_results():
    """Create results.tsv with header if it doesn't exist."""
    results_path = os.path.join(REPO_DIR, "results.tsv")
    if os.path.exists(results_path):
        print("  results.tsv: already exists")
        return

    with open(results_path, "w") as f:
        f.write(
            "commit\ttier\tbenchmark\tprimary_metric\tprimary_value\t"
            "secondary_metrics\tstatus\tdescription\n"
        )
    print("  results.tsv: created with header")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare autoresearch-semblend working directory"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-copy even if source hasn't changed",
    )
    parser.add_argument(
        "--build-datasets",
        action="store_true",
        help="Build benchmark dataset clusters after copying",
    )
    parser.add_argument(
        "--provision-gpu",
        action="store_true",
        help="Provision A10G GPU node if not running",
    )
    args = parser.parse_args()

    print("=== Autoresearch SemBlend Setup ===\n")

    # Step 1: Verify synapse repo exists
    if not os.path.isdir(SYNAPSE_DIR):
        print(f"ERROR: synapse repo not found at {SYNAPSE_DIR}")
        sys.exit(1)
    print(f"Source: {SYNAPSE_DIR}")
    print(f"Target: {REPO_DIR}\n")

    # Step 2: Copy source code
    print("Copying source code:")
    all_ok = True
    for name, src in SOURCE_PATHS.items():
        dst = DEST_PATHS[name]
        if not copy_source(name, src, dst, force=args.force):
            all_ok = False
    copy_analysis_to_paper()
    print()

    # Step 3: Seed literature
    print("Seeding literature:")
    seed_literature()
    print()

    # Step 4: Initialize results
    print("Results tracking:")
    init_results()
    print()

    # Step 5: Verify infrastructure
    print("Infrastructure:")
    kubectl_ok = verify_kubectl()
    gpu_ok = check_gpu_nodes()
    if args.provision_gpu and not gpu_ok:
        gpu_ok = provision_gpu()
    print()

    # Step 6: Summary
    print("=== Setup Summary ===")
    src_files = sum(
        1
        for dst in DEST_PATHS.values()
        if os.path.isdir(dst)
        for _, _, files in os.walk(dst)
        for _ in files
    )
    print(f"  Source files copied: {src_files}")
    print(f"  kubectl connected:  {kubectl_ok}")
    print(f"  GPU nodes running:  {gpu_ok}")
    print(f"  Results tracking:   {os.path.exists(os.path.join(REPO_DIR, 'results.tsv'))}")
    print()

    if all_ok:
        print("Done! Ready for experiments.")
    else:
        print("WARNING: Some copies failed. Check source paths.")
        sys.exit(1)
