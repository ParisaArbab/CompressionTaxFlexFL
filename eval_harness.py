"""
eval_harness.py
----------------
WP1 driver: "Baseline Auditing & The Compression Tax".

For each SWE-bench Lite instance:
  1. Clone the repo at base_commit, apply the test_patch, run the
     FAIL_TO_PASS tests to capture a REAL raw pytest failure log.
  2. Build an rtk-compressed version of that same log (rtk_simulator).
  3. Run agent_flexfl.run_agent() twice — once with the raw log as
     terminal_context (Control), once with the compressed log (Static) —
     against the same repo snapshot and the same problem_statement.
  4. Score both runs' ranked function lists against the gold patch's
     changed functions (Top-1/3/5, MAP, MRR — FlexFL's own metrics,
     Section 4.4).
  5. Flag "Compression Tax" instances: Control localizes the gold function
     at Top-5 and Static does not. Categorize WHY using the
     CompressionEvent trail from rtk_simulator, cross-referenced against
     which lines of the raw log the gold function's file/line appeared in.

Usage:
    python eval_harness.py --instances 5 --llm mock
    python eval_harness.py --instances 50 --llm anthropic --model claude-sonnet-5

`--llm mock` uses a deterministic stand-in (no API calls) so the harness,
metrics, and taxonomy logic can be validated end-to-end without a key.
Swap in `--llm anthropic` (requires ANTHROPIC_API_KEY) for real results.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import shutil
import unicodedata
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(__file__))
from rtk_simulator import compress, token_estimate, CompressionEvent
from agent_flexfl import RepoIndex, run_agent, LocalizationResult

REPO_CACHE = os.path.join(os.path.dirname(__file__), "..", "repos")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "swebench_lite_test.parquet")


# ---------------------------------------------------------------------------
# 0. Gold-label extraction: which functions did the real fix touch?
# ---------------------------------------------------------------------------

HUNK_HEADER_RE = re.compile(r"^@@ .* @@\s*(.*)$")
DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)")


def gold_functions_from_patch(patch_text: str) -> list[str]:
    """Best-effort extraction of 'file.py:function_or_class' touched by the
    gold patch, using the hunk header's trailing context (the enclosing
    def/class line `git diff` prints after @@ ... @@)."""
    out = []
    current_file = None
    for line in patch_text.splitlines():
        fm = DIFF_FILE_RE.match(line)
        if fm:
            current_file = fm.group(2)
            continue
        hm = HUNK_HEADER_RE.match(line)
        if hm and current_file:
            ctx = hm.group(1).strip()
            m = re.search(r"def\s+(\w+)", ctx) or re.search(r"class\s+(\w+)", ctx)
            if m:
                out.append(f"{current_file}:{m.group(1)}")
    return sorted(set(out))


# ---------------------------------------------------------------------------
# 1. Repo setup + real raw log capture
# ---------------------------------------------------------------------------

@dataclass
class InstanceSetup:
    instance_id: str
    repo_dir: str
    raw_log: str
    ok: bool
    error: str = ""


def setup_instance(row: dict, venv_python: str, force: bool = False) -> InstanceSetup:
    iid = row["instance_id"]
    org_repo = row["repo"]
    repo_name = org_repo.split("/")[-1]
    repo_dir = os.path.join(REPO_CACHE, f"{iid}")
    log_path = os.path.join(CACHE_DIR, f"{iid}_raw.log")

    if os.path.exists(log_path) and not force:
        return InstanceSetup(iid, repo_dir, open(log_path).read(), ok=True)

    try:
        if not os.path.exists(repo_dir):
            subprocess.run(["git", "clone", "-q", f"https://github.com/{org_repo}.git", repo_dir], check=True)
        subprocess.run(["git", "-C", repo_dir, "checkout", "-q", "-f", row["base_commit"]], check=True)
        subprocess.run(["git", "-C", repo_dir, "clean", "-qfdx"], check=False)
        test_patch_path = os.path.join(CACHE_DIR, f"{iid}_test.diff")
        open(test_patch_path, "w").write(row["test_patch"])
        subprocess.run(["git", "-C", repo_dir, "apply", "--whitespace=fix", test_patch_path], check=True)

        test_ids = json.loads(row["FAIL_TO_PASS"]) if isinstance(row["FAIL_TO_PASS"], str) else list(row["FAIL_TO_PASS"])
        cmd = [venv_python, "-m", "pytest", *test_ids, "-rA", "-o", "filterwarnings=default"]
        proc = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=300)
        raw_log = proc.stdout + proc.stderr
        open(log_path, "w").write(raw_log)
        return InstanceSetup(iid, repo_dir, raw_log, ok=True)
    except Exception as e:
        return InstanceSetup(iid, repo_dir, "", ok=False, error=str(e))


# ---------------------------------------------------------------------------
# 2. Metrics (FlexFL Section 4.4)
# ---------------------------------------------------------------------------

def score(ranked: list[str], gold: list[str]) -> dict:
    def norm(fq):
        return fq.split(":")[-1].split(".")[-1]  # compare on bare function name (best-effort)
    gold_names = {norm(g) for g in gold}
    hit_ranks = [i + 1 for i, r in enumerate(ranked) if norm(r) in gold_names]
    top1 = 1 if hit_ranks and hit_ranks[0] <= 1 else 0
    top3 = 1 if hit_ranks and hit_ranks[0] <= 3 else 0
    top5 = 1 if hit_ranks and hit_ranks[0] <= 5 else 0
    mrr = 1 / hit_ranks[0] if hit_ranks else 0.0
    ap = sum((k + 1) / r for k, r in enumerate(hit_ranks)) / max(1, len(gold_names)) if hit_ranks else 0.0
    return {"top1": top1, "top3": top3, "top5": top5, "mrr": mrr, "map": ap, "first_hit_rank": hit_ranks[0] if hit_ranks else None}


# ---------------------------------------------------------------------------
# 3. Mock LLM (deterministic, no API key needed — for harness validation)
# ---------------------------------------------------------------------------

def make_mock_llm(index: RepoIndex, gold: list[str], leak_rate: float = 0.6):
    """A stand-in 'LLM' that behaves like a mediocre agent: it correctly
    finds the gold function `leak_rate` of the time when its file/line
    signature is PRESENT in the given context text, and guesses randomly
    otherwise. This lets us validate the harness's plumbing (metrics,
    compression-tax detection) without spending API credits. It is NOT a
    substitute for a real model run — see README for wiring Claude/GPT in.
    """
    import random
    rnd = random.Random(42)

    def llm_call(system_prompt: str, user_prompt: str) -> str:
        if "Top_1:" in user_prompt or "provide the top" in user_prompt:
            found = [g for g in gold if re.search(re.escape(g.split(":")[-1]), user_prompt)]
            if found and rnd.random() < leak_rate:
                lines = [f"Top_{i+1}: {g}" for i, g in enumerate(found[:5])]
                pool = list(index.symbols.keys())
                while len(lines) < 5 and pool:
                    lines.append(f"Top_{len(lines)+1}: {rnd.choice(pool)}")
                return "\n".join(lines)
            pool = list(index.symbols.keys()) or ["unknown.py:unknown"]
            return "\n".join(f"Top_{i+1}: {rnd.choice(pool)}" for i in range(5))
        return "exit()"

    return llm_call


# ---------------------------------------------------------------------------
# 4. Compression-tax detection
# ---------------------------------------------------------------------------

def diagnose_compression_tax(gold: list[str], raw_log: str, events: list[CompressionEvent]) -> list[dict]:
    """For each gold function, check whether its file/line signature was
    inside a block that got dropped/grouped/truncated by rtk."""
    findings = []
    for g in gold:
        fname = g.split(":")[-1].split(".")[-1]
        for ev in events:
            joined = "\n".join(ev.original_lines)
            if fname and fname in joined and fname not in ev.replacement:
                findings.append({
                    "gold_function": g,
                    "compression_event": ev.kind,
                    "note": ev.note,
                    "evidence_line": next((l for l in ev.original_lines if fname in l), "")[:200],
                })
    return findings


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--repos", nargs="*", default=None, help="restrict to these org/repo names")
    ap.add_argument("--llm", choices=["mock", "anthropic"], default="mock")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--venv-python", default=sys.executable)
    ap.add_argument("--out", default=os.path.join(CACHE_DIR, "wp1_results.jsonl"))
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(DATA_PATH)
    if args.repos:
        df = df[df.repo.isin(args.repos)]
    df = df.head(args.instances)

    results = []
    for _, row in df.iterrows():
        row = row.to_dict()
        iid = row["instance_id"]
        setup = setup_instance(row, args.venv_python)
        if not setup.ok or not setup.raw_log.strip():
            print(f"[SKIP] {iid}: setup failed ({setup.error[:120]})")
            continue

        gold = gold_functions_from_patch(row["patch"])
        if not gold:
            print(f"[SKIP] {iid}: could not extract gold functions from patch")
            continue

        compressed_log, events = compress(setup.raw_log)
        index = RepoIndex(setup.repo_dir)

        if args.llm == "mock":
            llm_call = make_mock_llm(index, gold)
        else:
            from anthropic_llm import make_anthropic_llm  # user-provided, see README
            llm_call = make_anthropic_llm(model=args.model)

        control = run_agent(row["problem_statement"], setup.raw_log, index, llm_call)
        static = run_agent(row["problem_statement"], compressed_log, index, llm_call)

        control_score = score(control.ranked_fqns, gold)
        static_score = score(static.ranked_fqns, gold)

        compression_tax = control_score["top5"] == 1 and static_score["top5"] == 0
        taxonomy = diagnose_compression_tax(gold, setup.raw_log, events) if compression_tax else []

        rec = {
            "instance_id": iid,
            "repo": row["repo"],
            "gold_functions": gold,
            "raw_tokens_est": token_estimate(setup.raw_log),
            "rtk_tokens_est": token_estimate(compressed_log),
            "reduction_pct": round(100 * (1 - token_estimate(compressed_log) / max(1, token_estimate(setup.raw_log))), 1),
            "control_ranked": control.ranked_fqns,
            "static_ranked": static.ranked_fqns,
            "control_score": control_score,
            "static_score": static_score,
            "compression_tax": compression_tax,
            "compression_tax_taxonomy": taxonomy,
        }
        results.append(rec)
        print(f"[{iid}] ctrl top1/3/5={control_score['top1']}/{control_score['top3']}/{control_score['top5']} "
              f"static top1/3/5={static_score['top1']}/{static_score['top3']}/{static_score['top5']} "
              f"reduction={rec['reduction_pct']}% tax={compression_tax}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    if results:
        def agg(key, metric):
            return sum(r[key][metric] for r in results) / len(results)
        print("\n=== AGGREGATE (n={}) ===".format(len(results)))
        for cond in ["control_score", "static_score"]:
            print(cond, {m: round(agg(cond, m), 3) for m in ["top1", "top3", "top5", "mrr", "map"]})
        n_tax = sum(1 for r in results if r["compression_tax"])
        print(f"Compression Tax instances: {n_tax}/{len(results)} ({100*n_tax/len(results):.1f}%)")
    print(f"\nWrote {len(results)} records to {args.out}")


if __name__ == "__main__":
    main()
