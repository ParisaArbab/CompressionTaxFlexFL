# WP1 — Baseline Auditing & the "Compression Tax"

Deliverables for Work Package 1, built on **real SWE-bench_Lite data** (not
synthetic fixtures) and the **FlexFL** two-stage agent pipeline, adapted from
Java/Defects4J to Python/SWE-bench.

## What's here

| File | Role |
|---|---|
| `scripts/rtk_simulator.py` | Documented-behavior re-implementation of `rtk`'s compression (filter → group → truncate). See file docstring for why the real binary couldn't be fetched in this sandbox, and how to swap it in. |
| `scripts/agent_flexfl.py` | Python port of FlexFL's agent pipeline: `ast`-based repo indexer (FlexFL §3.2.2's "traverse all files, extract FQNs"), function-call tool layer (Table 2 equivalent), and the reason→act→summarize loop (§3.2.1) with Levenshtein postprocessing (Algorithm 1). |
| `scripts/eval_harness.py` | Orchestrates Control (raw) vs Static (rtk) runs per instance, scores with FlexFL's own metrics (Top-1/3/5, MAP, MRR — §4.4), and flags **Compression Tax** cases. |
| `data/swebench_lite_test.parquet` | Real `princeton-nlp/SWE-bench_Lite` (300 instances). |
| `cache/flask4045_raw.log` / `flask4045_rtk.log` | A real captured pytest failure log and its rtk-compressed counterpart (see Error Taxonomy Report). |
| `reports/WP1_Error_Taxonomy_Report.docx` | The formal deliverable. |

## What was actually run vs. what's ready-to-run

**Actually executed in this environment (real, verifiable):**
- Downloaded the real SWE-bench_Lite dataset (300 instances, 12 repos).
- Cloned `pallets/flask` at the exact `base_commit` for instance
  `pallets__flask-4045`, applied the real `test_patch`, and ran the real
  `FAIL_TO_PASS` tests in an isolated venv to capture a genuine raw pytest
  failure log (not a mock).
- Ran that log through the rtk heuristic simulator: **88.9% token
  reduction** (in-range with rtk's documented 60–90%), and found — by
  inspecting the compression events, not by assumption — that the
  truncation step hides the exact line (`src/flask/blueprints.py:364:
  AssertionError`) that pinpoints the bug. That's a live example of the
  Compression Tax; details in the report.
- Ran `eval_harness.py` end-to-end (repo setup → compression → agent →
  scoring → tax detection) in `--llm mock` mode to validate every stage of
  the pipeline works on real data before spending API budget on it.

**Not executed (needs credentials/compute you control):**
- Full-scale runs across the SWE-bench Lite subset with a real LLM
  (`--llm anthropic`, or wire in OpenAI/DeepSeek/Qwen the same way — see
  `make_anthropic_llm` stub referenced in `eval_harness.py`). No API key
  was available in this sandbox.
- The real `rtk` binary. GitHub's release API was rate-limited and no Rust
  toolchain was available, so `rtk_simulator.py` is a careful
  documented-behavior stand-in, deliberately tuned to *under*-compress
  relative to the real tool's reported numbers — meaning the real
  Compression Tax is likely **at least as large** as what this harness
  will report, not smaller.

## How to scale this to the full WP1 deliverable

```bash
# 1. Point at your own LLM backend (add scripts/anthropic_llm.py with
#    make_anthropic_llm(model) -> (system, user) -> str, calling the
#    Anthropic Messages API with your key)
export ANTHROPIC_API_KEY=...

# 2. Run across a meaningful subset (start with ~30-50 instances spanning
#    all 12 repos before committing to the full 300)
python scripts/eval_harness.py --instances 50 --llm anthropic --model claude-sonnet-5

# 3. Repeat with GPT/DeepSeek/Qwen backends for the cross-model comparison
#    called for in the project brief.

# 4. Aggregate wp1_results.jsonl across models/repos into the final
#    Error Taxonomy Report (the current report's taxonomy table generalizes;
#    swap in the larger sample's case studies).
```

## Known adaptation from FlexFL (documented, not hidden)

FlexFL is method-level FL for **Java**/Defects4J with a **two-stage**
pipeline (Agent4SR + Agent4LR) and non-LLM FL techniques (BoostN/Ochiai/SBIR)
feeding the candidate list. For WP1 we implemented the **single-stage**
core loop (task assignment → ReAct function-calling → structured
summarization + Levenshtein postprocessing) ported to **Python**/SWE-bench,
because WP1's question is about *context compression*, not about space
reduction. WP2/WP3 will extend this into FlexFL's full two-stage design
(and layer in GraphLocator as the second FL family called for in the brief).
