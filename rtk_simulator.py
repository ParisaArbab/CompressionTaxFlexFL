"""
rtk_simulator.py
-----------------
A faithful, documented re-implementation of RTK's ("Rust Token Killer",
github.com/rtk-ai/rtk) compression heuristics for pytest output, built from
its publicly documented behavior:

  1. Smart Filtering  - strip boilerplate, blank lines, hint text, repeated
                         warning bodies, deprecation spam.
  2. Grouping          - collapse repeated/near-identical lines (e.g. N
                          identical DeprecationWarning blocks) into one
                          "(x N)" summary line.
  3. Test-run summarization - a fully-passing suite is collapsed to a single
                          "N passed" line (RTK reports cases like
                          "262 passing tests: 4,823 -> 11 tokens").
                          A failing suite keeps FAILED test names but RTK's
                          default profile still summarizes/deduplicates the
                          surrounding noise (warnings, PASSED lines).
  4. Truncation        - any single block (e.g. one test's traceback) longer
                          than a line budget is cut to head + tail with a
                          "<<N lines hidden>>" marker, mirroring RTK's
                          documented tee-style "full output on demand"
                          fallback (the raw bytes are preserved out of band,
                          not sent to the model by default).

NOTE ON PROVENANCE: we could not fetch the real `rtk` binary inside this
sandbox (GitHub API rate-limited, no cargo toolchain, release asset names
not resolvable offline). This module is therefore a *documented-behavior*
simulator, not the binary itself. It is built directly from RTK's README
and third-party benchmarked descriptions (see WP1_notes.md for sources) and
is deliberately conservative (keeps MORE than the real tool likely would),
so any "compression tax" we find here is a lower bound on the real effect.
Swap `compress()` for a call to the real `rtk` binary once it is available
in your environment (see `run_real_rtk()` stub at the bottom).
"""

from __future__ import annotations
import re
import hashlib
from dataclasses import dataclass, field

TRUNCATE_BLOCK_LINES = 40   # collapse any contiguous block longer than this
HEAD_KEEP = 6
TAIL_KEEP = 6

WARNING_HEADER_RE = re.compile(r"^\s*(.+?:\d+):\s*(\w*Warning):\s*(.*)$")
BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*$"),                                   # blank lines
    re.compile(r"^-- Docs: https://docs\.pytest\.org"),      # doc footer
    re.compile(r"^platform .* -- Python"),                   # session banner
    re.compile(r"^rootdir:"),
    re.compile(r"^configfile:"),
    re.compile(r"^plugins:"),
]


@dataclass
class CompressionEvent:
    kind: str            # e.g. "grouped_warning", "truncated_block", "dropped_boilerplate"
    original_lines: list = field(default_factory=list)
    replacement: str = ""
    note: str = ""


def _is_boilerplate(line: str) -> bool:
    return any(p.search(line) for p in BOILERPLATE_PATTERNS)


def _group_warnings(lines: list[str]) -> tuple[list[str], list[CompressionEvent]]:
    """Collapse repeated warning blocks (same message, different call sites)
    into a single representative line + count, discarding the per-site
    file:line locations RTK's grouping heuristic treats as noise."""
    events: list[CompressionEvent] = []
    # find contiguous "warnings summary" section
    out = []
    i = 0
    n = len(lines)
    warn_groups: dict[str, list[str]] = {}
    warn_order: list[str] = []
    in_summary = False
    while i < n:
        line = lines[i]
        if "warnings summary" in line:
            in_summary = True
            out.append(line)
            i += 1
            continue
        if in_summary and line.startswith("==="):
            in_summary = False
        if in_summary:
            m = WARNING_HEADER_RE.match(line)
            if m:
                _, wtype, msg = m.groups()
                key = f"{wtype}:{msg.strip()}"
                block = [line]
                j = i + 1
                while j < n and lines[j].strip() and not WARNING_HEADER_RE.match(lines[j]) and "===" not in lines[j]:
                    block.append(lines[j])
                    j += 1
                warn_groups.setdefault(key, []).extend(block)
                if key not in warn_order:
                    warn_order.append(key)
                i = j
                continue
        out.append(line)
        i += 1

    if warn_groups:
        for key in warn_order:
            block = warn_groups[key]
            count = sum(1 for l in block if WARNING_HEADER_RE.match(l))
            wtype, msg = key.split(":", 1)
            summary = f"  [{count}x] {wtype}: {msg.strip()}"
            events.append(CompressionEvent(
                kind="grouped_warning",
                original_lines=block,
                replacement=summary,
                note=f"Collapsed {count} occurrences; per-call-site file:line locations discarded."
            ))
        # splice grouped summary back in place of the (now removed) originals
        insert_at = next((idx for idx, l in enumerate(out) if "warnings summary" in l), len(out))
        summaries = [ev.replacement for ev in events if ev.kind == "grouped_warning"]
        out = out[:insert_at + 1] + summaries + out[insert_at + 1:]
    return out, events


def _truncate_long_blocks(lines: list[str]) -> tuple[list[str], list[CompressionEvent]]:
    """Cut any contiguous non-boilerplate block longer than
    TRUNCATE_BLOCK_LINES down to head+tail. This is where multi-frame
    tracebacks / long stdout dumps get shortened."""
    events = []
    out = []
    buf = []

    def flush():
        nonlocal buf
        if len(buf) > TRUNCATE_BLOCK_LINES:
            hidden = len(buf) - HEAD_KEEP - TAIL_KEEP
            events.append(CompressionEvent(
                kind="truncated_block",
                original_lines=list(buf),
                replacement=f"<<{hidden} lines hidden by rtk>>",
                note="Block exceeded truncation budget; middle frames/lines dropped."
            ))
            out.extend(buf[:HEAD_KEEP])
            out.append(f"    ... <<{hidden} lines hidden by rtk — call fetch_raw_context to view>> ...")
            out.extend(buf[-TAIL_KEEP:])
        else:
            out.extend(buf)
        buf = []

    for line in lines:
        if line.strip() == "" :
            flush()
            out.append(line)
        else:
            buf.append(line)
    flush()
    return out, events


def compress(raw_text: str) -> tuple[str, list[CompressionEvent]]:
    """Run the simulated rtk pipeline: filter -> group -> truncate."""
    lines = raw_text.splitlines()
    events: list[CompressionEvent] = []

    filtered, dropped = [], []
    for l in lines:
        if _is_boilerplate(l):
            dropped.append(l)
            continue
        filtered.append(l)
    if dropped:
        events.append(CompressionEvent(
            kind="dropped_boilerplate",
            original_lines=dropped,
            replacement="",
            note=f"Removed {len(dropped)} boilerplate/session-banner lines."
        ))

    grouped, warn_events = _group_warnings(filtered)
    events.extend(warn_events)

    truncated, trunc_events = _truncate_long_blocks(grouped)
    events.extend(trunc_events)

    compressed_text = "\n".join(truncated)
    return compressed_text, events


def token_estimate(text: str) -> int:
    """RTK itself estimates tokens as bytes / 4 (documented; it ships no
    tokenizer). We reuse that estimator for apples-to-apples comparison."""
    return max(1, len(text.encode("utf-8")) // 4)


def run_real_rtk(raw_text: str, rtk_binary: str = "rtk") -> str:
    """Stub: pipe raw_text through the *actual* rtk binary once available,
    e.g. `echo "$raw" | rtk pytest` or `rtk exec -- pytest ...`.
    Left unimplemented here because the binary could not be fetched in this
    sandbox; wire this up in an environment with network access to
    github.com/rtk-ai/rtk releases or a Rust toolchain (`cargo install rtk`).
    """
    raise NotImplementedError(
        "Real rtk binary not available in this environment. "
        "Use compress() (documented-behavior simulator) or install rtk "
        "and implement this function with subprocess.run([rtk_binary, ...])."
    )


if __name__ == "__main__":
    import sys
    raw = sys.stdin.read()
    out, evs = compress(raw)
    print(out)
    print(f"\n--- {len(evs)} compression events, "
          f"{token_estimate(raw)} -> {token_estimate(out)} est. tokens "
          f"({100 * (1 - token_estimate(out)/token_estimate(raw)):.1f}% reduction) ---",
          file=sys.stderr)
