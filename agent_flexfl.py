"""
agent_flexfl.py
----------------
A Python/SWE-bench port of FlexFL's two-stage agent pipeline (Xu et al.,
"FlexFL: Flexible and Effective Fault Localization with Open-Source Large
Language Models"), scoped down for WP1's purpose: measuring how compressed
(rtk) vs raw terminal context changes localization accuracy.

FlexFL's original design (Java/Defects4J):
  Stage 1 (Agent4SR): space reduction via global search + non-LLM FL
  Stage 2 (Agent4LR): localization refinement, double-checks candidates

For WP1 we implement a *single-stage* agent that mirrors FlexFL's pipeline
mechanics exactly (task assignment -> ReAct-style reason+act loop with
function calls -> structured summarization + postprocessing), because WP1's
question is about the CONTEXT (raw vs compressed), not about space
reduction. WP2/WP3 will extend this into the full two-stage design plus the
adaptive proxy and zoom-in cache.

Function calls (Python analogue of FlexFL Table 2):
  get_paths()                      -> list source files in the repo
  get_symbols_of_file(path)        -> classes/functions defined in a file
  get_code_snippet(fqname)         -> source of a function/method/class
  find_symbol(name)                -> fuzzy search across the whole repo
  exit()                           -> stop gathering, ready to answer

The agent is given ONE of two textual "bug contexts" per run:
  - raw:  unedited pytest failure output (stdout+stderr)
  - rtk:  the same output passed through rtk_simulator.compress()
plus the SWE-bench problem_statement (bug report).

Output: ranked list of top-k suspicious functions, in FlexFL's
`path.Class.method(...)`-style FQN format, produced via the same
three-step pipeline (reason -> act -> summarize) and the same
Levenshtein-based postprocessing match against real repo symbols.
"""

from __future__ import annotations
import ast
import os
import re
import difflib
from dataclasses import dataclass, field
from typing import Callable, Optional

MAX_FUNCTION_CALLS = 10
TOP_K = 5


# ---------------------------------------------------------------------------
# Repository indexing (equivalent to FlexFL's "traverse all files, extract
# FQNs of all methods using a parser" — Section 3.2.2)
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    fqname: str          # e.g. "src/flask/blueprints.py:Blueprint.add_url_rule"
    kind: str            # "function" | "method" | "class"
    file: str
    lineno: int
    end_lineno: int
    source: str


class RepoIndex:
    """Parses a Python repo with `ast` and builds the FQN -> Symbol table
    FlexFL's get_/find_ function calls operate over."""

    def __init__(self, repo_root: str, include_globs=("**/*.py",), exclude_dirs=("tests", "test", ".git", "venv", "build")):
        self.repo_root = repo_root
        self.symbols: dict[str, Symbol] = {}
        self.files: list[str] = []
        self._index(exclude_dirs)

    def _index(self, exclude_dirs):
        for dirpath, dirnames, filenames in os.walk(self.repo_root):
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith(".")]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, self.repo_root)
                self.files.append(rel)
                try:
                    src = open(full, encoding="utf-8", errors="ignore").read()
                    tree = ast.parse(src)
                except (SyntaxError, UnicodeDecodeError):
                    continue
                lines = src.splitlines()
                self._index_node(tree, rel, lines, prefix="")

    def _index_node(self, node, relpath, lines, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                fq = f"{relpath}:{name}"
                snippet = "\n".join(lines[child.lineno - 1:child.end_lineno])
                self.symbols[fq] = Symbol(fq, "method" if prefix else "function",
                                           relpath, child.lineno, child.end_lineno, snippet)
                # do not recurse into nested defs' own nested defs beyond one level
            elif isinstance(child, ast.ClassDef):
                name = f"{prefix}{child.name}"
                fq = f"{relpath}:{name}"
                snippet = "\n".join(lines[child.lineno - 1:child.end_lineno])
                self.symbols[fq] = Symbol(fq, "class", relpath, child.lineno, child.end_lineno, snippet)
                self._index_node(child, relpath, lines, prefix=f"{name}.")

    def paths(self) -> list[str]:
        return sorted(set(os.path.dirname(f) or "." for f in self.files))

    def symbols_of_file(self, path: str) -> list[str]:
        return sorted(fq for fq in self.symbols if fq.startswith(f"{path}:"))

    def code_snippet(self, fqname: str) -> Optional[str]:
        sym = self.symbols.get(fqname)
        return sym.source if sym else None

    def find(self, query: str, limit: int = 5) -> list[str]:
        """FlexFL's Algorithm 1 postprocessing: exact-token match first,
        then Levenshtein distance < 5, then top-N closest."""
        candidates = list(self.symbols.keys())
        q_parts = set(re.split(r"[.\:/]", query))
        exact = [c for c in candidates if q_parts <= set(re.split(r"[.\:/]", c))]
        if exact:
            return exact[:limit]
        scored = sorted(candidates, key=lambda c: _levenshtein(query, c.split(":")[-1]))
        close = [c for c in scored if _levenshtein(query, c.split(":")[-1]) < 5]
        return (close or scored)[:limit]


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# Function-call tool layer (mirrors FlexFL Table 2 + Section 3.2.1 Step 2)
# ---------------------------------------------------------------------------

FUNCTION_CALL_RE = re.compile(r'^\s*([a-zA-Z_]+)\((.*)\)\s*$')


class ToolLayer:
    def __init__(self, index: RepoIndex):
        self.index = index
        self.call_log: list[tuple[str, str, str]] = []  # (name, arg, result)

    def call(self, name: str, arg: str) -> str:
        arg = arg.strip().strip('"\'')
        if name == "get_paths":
            result = "\n".join(self.index.paths()[:100])
        elif name == "get_symbols_of_file":
            syms = self.index.symbols_of_file(arg)
            result = "\n".join(s.split(":", 1)[1] for s in syms) or "(no symbols found; check path)"
        elif name == "get_code_snippet":
            matches = self.index.find(arg, limit=1)
            if not matches:
                result = f"(no symbol matched '{arg}')"
            else:
                result = self.index.code_snippet(matches[0]) or "(source unavailable)"
        elif name == "find_symbol":
            matches = self.index.find(arg, limit=5)
            result = "\n".join(matches) or "(no match)"
        elif name == "exit":
            result = "(exiting function-call loop)"
        else:
            result = f"Please call functions in the right format 'FunctionName(Argument)'."
        self.call_log.append((name, arg, result))
        return result


FUNCTION_CALL_DESCRIPTIONS = """\
get_paths()                    - list top-level source directories in the repo
get_symbols_of_file(path)      - list classes/functions defined in a file
get_code_snippet(fqname_or_name) - get source code of a function/method/class
find_symbol(name)              - fuzzy-search a class/function name across the repo
exit()                         - stop gathering info, you are ready to answer
"""


# ---------------------------------------------------------------------------
# Three-step pipeline (Section 3.2.1): task assignment -> react loop -> summarize
# ---------------------------------------------------------------------------

LLMCallFn = Callable[[str, str], str]  # (system_prompt, user_prompt) -> completion text


@dataclass
class LocalizationResult:
    ranked_fqns: list[str]
    raw_llm_transcript: list[str] = field(default_factory=list)
    num_function_calls: int = 0
    postprocessing_matches: dict = field(default_factory=dict)


def build_system_prompt() -> str:
    return f"""You are a debugging assistant for a Python software project.
You will be given a bug report and/or terminal output from a failing test run,
plus tools (function calls) to explore the source code of the repository.
Your task is to identify the top-{TOP_K} most likely culprit functions/methods
responsible for the bug.

Function calls you can use:
{FUNCTION_CALL_DESCRIPTIONS}
You have {MAX_FUNCTION_CALLS} chances to call a function. First reason step by
step about where the bug likely is, then call functions one at a time in the
exact format `FunctionName(Argument)` on its own line, with no other text.
When confident, call exit().
"""


def build_task_prompt(problem_statement: str, terminal_context: str) -> str:
    parts = []
    if problem_statement:
        parts.append(f"The bug report is as follows:\n```\n{problem_statement.strip()}\n```")
    if terminal_context:
        parts.append(f"The terminal output from the failing test run is as follows:\n```\n{terminal_context.strip()}\n```")
    return "\n\n".join(parts)


def run_agent(problem_statement: str, terminal_context: str, index: RepoIndex,
              llm_call: LLMCallFn) -> LocalizationResult:
    """FlexFL's Step1 (task assignment) -> Step2 (react+function-calls loop)
    -> Step3 (summarize + postprocess), single-stage version for WP1."""
    system_prompt = build_system_prompt()
    task_prompt = build_task_prompt(problem_statement, terminal_context)
    tools = ToolLayer(index)
    transcript = [f"[SYSTEM]\n{system_prompt}", f"[USER]\n{task_prompt}"]

    context = task_prompt
    calls_made = 0
    while calls_made < MAX_FUNCTION_CALLS:
        step_prompt = (context +
                       "\n\nNow reason and plan how to locate the buggy function(s), "
                       "then call one function in the format 'FunctionName(Argument)' "
                       "on a single line with no other text.")
        response = llm_call(system_prompt, step_prompt)
        transcript.append(f"[ASSISTANT]\n{response}")

        call_line = None
        for line in response.splitlines()[::-1]:
            if FUNCTION_CALL_RE.match(line.strip()):
                call_line = line.strip()
                break
        if call_line is None:
            context += f"\n\n{response}\n\nPlease call functions in the right format 'FunctionName(Argument)'."
            continue

        m = FUNCTION_CALL_RE.match(call_line)
        name, arg = m.group(1), m.group(2)
        if name == "exit":
            calls_made += 1
            break
        result = tools.call(name, arg)
        context += f"\n\n{response}\n\n[FUNCTION RESULT for {name}({arg})]\n{result}"
        transcript.append(f"[TOOL {name}({arg})]\n{result}")
        calls_made += 1

    summarize_prompt = (context +
                         f"\n\nBased on the available information, provide the top-{TOP_K} "
                         "most likely culprit functions. Respond ONLY in the format:\n"
                         "Top_1: path/to/file.py:ClassName.method_name\nTop_2: ...\n")
    final = llm_call(system_prompt, summarize_prompt)
    transcript.append(f"[ASSISTANT-SUMMARY]\n{final}")

    raw_names = re.findall(r"Top_\d+:\s*(.+)", final)
    postproc = {}
    ranked = []
    for raw_name in raw_names[:TOP_K]:
        matches = index.find(raw_name.strip(), limit=1)
        matched = matches[0] if matches else None
        postproc[raw_name.strip()] = matched
        if matched:
            ranked.append(matched)

    return LocalizationResult(
        ranked_fqns=ranked,
        raw_llm_transcript=transcript,
        num_function_calls=calls_made,
        postprocessing_matches=postproc,
    )
