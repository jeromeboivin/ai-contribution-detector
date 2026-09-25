"""Real-repository training data: commits signed by coding agents vs human commits from before them.

Benchmark snippets don't look like real repository code, and no benchmark has TypeScript. This
builds labelled data from real repositories instead, one row per file changed in a commit, rebuilt
exactly as `classify-commit` sees it (the post-image of each hunk, see aicontrib.diff.commit):

- ai: commits carrying a self-declared coding-agent signature -- a `Co-Authored-By: Claude` (Copilot,
  Cursor, Codex...) trailer, a tool-written footer, or an agent bot identity. The rules are ported
  from qmmit-cli (MIT, https://github.com/pandey019/qmmit-cli, src/signatures.js).
- human: commits from before `human_until` (default mid-2021, before GitHub Copilot existed), minus
  CI/dependency bots.

Repositories come from the qmmit agent-commit index (Hugging Face dataset
balrampandey/qmmit-open-source-agent-commit-index): 2,000 popular GitHub repos with their share of
agent-signed commits. The index itself has no code; each repo is cloned (blobless: history without
file contents, which are fetched per selected commit).

Each repository contributes the same number of human and AI rows, so its identity can't predict the
label, and each AI row is paired with a human row of similar size (agents write bigger changes), so
size can't either; repositories without both are skipped. Rows are split into train/validation/test by
repository, so the test split measures unseen repositories. What remains confounded is time: the
human rows are ~2021 code and the AI rows 2025-2026 code.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable

from unidiff import PatchSet

from aicontrib.config import REPO_ROOT
from aicontrib.diff.commit import _post_image_text

# ---- Signatures: ported from qmmit-cli src/signatures.js (MIT). Literal, tool-written markers only. ----
# kind: trailer/footer -> commit body, identity -> "Name <email>" of author and committer, subject -> subject.
AGENT_SIGNATURES = {
    "claude-code": [("trailer", r"^\s*Co-Authored-By:\s*Claude\b"), ("footer", r"Generated with .{0,20}Claude Code"),
                    ("identity", r"^claude\[bot\]\s*<"), ("identity", r"<[^>]*@anthropic\.com>")],
    "cursor": [("trailer", r"^\s*Co-Authored-By:\s*Cursor\b"), ("identity", r"<cursoragent@cursor\.com>"),
               ("identity", r"^Cursor Agent\s*<"), ("identity", r"\bcursor\[bot\]")],
    "copilot": [("trailer", r"^\s*Co-Authored-By:\s*Copilot\b"), ("identity", r"^copilot(-swe-agent)?\[bot\]\s*<")],
    "devin": [("identity", r"devin-ai-integration\[bot\]"), ("identity", r"^Devin AI\s*<"),
              ("trailer", r"^\s*Co-Authored-By:\s*Devin\b")],
    "codex": [("footer", r"https://chatgpt\.com/codex/"), ("trailer", r"^\s*Co-Authored-By:\s*Codex\b"),
              ("identity", r"^chatgpt-codex-connector\[bot\]")],
    "aider": [("subject", r"\(aider\)\s*$")],
    "gemini-cli": [("trailer", r"^\s*Co-Authored-By:\s*Gemini\b"), ("identity", r"^gemini-code-assist\[bot\]")],
    "codegen-generic": [("trailer", r"^\s*Co-Authored-By:\s*(OpenHands|Sweep|Codegen|Jules)\b"),
                        ("identity", r"^(openhands|sweep-ai|jules)\[bot\]")],
}
_COMPILED = {agent: [(kind, re.compile(rx, re.I | re.M)) for kind, rx in rules] for agent, rules in AGENT_SIGNATURES.items()}
# CI / dependency / release automation: machine-made, but not agent-authored source code.
EXCLUDED_BOTS = [re.compile(rx, re.I | re.M) for rx in (
    r"\[bot\]\s*<", r"\[bot\]@users\.noreply\.github\.com", r"^Vercel Release Bot\s*<", r"release[- ]bot\s*<",
    r"\bsnyk-bot\b", r"\bsemantic-release-bot\b", r"\bgreenkeeper\b", r"\bweblate\b", r"\bcrowdin[- ]?bot\b",
    r"\btravis[- ]ci\b", r"\bnetlify\[?bot\]?\b",
)]
_LANGUAGE_BY_EXTENSION = {".ts": "TypeScript", ".tsx": "TypeScript", ".mts": "TypeScript", ".cts": "TypeScript",
                          ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript"}
_IMPORTS_REACT = re.compile(r"""(from\s+|require\(\s*)['"]react(-dom)?(/[^'"]*)?['"]""")


@dataclass
class Commit:
    sha: str
    author: str      # "Name <email>"
    committer: str
    date: str        # ISO 8601
    subject: str
    body: str
    paths: list[str]  # changed files (no renames)

    @property
    def identity(self) -> str:
        return f"{self.author}\n{self.committer}"


def detect_agents(commit: Commit) -> list[str]:
    hits = []
    for agent, rules in _COMPILED.items():
        for kind, rx in rules:
            text = commit.identity if kind == "identity" else commit.subject if kind == "subject" else commit.body
            if rx.search(text):
                hits.append(agent)
                break
    return hits


def is_excluded_bot(commit: Commit) -> bool:
    return any(rx.search(commit.identity) for rx in EXCLUDED_BOTS)


def is_autonomous(commit: Commit) -> bool:
    """A bot identity authored the commit (vs a person, with the agent as co-author: "assisted")."""
    return bool(re.search(r"\[bot\]|devin-ai-integration|^Devin AI\s*<", commit.author, re.I))


# ---- Reading a repository ----

def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), "-c", "core.quotePath=false", *args], capture_output=True, check=True)
    return result.stdout.decode("utf-8", errors="replace")


def list_commits(repo: Path, since: str | None = None, until: str | None = None) -> list[Commit]:
    """Non-merge commits with their changed paths. Needs only commits and trees, never file contents,
    so it's fast in a blobless clone."""
    fmt = "%x1e%H%x1f%an <%ae>%x1f%cn <%ce>%x1f%cI%x1f%s%x1f%b%x1f"
    args = ["log", "--no-merges", "--no-renames", "--raw", "--no-abbrev", f"--format={fmt}"]
    args += [f"--since={since}"] if since else []
    args += [f"--until={until}"] if until else []
    commits = []
    for record in _git(repo, *args).split("\x1e")[1:]:
        *fields, raw = record.split("\x1f")
        if len(fields) != 6:
            continue
        sha, author, committer, date, subject, body = fields
        paths = [line.split("\t", 1)[1] for line in raw.splitlines() if line.startswith(":") and "\t" in line]
        commits.append(Commit(sha, author, committer, date, subject, body, paths))
    return commits


def _wanted(path: str, extensions: set[str], exclude: list[str]) -> bool:
    return Path(path).suffix.lower() in extensions and not any(fnmatch(path, pat) for pat in exclude)


def _file_rows(repo: Path, commit: Commit, paths: list[str], acfg: dict) -> list[dict]:
    """One row per changed file with enough changed lines, largest changes first."""
    diff = _git(repo, "show", "--no-renames", "--unified=3", "--format=", "--no-color", "--no-ext-diff",
                commit.sha, "--", *paths)
    rows = []
    for patched in PatchSet(diff):
        if patched.is_removed_file or patched.is_binary_file:
            continue
        text, changed = _post_image_text(patched)
        if changed < acfg["min_changed_lines"] or not text.strip():
            continue
        suffix = Path(patched.path).suffix.lower()
        rows.append({"code": text, "path": patched.path, "language": _LANGUAGE_BY_EXTENSION.get(suffix, suffix),
                     "react": suffix in (".tsx", ".jsx") or bool(_IMPORTS_REACT.search(text)),
                     "added": patched.is_added_file, "lines_changed": changed})
    rows.sort(key=lambda r: r["lines_changed"], reverse=True)
    return rows[: acfg["max_files_per_commit"]]


def _collect(repo: Path, commits: list[Commit], acfg: dict, limit: int, label: str, repo_name: str,
             seed: int) -> list[dict]:
    extensions, exclude = set(acfg["extensions"]), acfg["exclude"]
    candidates = [c for c in commits if any(_wanted(p, extensions, exclude) for p in c.paths)]
    random.Random(seed).shuffle(candidates)  # spread over the whole period, not just the latest commits
    rows = []
    for commit in candidates:
        if len(rows) >= limit:
            break
        agents = detect_agents(commit) if label == "ai" else []
        for row in _file_rows(repo, commit, [p for p in commit.paths if _wanted(p, extensions, exclude)], acfg):
            row.update(label=label, repo=repo_name, commit=commit.sha, date=commit.date, agents=agents,
                       mode=("autonomous" if is_autonomous(commit) else "assisted") if agents else None)
            rows.append(row)
    return rows[:limit]


def match_sizes(ai_rows: list[dict], human_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pairs of (AI row, human row) of similar size -- closest pairs first, each row used once -- so the
    two classes have the same distribution of changed lines."""
    size = lambda r: math.log(r["lines_changed"])  # noqa: E731 - relative, not absolute, difference
    pairs = sorted((abs(size(a) - size(h)), i, j) for i, a in enumerate(ai_rows) for j, h in enumerate(human_rows))
    used_ai, used_human, matched = set(), set(), []
    for _, i, j in pairs:
        if i not in used_ai and j not in used_human:
            used_ai.add(i)
            used_human.add(j)
            matched.append((i, j))
    return [ai_rows[i] for i, _ in matched], [human_rows[j] for _, j in matched]


def build_repo_rows(repo: Path, repo_name: str, acfg: dict) -> tuple[list[dict], dict]:
    """(rows, stats) for one cloned repository: equal numbers of size-matched AI and human rows, or none."""
    seed = int(hashlib.sha1(repo_name.encode()).hexdigest()[:8], 16)
    signed = [c for c in list_commits(repo, since=acfg["ai_since"]) if detect_agents(c)]
    human = [c for c in list_commits(repo, until=acfg["human_until"]) if not is_excluded_bot(c) and not detect_agents(c)]
    stats = {"repo": repo_name, "signed_commits": len(signed), "human_commits": len(human)}
    if not signed or not human:
        return [], {**stats, "skipped": "no signed commits" if not signed else f"no commits before {acfg['human_until']}"}
    ai_rows = _collect(repo, signed, acfg, acfg["rows_per_class_per_repo"], "ai", repo_name, seed)
    # More human candidates than needed, to find one of similar size for each AI row.
    human_pool = _collect(repo, human, acfg, len(ai_rows) * acfg["human_pool_factor"], "human", repo_name, seed)
    ai_rows, human_rows = match_sizes(ai_rows, human_pool)
    if not ai_rows:
        return [], {**stats, "skipped": f"no {acfg['extensions']} changes of {acfg['min_changed_lines']}+ lines on one side"}
    return ai_rows + human_rows, {**stats, "rows_per_class": len(ai_rows)}


def split_of(repo_name: str, acfg: dict) -> str:
    """Deterministic, by repository: the same repo always lands in the same split."""
    bucket = int(hashlib.sha1(repo_name.encode()).hexdigest()[:8], 16) % 100
    if bucket < acfg["split_percent"]["test"]:
        return "test"
    if bucket < acfg["split_percent"]["test"] + acfg["split_percent"]["validation"]:
        return "validation"
    return "train"


# ---- Building the whole dataset ----

def select_repos(acfg: dict) -> list[dict]:
    """Index rows in the configured languages with enough signed commits, most signed commits first."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    parquet = hf_hub_download(acfg["index_repo"], "data/train-00000-of-00001.parquet", repo_type="dataset",
                              revision=acfg["index_revision"])
    df = pd.read_parquet(parquet, columns=["repo", "github_url", "language", "agent_attributed_commits"])
    df = df[df["language"].isin(acfg["repo_languages"]) & (df["agent_attributed_commits"] >= acfg["min_signed_commits"])]
    df = df.sort_values("agent_attributed_commits", ascending=False).head(acfg["max_repos"])
    return df.to_dict("records")


def _clone(url: str, dest: Path) -> None:
    if (dest / ".git").exists() or (dest / "HEAD").exists():
        return
    tmp = dest.with_name(dest.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    # Blobless and without a working tree: history and trees only; file contents come per selected commit.
    subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", "--single-branch", url, str(tmp)],
                   capture_output=True, check=True)
    tmp.rename(dest)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def build_agent_commits(cfg: dict, max_repos: int | None = None, log: Callable[[str], None] = print) -> dict:
    acfg = dict(cfg["agent_commits"])
    if max_repos:
        acfg["max_repos"] = max_repos
    clone_dir, out_dir = _resolve(acfg["clone_dir"]), _resolve(acfg["out_dir"])
    per_repo_dir = out_dir / "repos"
    per_repo_dir.mkdir(parents=True, exist_ok=True)
    repos = select_repos(acfg)
    log(f"{len(repos)} repositories selected from the index ({', '.join(acfg['repo_languages'])}, "
        f"{acfg['min_signed_commits']}+ signed commits)")

    def process(entry: dict) -> dict:
        name = entry["repo"]
        slug = name.replace("/", "__")
        done = per_repo_dir / f"{slug}.json"
        if done.exists():  # resume: each finished repo is saved on its own
            return json.loads(done.read_text(encoding="utf-8"))["stats"]
        try:
            _clone(entry["github_url"], clone_dir / slug)
            rows, stats = build_repo_rows(clone_dir / slug, name, acfg)
        except subprocess.CalledProcessError as exc:  # not saved: retried on the next run
            return {"repo": name, "error": exc.stderr.decode("utf-8", "replace").strip()[-300:]}
        except Exception as exc:  # noqa: BLE001 - one odd repository mustn't stop the other workers
            return {"repo": name, "error": f"{type(exc).__name__}: {exc}"[:300]}
        tmp = done.with_suffix(".tmp")
        tmp.write_text(json.dumps({"stats": stats, "rows": rows}), encoding="utf-8")
        tmp.replace(done)
        return stats

    all_stats = []
    with ThreadPoolExecutor(max_workers=acfg["workers"]) as pool:
        futures = {pool.submit(process, entry): entry["repo"] for entry in repos}
        for i, future in enumerate(as_completed(futures), 1):
            stats = future.result()
            all_stats.append(stats)
            outcome = (f"{stats['rows_per_class']} rows per class" if "rows_per_class" in stats
                       else f"skipped ({stats.get('skipped') or 'error: ' + stats.get('error', '')})")
            log(f"[{i}/{len(repos)}] {stats['repo']}: {outcome}")

    return write_splits(per_repo_dir, out_dir, acfg, [e["repo"] for e in repos], all_stats, log)


def write_splits(per_repo_dir: Path, out_dir: Path, acfg: dict, repo_names: list[str], all_stats: list[dict],
                 log: Callable[[str], None] = print) -> dict:
    counts: dict = {}
    files = {split: open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") for split in ("train", "validation", "test")}
    try:
        for name in repo_names:
            path = per_repo_dir / f"{name.replace('/', '__')}.json"
            if not path.exists():
                continue
            split = split_of(name, acfg)
            for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
                files[split].write(json.dumps({**row, "split": split}) + "\n")
                key = (split, row["label"])
                counts[key] = counts.get(key, 0) + 1
                counts[(split, "react")] = counts.get((split, "react"), 0) + row["react"]
                counts[(split, "repos")] = counts.get((split, "repos"), set()) | {name}
    finally:
        for f in files.values():
            f.close()
    summary = {split: {"repos": len(counts.get((split, "repos"), set())), "human": counts.get((split, "human"), 0),
                       "ai": counts.get((split, "ai"), 0), "react": counts.get((split, "react"), 0)}
               for split in ("train", "validation", "test")}
    (out_dir / "summary.json").write_text(json.dumps({"splits": summary, "repos": all_stats}, indent=2) + "\n",
                                          encoding="utf-8")
    for split, s in summary.items():
        log(f"[{split}] {s['repos']} repos: {s['human']} human rows, {s['ai']} AI rows ({s['react']} React)")
    log(f"Written to {out_dir}")
    return summary
