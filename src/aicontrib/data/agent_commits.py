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
import os
import re
import shutil
import stat
import subprocess
import urllib.request
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


# ---- Licenses: only process what can be kept ----

def license_group(spdx: str | None, acfg: dict) -> str | None:
    """"permissive", "copyleft", or None when the license doesn't let the extracted rows be redistributed
    (no license, or custom / source-available terms: GitHub reports those as NOASSERTION)."""
    for group, ids in (acfg.get("licenses") or {}).items():
        if spdx in (ids or []):
            return group
    return None


def github_license(repo: str) -> str | None:
    """The repository's SPDX license id according to GitHub. Uses GITHUB_TOKEN / GH_TOKEN, else the `gh`
    CLI if installed, else anonymous requests (limited to 60 per hour)."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token and shutil.which("gh"):
        result = subprocess.run(["gh", "api", f"repos/{repo}", "--jq", '.license.spdx_id // ""'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip() or None
    headers = {"Accept": "application/vnd.github+json", **({"Authorization": f"Bearer {token}"} if token else {})}
    try:
        with urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/repos/{repo}", headers=headers),
                                    timeout=30) as response:
            return (json.load(response).get("license") or {}).get("spdx_id")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Can't read the license of {repo} from GitHub ({exc}). Anonymous requests are limited "
                           "to 60 per hour: set GITHUB_TOKEN or install the gh CLI, or re-run later.") from exc


def fetch_licenses(repos: list[str], cache_path: Path, log: Callable[[str], None] = print) -> dict[str, str | None]:
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    missing = [r for r in repos if r not in cache]
    if missing:
        log(f"Checking the license of {len(missing)} repositories on GitHub...")
    try:
        for repo in missing:
            cache[repo] = github_license(repo)
    finally:
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {r: cache[r] for r in repos}


# ---- Reusing earlier results ----

# The settings that decide which rows a repository gives. Saved with each result; a result built with other
# settings is redone.
_ROW_SETTINGS = ("ai_since", "human_until", "extensions", "exclude", "min_changed_lines", "max_files_per_commit",
                 "rows_per_class_per_repo", "human_pool_factor")


def _reusable(stats: dict, acfg: dict) -> bool:
    if stats.get("settings") == {k: acfg[k] for k in _ROW_SETTINGS}:
        return True
    # "No agent-signed commits" and "no early history" depend only on the dates, which the message records
    # (older results have no saved settings; ai_since was 2024-01-01 in all of them).
    ai_since = (stats.get("settings") or {}).get("ai_since", "2024-01-01")
    return ai_since == acfg["ai_since"] and stats.get("skipped") in (
        "no signed commits", f"no commits before {acfg['human_until']}")


def cap_repo_shares(rows_per_repo: dict[str, int], max_share: float | None) -> dict[str, int]:
    """Rows per class to keep per repository so that none exceeds max_share of the total (computed after
    the cap). With too few repositories for the cap to be reachable, nothing is capped."""
    if not max_share or len(rows_per_repo) * max_share < 1:
        return dict(rows_per_repo)
    limit = max(rows_per_repo.values())
    while True:  # the limit only decreases, so this converges
        new = max(1, int(max_share * sum(min(n, limit) for n in rows_per_repo.values())))
        if new >= limit:
            break
        limit = new
    return {repo: min(n, limit) for repo, n in rows_per_repo.items()}


# ---- Building the whole dataset ----

def select_repos(acfg: dict) -> list[dict]:
    """Index rows in the configured languages with enough signed commits, most signed commits first."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    parquet = hf_hub_download(acfg["index_repo"], "data/train-00000-of-00001.parquet", repo_type="dataset",
                              revision=acfg["index_revision"])
    df = pd.read_parquet(parquet, columns=["repo", "github_url", "language", "agent_attributed_commits"])
    df = df[df["language"].isin(acfg["repo_languages"]) & (df["agent_attributed_commits"] >= acfg["min_signed_commits"])]
    df = df.sort_values("agent_attributed_commits", ascending=False)
    return (df.head(acfg["max_repos"]) if acfg.get("max_repos") else df).to_dict("records")


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


def _remove_clone(path: Path) -> None:
    # Git marks its object files read-only, which makes a plain rmtree fail on Windows.
    def make_writable_and_retry(func, target, _):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    shutil.rmtree(path, onerror=make_writable_and_retry)


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
    licenses = fetch_licenses([e["repo"] for e in repos], out_dir / "licenses.json", log)
    if acfg.get("redistributable_only"):
        keep = [e for e in repos if license_group(licenses[e["repo"]], acfg)]
        log(f"{len(keep)} of them under a license that lets the result be redistributed; the others are skipped")
        repos = keep
    settings = {k: acfg[k] for k in _ROW_SETTINGS}

    def process(entry: dict) -> dict:
        name = entry["repo"]
        slug = name.replace("/", "__")
        done = per_repo_dir / f"{slug}.json"
        if done.exists():  # resume: each finished repo is saved on its own, with the settings it was built with
            saved = json.loads(done.read_text(encoding="utf-8"))["stats"]
            if _reusable(saved, acfg):
                return {**saved, "license": licenses[name]}
        try:
            _clone(entry["github_url"], clone_dir / slug)
            rows, stats = build_repo_rows(clone_dir / slug, name, acfg)
            stats.update(settings=settings, license=licenses[name])
            for row in rows:
                row["license"] = licenses[name]
            if not acfg.get("keep_clones"):
                _remove_clone(clone_dir / slug)  # its rows are all we need; big clones add up to many GB
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

    built = {s["repo"] for s in all_stats if "rows_per_class" in s}
    return write_splits(per_repo_dir, out_dir, acfg, [e["repo"] for e in repos if e["repo"] in built], all_stats, log,
                        licenses=licenses)


def write_splits(per_repo_dir: Path, out_dir: Path, acfg: dict, repo_names: list[str], all_stats: list[dict],
                 log: Callable[[str], None] = print, licenses: dict | None = None) -> dict:
    """The rows of repo_names into {train,validation,test}.jsonl, each repository capped to
    max_repo_share of the total (its best-matched pairs are kept)."""
    rows_by_repo = {}
    for name in repo_names:
        path = per_repo_dir / f"{name.replace('/', '__')}.json"
        if path.exists():
            rows_by_repo[name] = json.loads(path.read_text(encoding="utf-8"))["rows"]
    per_class = {name: len(rows) // 2 for name, rows in rows_by_repo.items()}
    keep = cap_repo_shares(per_class, acfg.get("max_repo_share"))
    capped = {name: f"{per_class[name]} -> {keep[name]}" for name in per_class if keep[name] < per_class[name]}
    if capped:
        log(f"Capped to {acfg['max_repo_share']:.0%} of the rows each: {capped}")
    counts: dict = {}
    files = {split: open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") for split in ("train", "validation", "test")}
    try:
        for name, rows in rows_by_repo.items():
            split = split_of(name, acfg)
            n = per_class[name]
            # Rows are the AI rows then their size-matched human rows, best-matched pairs first.
            for row in rows[: keep[name]] + rows[n : n + keep[name]]:
                if licenses and "license" not in row:  # rows of older builds
                    row = {**row, "license": licenses.get(name)}
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
