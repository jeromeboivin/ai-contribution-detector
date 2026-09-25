import json
import os
import subprocess

import pytest

from aicontrib.config import load_config
from aicontrib.data import agent_commits as ac
from aicontrib.data.sources import iter_agent_commits

HUMAN = ("Ada Dev", "ada@example.com")
CLAUDE_TRAILER = "\n\nCo-Authored-By: Claude <noreply@anthropic.com>"


def _commit(repo, date, message, files, author=HUMAN):
    env = {"GIT_AUTHOR_DATE": f"{date}T12:00:00", "GIT_COMMITTER_DATE": f"{date}T12:00:00",
           "GIT_AUTHOR_NAME": author[0], "GIT_AUTHOR_EMAIL": author[1],
           "GIT_COMMITTER_NAME": author[0], "GIT_COMMITTER_EMAIL": author[1], "PATH": os.environ["PATH"]}
    for name, content in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(content)
        subprocess.run(["git", "add", name], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True, env=env)


def _ts(name, n):
    return "".join(f"export const {name}{i} = (x: number): number => x * {i};\n" for i in range(n))


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _commit(repo, "2019-03-01", "human 1", {"src/a.ts": _ts("a", 8), "src/types.d.ts": _ts("t", 8)})
    _commit(repo, "2020-05-01", "human 2", {"src/b.ts": _ts("b", 8), "vendor/lib.ts": _ts("v", 8)})
    _commit(repo, "2020-06-01", "bump deps", {"src/c.ts": _ts("c", 8)}, author=("dependabot[bot]", "x@users.noreply.github.com"))
    _commit(repo, "2020-07-01", "tiny", {"src/a.ts": _ts("a", 8).replace("x * 1", "x * 11")})  # 2 changed lines
    _commit(repo, "2026-03-01", "feat: add view" + CLAUDE_TRAILER,
            {"src/View.tsx": "import React from 'react';\n" + _ts("v", 8)})
    _commit(repo, "2026-04-01", "fix: guard input", {"src/d.ts": _ts("d", 8)},
            author=("copilot-swe-agent[bot]", "198982749+Copilot@users.noreply.github.com"))
    _commit(repo, "2026-05-01", "unsigned 2026 change", {"src/e.ts": _ts("e", 8)})  # neither side
    _commit(repo, "2026-05-02", "docs only" + CLAUDE_TRAILER, {"README.md": "hello\n" * 10})
    return repo


def _acfg():
    return {**load_config()["agent_commits"], "rows_per_class_per_repo": 10, "min_changed_lines": 5}


def test_signature_rules():
    c = lambda author="Ada <a@x.com>", subject="s", body="": ac.Commit("0", author, author, "", subject, body, [])  # noqa: E731
    assert ac.detect_agents(c(body="x\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")) == ["claude-code"]
    assert ac.detect_agents(c(body="🤖 Generated with [Claude Code](https://claude.com/claude-code)")) == ["claude-code"]
    assert ac.detect_agents(c(subject="refactor parser (aider)")) == ["aider"]
    bot = c(author="devin-ai-integration[bot] <158243242+devin-ai-integration[bot]@users.noreply.github.com>")
    assert ac.detect_agents(bot) == ["devin"] and ac.is_autonomous(bot)
    assert ac.detect_agents(c(body="Co-authored-by: Jane <jane@x.com>")) == []  # a person, not an agent
    assert ac.is_excluded_bot(c(author="renovate[bot] <29139614+renovate[bot]@users.noreply.github.com>"))


def test_build_repo_rows_pairs_signed_commits_with_early_human_ones(repo):
    rows, stats = ac.build_repo_rows(repo, "owner/repo", _acfg())

    assert stats["signed_commits"] == 3 and stats["rows_per_class"] == 2
    ai = sorted((r for r in rows if r["label"] == "ai"), key=lambda r: r["path"])
    human = sorted((r for r in rows if r["label"] == "human"), key=lambda r: r["path"])
    assert [r["path"] for r in ai] == ["src/View.tsx", "src/d.ts"]  # not README.md
    assert [r["path"] for r in human] == ["src/a.ts", "src/b.ts"]  # no .d.ts, vendor/, bot, or 2-line change
    view, d = ai
    assert view["react"] and view["agents"] == ["claude-code"] and view["mode"] == "assisted" and view["added"]
    assert d["agents"] == ["copilot"] and d["mode"] == "autonomous" and not d["react"]
    assert all(r["language"] == "TypeScript" and r["lines_changed"] >= 5 for r in rows)
    assert "export const a0" in human[0]["code"]


def test_repository_without_early_history_is_skipped(tmp_path):
    repo = tmp_path / "new"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _commit(repo, "2026-03-01", "feat" + CLAUDE_TRAILER, {"src/a.ts": _ts("a", 8)})
    rows, stats = ac.build_repo_rows(repo, "owner/new", _acfg())
    assert rows == [] and "before 2021-06-01" in stats["skipped"]


def test_splits_are_by_repository_and_readable_by_prepare(tmp_path, repo):
    acfg = _acfg()
    rows, stats = ac.build_repo_rows(repo, "owner/repo", acfg)
    per_repo = tmp_path / "out" / "repos"
    per_repo.mkdir(parents=True)
    (per_repo / "owner__repo.json").write_text(json.dumps({"stats": stats, "rows": rows}))

    summary = ac.write_splits(per_repo, tmp_path / "out", acfg, ["owner/repo"], [stats], log=lambda _: None)

    split = ac.split_of("owner/repo", acfg)
    assert summary[split] == {"repos": 1, "human": 2, "ai": 2, "react": 1}
    read = list(iter_agent_commits({"name": "agent_commits", "path": str(tmp_path / "out")}, split))
    assert sorted(label for _, label, _ in read) == ["ai", "ai", "human", "human"]
    assert {lang for _, _, lang in read} == {"TypeScript"}
    assert list(iter_agent_commits({"name": "agent_commits", "path": str(tmp_path / "missing")}, "train")) == []


def test_match_sizes_pairs_rows_of_similar_size():
    ai = [{"id": "a-big", "lines_changed": 100}, {"id": "a-small", "lines_changed": 6}]
    human = [{"id": f"h{n}", "lines_changed": n} for n in (5, 7, 30, 90, 400)]

    ai_rows, human_rows = ac.match_sizes(ai, human)

    assert {(a["id"], h["id"]) for a, h in zip(ai_rows, human_rows)} == {("a-big", "h90"), ("a-small", "h7")}
    assert ac.match_sizes(ai, [human[0]]) == ([ai[1]], [human[0]])  # fewer humans: the closest AI row is kept
