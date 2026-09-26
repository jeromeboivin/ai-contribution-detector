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


def test_remove_clone_handles_read_only_git_objects(tmp_path):
    clone = tmp_path / "clone"
    (clone / ".git" / "objects" / "pack").mkdir(parents=True)
    pack = clone / ".git" / "objects" / "pack" / "pack-1.pack"
    pack.write_bytes(b"x")
    pack.chmod(0o444)  # as git leaves it; blocks deletion on Windows
    ac._remove_clone(clone)
    assert not clone.exists()


def _write_gz(path, rows):
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for code, label in rows:
            f.write(json.dumps({"code": code, "label": label, "language": "TypeScript"}) + "\n")


def test_archives_are_read_when_there_is_no_local_build(tmp_path):
    _write_gz(tmp_path / "archive" / "train.jsonl.gz", [("mit-h", "human"), ("mit-a", "ai")])
    _write_gz(tmp_path / "archive" / "copyleft" / "train.jsonl.gz", [("gpl-h", "human")])
    source = {"name": "agent_commits", "path": str(tmp_path / "no-local-build"),
              "archive": [str(tmp_path / "archive"), str(tmp_path / "archive" / "copyleft")]}

    assert [code for code, _, _ in iter_agent_commits(source, "train")] == ["mit-h", "mit-a", "gpl-h"]
    assert list(iter_agent_commits(source, "validation")) == []  # no file for that split


def test_a_local_build_takes_precedence_over_the_archives(tmp_path):
    _write_gz(tmp_path / "archive" / "train.jsonl.gz", [("archived", "ai")])
    (tmp_path / "local").mkdir()
    (tmp_path / "local" / "train.jsonl").write_text(json.dumps({"code": "local", "label": "ai", "language": "ts"}) + "\n")
    source = {"name": "agent_commits", "path": str(tmp_path / "local"), "archive": [str(tmp_path / "archive")]}

    assert list(iter_agent_commits(source, "train")) == [("local", "ai", "TypeScript")]


def test_shipped_archives_are_readable_and_licensed():
    from aicontrib.config import REPO_ROOT, load_config

    source = next(s for s in load_config()["dataset"]["sources"] if s["name"] == "agent_commits")
    rows = [json.loads(line) for d in source["archive"] for line in
            __import__("gzip").open(REPO_ROOT / d / "train.jsonl.gz", "rt", encoding="utf-8")]
    assert rows and {r["label"] for r in rows} == {"human", "ai"}
    for r in rows:
        assert (REPO_ROOT / "datasets/agent_commits" / ("copyleft" if r["license"] in ("GPL-3.0", "AGPL-3.0") else "")
                / "LICENSES" / f"{r['repo'].replace('/', '__')}.txt").exists()
