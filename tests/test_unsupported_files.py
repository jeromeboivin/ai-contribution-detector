"""Commits touching only non-code files (XML, JSON, SVG, proprietary binaries) must get
no prediction; mixed commits must be classified from their code files alone."""
import subprocess
from pathlib import Path

import numpy as np

from aicontrib.config import load_config
from aicontrib.diff.commit import CommitClassifier


def _commit(repo: Path, files: dict[str, bytes], message: str) -> str:
    for name, content in files.items():
        (repo / name).write_bytes(content)
        subprocess.run(["git", "add", name], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _classifier_with_stub_model() -> CommitClassifier:
    # Real filtering/diff logic; only the model's output is stubbed, so no checkpoint is needed.
    cfg = load_config()
    clf = object.__new__(CommitClassifier)
    clf.cfg = cfg
    clf.class_names = cfg["classes"]["names"]
    clf.extensions = cfg["commit_classification"]["supported_extensions"]
    clf.max_files = cfg["commit_classification"]["max_files_per_commit"]
    clf.added_files_only = False
    clf._probabilities = lambda texts: np.tile([0.2, 0.3, 0.5], (len(texts), 1))
    return clf


def test_non_code_changes_are_ignored(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)

    xml_only = _commit(repo, {"config.xml": b"<config><item>1</item></config>\n"}, "xml")
    binary_only = _commit(repo, {"asset.dat": bytes(range(256)) * 4}, "proprietary binary")
    data_only = _commit(repo, {"data.json": b'{"a": 1}\n', "logo.svg": b"<svg></svg>\n"}, "json+svg")
    mixed = _commit(
        repo,
        {"layout.xml": b"<layout/>\n", "App.tsx": b"export const App = () => <div>hi</div>;\n"},
        "xml + tsx",
    )
    esm = _commit(repo, {"build.mjs": b"export default function build() { return 1; }\n"}, "mjs")

    clf = _classifier_with_stub_model()

    for sha in (xml_only, binary_only, data_only):
        result = clf.classify(str(repo), sha)
        assert result["aggregate"] is None and result["files"] == []

    mixed_result = clf.classify(str(repo), mixed)
    assert [f["path"] for f in mixed_result["files"]] == ["App.tsx"]
    assert mixed_result["files"][0]["language"] == "javascript"
    assert mixed_result["aggregate"] is not None

    esm_result = clf.classify(str(repo), esm)
    assert [(f["path"], f["language"]) for f in esm_result["files"]] == [("build.mjs", "javascript")]


def test_bulk_commit_classifies_only_the_largest_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    files = {f"f{n}.py": b"x = 1\n" * n for n in range(1, 6)}  # f5.py has the most lines
    sha = _commit(repo, files, "bulk")

    clf = _classifier_with_stub_model()
    clf.max_files = 2
    result = clf.classify(str(repo), sha)
    assert sorted(f["path"] for f in result["files"]) == ["f4.py", "f5.py"]
    assert result["files_not_classified_over_cap"] == 3


def test_crlf_and_lone_carriage_returns_parse_and_are_normalized(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=repo, check=True)
    # A lone \r inside a line used to become an extra line under text-mode decoding,
    # making unidiff fail with "Hunk is longer than expected".
    sha = _commit(repo, {"win.py": b"a = 1\r\nb = 'x\ry'\r\nc = 2\r\n"}, "crlf")

    clf = _classifier_with_stub_model()
    seen = []
    clf._probabilities = lambda texts: seen.extend(texts) or np.tile([0.2, 0.3, 0.5], (len(texts), 1))

    result = clf.classify(str(repo), sha)
    assert [f["path"] for f in result["files"]] == ["win.py"]
    assert result["files"][0]["lines_changed"] == 3
    assert seen == ["a = 1\nb = 'x\ry'\nc = 2\n"]  # CRLF normalized, the lone \r preserved in place
