"""Commit-level classification (MVP heuristic).

The MLP was trained on whole human/AI code *snippets*, not diffs -- there is
no diff-level ground-truth dataset (see README). As an approximation, we
reconstruct the post-change text of each hunk (context + added lines) per
file, classify each changed file, and aggregate to one commit-level
distribution weighted by lines changed per file. Expect this to be less
accurate than the snippet-level test metrics.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from unidiff import PatchSet

from aicontrib.config import load_config
from aicontrib.device import get_device
from aicontrib.features.embed import CodeEmbedder
from aicontrib.model.evaluate import load_checkpoint


@dataclass
class FileResult:
    path: str
    language: str
    lines_changed: int
    probabilities: dict[str, float]


def run_git(repo_path: str, *args: str) -> str:
    # Capture bytes and decode ourselves: text mode would apply universal-newline translation,
    # turning a lone \r inside file content into an extra line that breaks diff hunk counts.
    # Explicit utf-8 because Windows would otherwise use the locale codepage (cp1252).
    result = subprocess.run(["git", "-C", repo_path, *args], capture_output=True, check=True)
    return result.stdout.decode("utf-8", errors="replace")


def _get_unified_diff(repo_path: str, sha: str) -> str:
    # --no-color / --no-ext-diff: a user's color.ui=always or external diff driver would
    # otherwise change the output format. --diff-merges=first-parent: merges default to a
    # combined `diff --cc`, which unidiff can't parse.
    return run_git(
        repo_path, "show", "--unified=3", "--format=", "--no-color", "--no-ext-diff",
        "--diff-merges=first-parent", sha,
    )


def _post_image_text(patched_file) -> tuple[str, int]:
    lines = []
    changed = 0
    for hunk in patched_file:
        for line in hunk:
            if line.is_removed:
                changed += 1
                continue
            lines.append(line.value)
            if line.is_added:
                changed += 1
    # CRLF -> LF so files committed with Windows line endings look like the (LF) training data.
    return "".join(lines).replace("\r\n", "\n"), changed


class CommitClassifier:
    """Loads the encoder and MLP once, then classifies any number of commits --
    reloading the 110M-param encoder per commit would dominate a full-history scan."""

    def __init__(self, config_path: str | None = None, added_files_only: bool = False):
        self.cfg = load_config(config_path) if config_path else load_config()
        self.device = get_device()
        self.embedder = CodeEmbedder(self.cfg)
        self.model = load_checkpoint(Path(self.cfg["paths"]["models_dir"]) / "mlp_classifier.pt", self.device,
                                     representation=self.embedder.representation_id())
        self.class_names = self.cfg["classes"]["names"]
        self.extensions = self.cfg["commit_classification"]["supported_extensions"]
        self.max_files = self.cfg["commit_classification"].get("max_files_per_commit")
        # Newly added files are whole files, like the training snippets; edits to existing files are
        # stitched-together hunks. Scoring only added files isolates that mismatch (see evaluate-repo).
        self.added_files_only = added_files_only

    @torch.no_grad()
    def _probabilities(self, texts: list[str]) -> np.ndarray:
        batch_size = self.cfg["embedding"]["batch_size"]
        chunks = []
        for i in range(0, len(texts), batch_size):
            x = torch.tensor(self.embedder.embed_batch(texts[i : i + batch_size]), dtype=torch.float32)
            chunks.append(torch.softmax(self.model(x.to(self.device)), dim=-1).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def classify(self, repo_path: str, sha: str) -> dict:
        patch = PatchSet(_get_unified_diff(repo_path, sha))

        candidates = []
        for patched_file in patch:
            if patched_file.is_removed_file or patched_file.is_binary_file:
                continue
            if self.added_files_only and not patched_file.is_added_file:
                continue
            language = self.extensions.get(Path(patched_file.path).suffix.lower())
            if language is None:
                continue
            text, lines_changed = _post_image_text(patched_file)
            if text.strip() and lines_changed:
                candidates.append((patched_file.path, language, lines_changed, text))

        if not candidates:
            kind = "newly added supported-language files" if self.added_files_only else "supported-language files with changes"
            return {"commit": sha, "files": [], "aggregate": None, "note": f"no {kind} found"}

        files_over_cap = 0
        if self.max_files and len(candidates) > self.max_files:
            candidates.sort(key=lambda c: c[2], reverse=True)
            files_over_cap = len(candidates) - self.max_files
            candidates = candidates[: self.max_files]

        probs = self._probabilities([c[3] for c in candidates])
        file_results = [
            FileResult(path=path, language=language, lines_changed=lines_changed,
                       probabilities=dict(zip(self.class_names, p.tolist())))
            for (path, language, lines_changed, _), p in zip(candidates, probs)
        ]

        weights = np.array([fr.lines_changed for fr in file_results], dtype=np.float64)
        aggregate = (weights[:, None] / weights.sum() * probs).sum(axis=0)

        return {
            "commit": sha,
            "files": [
                {"path": fr.path, "language": fr.language, "lines_changed": fr.lines_changed, "probabilities": fr.probabilities}
                for fr in file_results
            ],
            "aggregate": dict(zip(self.class_names, aggregate.tolist())),
            "files_not_classified_over_cap": files_over_cap,
        }


def classify_commit(repo_path: str, sha: str, config_path: str | None = None) -> dict:
    return CommitClassifier(config_path).classify(repo_path, sha)
