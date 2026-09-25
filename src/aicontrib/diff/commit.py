"""Commit-level classification (MVP heuristic).

The MLP was trained on whole human/AI code *snippets*, not diffs -- there is
no diff-level ground-truth dataset (see plan/README). As an approximation,
we reconstruct the post-change text of each hunk (context + added lines) per
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


def _get_unified_diff(repo_path: str, sha: str) -> str:
    result = subprocess.run(
        ["git", "-C", repo_path, "show", "--unified=3", "--format=", sha],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


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
    return "".join(lines), changed


def _language_for(path: str, cfg: dict) -> str | None:
    ext = Path(path).suffix.lower()
    return cfg["commit_classification"]["supported_extensions"].get(ext)


def classify_commit(repo_path: str, sha: str, config_path: str | None = None) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    device = get_device()

    checkpoint_path = Path(cfg["paths"]["models_dir"]) / "mlp_classifier.pt"
    model = load_checkpoint(checkpoint_path, device)
    embedder = CodeEmbedder(cfg)
    class_names = cfg["classes"]["names"]

    diff_text = _get_unified_diff(repo_path, sha)
    patch = PatchSet(diff_text)

    file_results: list[FileResult] = []
    for patched_file in patch:
        if patched_file.is_removed_file or patched_file.is_binary_file:
            continue
        language = _language_for(patched_file.path, cfg)
        if language is None:
            continue

        text, lines_changed = _post_image_text(patched_file)
        if not text.strip() or lines_changed == 0:
            continue

        embedding = embedder.embed_many([text])
        with torch.no_grad():
            x = torch.tensor(embedding, dtype=torch.float32).to(device)
            probs = torch.softmax(model(x), dim=-1).cpu().numpy()[0]

        file_results.append(
            FileResult(
                path=patched_file.path,
                language=language,
                lines_changed=lines_changed,
                probabilities=dict(zip(class_names, probs.tolist())),
            )
        )

    if not file_results:
        return {"commit": sha, "files": [], "aggregate": None, "note": "no supported-language files with changes found"}

    weights = np.array([fr.lines_changed for fr in file_results], dtype=np.float64)
    weights = weights / weights.sum()
    prob_matrix = np.array([[fr.probabilities[name] for name in class_names] for fr in file_results])
    aggregate = (weights[:, None] * prob_matrix).sum(axis=0)

    return {
        "commit": sha,
        "files": [
            {"path": fr.path, "language": fr.language, "lines_changed": fr.lines_changed, "probabilities": fr.probabilities}
            for fr in file_results
        ],
        "aggregate": dict(zip(class_names, aggregate.tolist())),
    }
