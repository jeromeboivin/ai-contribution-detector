from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
# Gitignored -- for machine-local settings that shouldn't enter git history, such as the
# path to a private known-authorship repo (see README "Real-world validation").
LOCAL_CONFIG_PATH = REPO_ROOT / "configs" / "local.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    # Explicit utf-8: Windows' default locale encoding (cp1252) can't decode every byte.
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("raw_cache_dir", "processed_dir", "embeddings_dir", "models_dir", "timeline_cache_dir"):
        if key in cfg["paths"]:
            cfg["paths"][key] = str(REPO_ROOT / cfg["paths"][key])

    # Only auto-merge local.yaml for the default config -- an explicitly passed config_path
    # (e.g. in tests, or a smoke-test config) should stay self-contained and reproducible.
    if Path(path) == DEFAULT_CONFIG_PATH and LOCAL_CONFIG_PATH.exists():
        with open(LOCAL_CONFIG_PATH, encoding="utf-8") as f:
            try:
                local_cfg = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"Could not parse {LOCAL_CONFIG_PATH}: {exc}\n"
                    "If this is a Windows path, write it as 'C:\\Users\\...' (single quotes) or "
                    '"C:/Users/..." (forward slashes) -- see configs/local.example.yaml.'
                ) from exc
        cfg["known_repos"] = cfg.get("known_repos", []) + local_cfg.get("known_repos", [])

    return cfg
