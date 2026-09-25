from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
# Gitignored -- personal overrides of any setting in default.yaml (e.g. a smaller dataset for a
# quick prototype, or paths to private known-authorship repos). Keeping them out of the tracked
# default.yaml means `git pull` never conflicts with a user's local changes.
LOCAL_CONFIG_PATH = REPO_ROOT / "configs" / "local.yaml"


def _merge(base: dict, override: dict) -> None:
    """Recursive override: nested dicts are merged key by key, anything else replaced."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    # Explicit utf-8: Windows' default locale encoding (cp1252) can't decode every byte.
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

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
        known_repos = cfg.get("known_repos", []) + (local_cfg.pop("known_repos", None) or [])
        _merge(cfg, local_cfg)
        cfg["known_repos"] = known_repos  # appended rather than replaced

    for key in ("raw_cache_dir", "processed_dir", "embeddings_dir", "models_dir", "timeline_cache_dir"):
        if key in cfg["paths"]:
            cfg["paths"][key] = str(REPO_ROOT / cfg["paths"][key])
    return cfg
