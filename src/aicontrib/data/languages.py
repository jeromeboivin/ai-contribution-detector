"""Programming-language labels for the training data, and language filtering.

CodeMirage has a `language` column. AICD-Bench doesn't, so its languages are computed once by
`aicontrib language-census` and shipped with the code as a compact per-row index
(aicd_t3_languages.npz, one byte per row, labels only -- no code):

- AICD-Bench is built on DroidCollection, which is language-labelled: a row whose code is
  byte-identical to a Droid row takes Droid's label.
- The remaining rows (AICD-Bench's own additions) are labelled by a token n-gram classifier
  trained on Droid's labels. The census validates it on held-out Droid rows and on CodeMirage
  (an independent pipeline); both results are recorded in language_stats.json.

The index is tied to a pinned dataset revision (`hf_revision` in configs/default.yaml), and
`prepare` checks a row fingerprint every CHECKPOINT_EVERY rows so a misaligned index fails
loudly instead of silently mislabelling.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
INDEX_PATH = HERE / "aicd_t3_languages.npz"
STATS_PATH = HERE / "language_stats.json"

# Append only: the shipped AICD-Bench index stores positions in the list as it was when built.
LANGUAGES = ["C", "C#", "C++", "Go", "Java", "JavaScript", "PHP", "Python", "Rust", "Ruby", "HTML", "TypeScript"]
UNKNOWN = 255
CHECKPOINT_EVERY = 50_000
AICD_SPLITS = ("train", "validation", "test")
_ALIASES = {"cpp": "C++", "csharp": "C#", "cs": "C#", "golang": "Go", "js": "JavaScript", "py": "Python", "ts": "TypeScript"}


def canonical(name: str | None) -> str | None:
    if name is None:
        return None
    by_lower = {lang.lower(): lang for lang in LANGUAGES}
    key = str(name).strip().lower()
    return by_lower.get(key) or _ALIASES.get(key)


def language_filter(cfg: dict) -> Callable[[str | None], bool] | None:
    """From `dataset.languages: {include: [...]} / {exclude: [...]}`; None when no filter is set.
    If both are given, a language must be in `include` and not in `exclude`."""
    spec = cfg["dataset"].get("languages") or {}
    include, exclude = spec.get("include") or [], spec.get("exclude") or []
    if not include and not exclude:
        return None

    def resolve(names: list) -> set[str]:
        out = set()
        for name in names:
            lang = canonical(name)
            if lang is None:
                raise ValueError(f"Unknown language {name!r} in dataset.languages. "
                                 f"Valid names: {', '.join(LANGUAGES)}")
            out.add(lang)
        return out

    include_set, exclude_set = resolve(include), resolve(exclude)

    def allowed(lang: str | None) -> bool:
        if lang is None:
            return False
        return (not include_set or lang in include_set) and lang not in exclude_set

    return allowed


def _fingerprint(code: str | None) -> str:
    if code is None:
        return "none"
    return hashlib.sha1(code.encode("utf-8", errors="replace")).hexdigest()[:16]


class AicdLanguageIndex:
    def __init__(self, path: Path = INDEX_PATH):
        data = np.load(path, allow_pickle=False)
        self.revision = str(data["revision"])
        self.names = [str(n) for n in data["languages"]]
        self.codes = {split: data[split] for split in AICD_SPLITS}
        self.checkpoints = {
            split: dict(zip(data[f"{split}_ckpt_pos"].tolist(), [str(h) for h in data[f"{split}_ckpt_hash"]]))
            for split in AICD_SPLITS
        }

    def language(self, split: str, row: int) -> str | None:
        code = int(self.codes[split][row])
        return None if code == UNKNOWN else self.names[code]

    def verify(self, split: str, row: int, code: str | None) -> None:
        expected = self.checkpoints[split].get(row)
        if expected is not None and expected != _fingerprint(code):
            raise RuntimeError(
                f"AICD-Bench {split} row {row} doesn't match the language index -- the dataset no longer "
                f"lines up with revision {self.revision}. Run `aicontrib language-census` to rebuild it."
            )


@lru_cache(maxsize=1)
def load_index() -> AicdLanguageIndex | None:
    return AicdLanguageIndex() if INDEX_PATH.exists() else None


def save_index(path: Path, revision: str, codes: dict[str, np.ndarray],
               checkpoints: dict[str, tuple[list[int], list[str]]]) -> None:
    np.savez_compressed(
        path, revision=np.array(revision), languages=np.array(LANGUAGES),
        **{split: np.asarray(codes[split], dtype=np.uint8) for split in AICD_SPLITS},
        **{f"{s}_ckpt_pos": np.array(checkpoints[s][0], dtype=np.int64) for s in AICD_SPLITS},
        **{f"{s}_ckpt_hash": np.array(checkpoints[s][1], dtype="U16") for s in AICD_SPLITS},
    )


def stats_table_markdown(stats: dict) -> str:
    """Rows per language across all splits, per dataset and combined, as a Markdown table."""
    per = {}
    for dataset in ("aicd_bench", "codemirage"):
        counts = Counter()
        for split in stats[dataset]["splits"].values():
            counts.update(split["by_language"])
        per[dataset] = counts
    combined = per["aicd_bench"] + per["codemirage"]
    totals = {k: sum(v.values()) for k, v in per.items()} | {"combined": sum(combined.values())}

    def cell(n, total):
        return f"{n:,} ({100 * n / total:.1f}%)" if n else "—"

    lines = ["| Language | AICD-Bench | CodeMirage | Combined |", "|---|---:|---:|---:|"]
    for lang, n in combined.most_common():
        lines.append(f"| {lang} | {cell(per['aicd_bench'][lang], totals['aicd_bench'])} | "
                     f"{cell(per['codemirage'][lang], totals['codemirage'])} | {cell(n, totals['combined'])} |")
    lines.append(f"| **Total** | **{totals['aicd_bench']:,}** | **{totals['codemirage']:,}** | "
                 f"**{totals['combined']:,}** |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Census (maintainers only: streams DroidCollection, AICD-Bench and CodeMirage, ~15-20 minutes)
# ---------------------------------------------------------------------------------------------

def _vectorizer():
    from sklearn.feature_extraction.text import HashingVectorizer
    # Case-sensitive identifier tokens (String vs string) + short punctuation runs (::, ->, =>, <?).
    return HashingVectorizer(token_pattern=r"[A-Za-z_]\w*|[^\w\s]{1,2}", lowercase=False, ngram_range=(1, 2),
                             n_features=2**20, alternate_sign=False, preprocessor=lambda s: s[:3000])


def _sha1(code: str) -> bytes:
    return hashlib.sha1(code.encode("utf-8", errors="replace")).digest()


def run_census(cfg: dict, log: Callable[[str], None] = print) -> dict:
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.svm import LinearSVC

    aicd_cfg = next(s for s in cfg["dataset"]["sources"] if s["adapter"] == "aicd_bench")
    cm_cfg = next(s for s in cfg["dataset"]["sources"] if s["adapter"] == "codemirage")
    revision = aicd_cfg["hf_revision"]

    # 1. DroidCollection: exact code -> language, plus a per-language sample to train the classifier.
    log("Streaming DroidCollection (language-labelled)...")
    droid, reservoir, seen, rng = {}, defaultdict(list), Counter(), random.Random(0)
    for split in ("train", "dev", "test"):
        for row in load_dataset("project-droid/DroidCollection", split=split, streaming=True):
            lang = canonical(row["Language"])
            if not row["Code"] or lang is None:
                continue
            droid[_sha1(row["Code"])] = lang
            seen[lang] += 1
            if len(reservoir[lang]) < 3000:
                reservoir[lang].append(row["Code"])
            elif (j := rng.randrange(seen[lang])) < 3000:
                reservoir[lang][j] = row["Code"]
    log(f"  {len(droid):,} unique labelled snippets")

    # 2. Train the classifier on 80% of the sample; validate on the held-out 20% and on CodeMirage.
    train_x, train_y, test_x, test_y = [], [], [], []
    for lang, codes in sorted(reservoir.items()):
        rng.shuffle(codes)
        cut = int(len(codes) * 0.8)
        train_x += codes[:cut]; train_y += [lang] * cut
        test_x += codes[cut:]; test_y += [lang] * (len(codes) - cut)
    vec = _vectorizer()
    clf = LinearSVC(C=0.5).fit(vec.transform(train_x), train_y)
    droid_acc = accuracy_score(test_y, clf.predict(vec.transform(test_x)))

    cm_x, cm_y, per = [], [], Counter()
    for row in load_dataset(cm_cfg["hf_repo"], split="test", streaming=True):
        lang = canonical(row["language"])
        if lang in reservoir and row["code"] and per[lang] < 1000:
            cm_x.append(row["code"]); cm_y.append(lang); per[lang] += 1
    cm_pred = clf.predict(vec.transform(cm_x))
    cm_report = classification_report(cm_y, cm_pred, output_dict=True, zero_division=0)
    log(f"  classifier accuracy: held-out Droid {droid_acc:.1%}, CodeMirage {accuracy_score(cm_y, cm_pred):.1%}")

    # 3. AICD-Bench: label every row (exact Droid match, else classifier) and build the index.
    names = LANGUAGES
    arrays, ckpts, aicd_stats = {}, {}, {}
    for split in AICD_SPLITS:
        log(f"Labelling AICD-Bench {split}...")
        codes_out, exact, classified = [], Counter(), Counter()
        ck_pos, ck_hash, pending_pos, pending_code = [], [], [], []

        def flush():
            if pending_code:
                for pos, lang in zip(pending_pos, clf.predict(vec.transform(pending_code))):
                    codes_out[pos] = names.index(lang)
                    classified[lang] += 1
            pending_pos.clear(); pending_code.clear()

        for i, row in enumerate(load_dataset(aicd_cfg["hf_repo"], aicd_cfg["hf_config"], split=split,
                                             streaming=True, revision=revision)):
            code = row["code"]
            if i % CHECKPOINT_EVERY == 0:
                ck_pos.append(i); ck_hash.append(_fingerprint(code))
            codes_out.append(UNKNOWN)
            if not code or not code.strip():
                continue
            lang = droid.get(_sha1(code))
            if lang:
                codes_out[i] = names.index(lang)
                exact[lang] += 1
            else:
                pending_pos.append(i); pending_code.append(code)
                if len(pending_code) >= 20_000:
                    flush()
        flush()
        arrays[split] = codes_out
        ckpts[split] = (ck_pos, ck_hash)
        by_lang = exact + classified
        aicd_stats[split] = {"rows": len(codes_out), "by_language": dict(sorted(by_lang.items())),
                             "exact_droid_match": sum(exact.values()), "classified": sum(classified.values())}
        log(f"  {len(codes_out):,} rows: {sum(exact.values()):,} exact Droid matches, "
            f"{sum(classified.values()):,} classified")

    save_index(INDEX_PATH, revision, arrays, ckpts)

    # 4. CodeMirage: exact, from its own column.
    cm_stats = {}
    for split in ("train", "test"):
        counts = Counter()
        for row in load_dataset(cm_cfg["hf_repo"], split=split, streaming=True):
            if row["code"] and row["code"].strip():
                counts[canonical(row["language"])] += 1
        cm_stats[split] = {"rows": sum(counts.values()), "by_language": dict(sorted(counts.items()))}

    stats = {
        "generated": dt.date.today().isoformat(),
        "aicd_bench": {"revision": revision, "splits": aicd_stats},
        "codemirage": {"splits": cm_stats},
        "classifier_validation": {
            "held_out_droid_accuracy": round(droid_acc, 4),
            "codemirage_accuracy": round(accuracy_score(cm_y, cm_pred), 4),
            # Only languages CodeMirage actually has (no Rust there -- a 0.0 would be meaningless).
            "codemirage_f1_by_language": {k: round(v["f1-score"], 4) for k, v in cm_report.items()
                                          if k in names and v["support"] > 0},
        },
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    load_index.cache_clear()
    log(f"Wrote {INDEX_PATH.name} and {STATS_PATH.name}. README table:\n{stats_table_markdown(stats)}")
    return stats
