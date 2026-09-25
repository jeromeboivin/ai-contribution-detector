# ai-contribution-detector

Classifies whether code was **written by hand**, **co-authored** (human + AI), or **fully AI-generated** —
first at the snippet/file level, then applied to git commits as an approximation. Targets Python,
JavaScript/TypeScript, C++, and C#. Trains on CPU; auto-uses a GPU if one is available.

## How it works

1. A frozen pretrained code encoder ([`Salesforce/codet5p-110m-embedding`](https://huggingface.co/Salesforce/codet5p-110m-embedding))
   turns a code snippet into a 256-dim vector.
2. A small trainable MLP classifies that vector into `human` / `co_authored` / `ai`.
3. Only the MLP is trained — the encoder is frozen, so training is fast even on CPU. A GPU, if present,
   is used automatically to speed up the one-time embedding pass over the dataset.
4. For git commits, the changed region of each file in the diff is reconstructed and classified the same
   way, then aggregated to one commit-level prediction weighted by lines changed per file.

## Dataset

Training data is pulled from two sources (`configs/default.yaml` → `dataset.sources`), combined until each
class hits its per-split cap (see `data/prepare.py`):

1. **[AICD-Bench](https://huggingface.co/datasets/AICD-bench/AICD-Bench)** (EACL 2026, config `T3`), our
   primary and only source for `co_authored` samples. It labels code as human / machine / hybrid /
   adversarial across 9 languages. We map:

   | AICD-Bench label | Our class     |
   |---|---|
   | human             | `human`       |
   | hybrid            | `co_authored` |
   | machine           | `ai`          |
   | adversarial       | `ai` (folded in — still AI-authored, just harder to detect) |

   The exact label→int mapping isn't published by the dataset authors; `aicontrib/data/sources.py` encodes
   a hypothesis inferred from spot-checking real rows. **Run `aicontrib audit` and eyeball the output before
   trusting it** (see Known limitations).

2. **[CodeMirage](https://huggingface.co/datasets/HanxiGuo/CodeMirage)** (arXiv:2506.11059) tops up the
   `human`/`ai` buckets with extra diversity — different generator families (Claude, o3-mini) not in
   AICD-Bench's 11. It's binary-only (no hybrid label), so it never contributes to `co_authored`.
   **Licensed CC-BY-NC-ND-4.0** — non-commercial use only, no redistributing a derivative model. Set
   `enabled: false` on it in `configs/default.yaml` if this stops being a personal/research project.

**DroidCollection is deliberately not a separate source.** AICD-Bench is built directly on top of Droid —
extended with new generators/languages and MinHash-deduplicated *against* Droid itself (AICD-Bench paper,
arXiv:2602.02079, §3) — so Droid's content is already folded into AICD-Bench. Adding it again would only
reintroduce near-duplicate rows the AICD authors deliberately removed, risking train/test leakage.

Both sources cover Python, JavaScript, C++, and C# directly, but not TypeScript — nor does any other
AI-vs-human code dataset I could find (also checked: HybridCodeAuthorship, MultiAIGCD, HMCorp,
AIGCodeSet). `.ts`/`.tsx` files are routed through the JavaScript-trained path as an approximation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`torch` installs the CPU wheel by default. To use a GPU, install a CUDA build of `torch` matching your
driver instead (see [pytorch.org/get-started](https://pytorch.org/get-started/locally/)) — everything else
in this repo auto-detects it via `torch.cuda.is_available()`.

## Usage

```bash
# 1. Sanity-check the raw label mapping against real content (see Known limitations)
python -m aicontrib audit

# 2. Stream AICD-Bench and write capped, remapped train/val/test JSONL to data/processed/
python -m aicontrib prepare

# 3. Embed every split with the frozen encoder (the slow, one-time step)
python -m aicontrib embed

# 4. Train the MLP head (also starts a live dashboard at http://127.0.0.1:8765)
python -m aicontrib train

# 5. Evaluate on the held-out test split
python -m aicontrib evaluate

# 6. Classify a real commit
python -m aicontrib classify-commit /path/to/some/repo <commit-sha>

# 7. Sanity-check against a repo with KNOWN ground-truth authorship
python -m aicontrib evaluate-repo /path/to/some/repo human --samples 50
```

Config (dataset caps, encoder choice, MLP size, hyperparameters, supported file extensions) lives in
`configs/default.yaml`.

## Live training dashboard

`aicontrib train` starts a small local HTTP server (port configurable via `monitor.port` in
`configs/default.yaml`, default `8765`) that serves a self-contained HTML page — no external JS
libraries, no internet needed — showing live-updating charts of training loss and validation macro-F1,
polling every second. Open the printed URL in a browser during training.

To reopen the dashboard for a run that's already in progress (or to re-view a finished run's log)
without starting a new training job:

```bash
python -m aicontrib monitor
```

It reads `models/metrics.jsonl`, one JSON line per epoch (`{"epoch": ..., "train_loss": ..., "val_macro_f1": ...}`),
written by `train.py`.

## Real-world validation against known-authorship repos

The held-out AICD-Bench/CodeMirage test split measures in-distribution accuracy, but can't catch
domain-shift failures — e.g. the TypeScript-via-JavaScript-proxy approximation, or a codebase whose style
just doesn't resemble the training sources. `aicontrib evaluate-repo` samples commits (evenly spread
across the repo's full history, not just recent ones) from a *local* repo whose authorship is already
known with certainty, runs each through `classify_commit`, and reports accuracy and mean predicted
probability for the expected class — broken down by language, so a TypeScript-specific weakness shows up
directly instead of being averaged away. Results are logged to the [live dashboard](#live-training-dashboard)
(a new "Known-repo validation" section) and accumulate across runs, so re-running after each retrain
builds a trend of accuracy over time per repo.

```bash
# One-off, explicit path + expected class:
python -m aicontrib evaluate-repo /path/to/some/repo human --samples 50

# Or configure a standing list once and run them all together:
python -m aicontrib evaluate-repo --all
```

**Configuring known repos** (`configs/local.yaml`, gitignored — never enters git history): copy
`configs/local.example.yaml` to `configs/local.yaml` and list your repos there. A repo's local path is
machine-specific by nature, and for a *private* repo it also shouldn't be visible in this project's public
history at all — `configs/default.yaml` ships `known_repos: []` and only `local.yaml` (merged in
automatically by `aicontrib/config.py`) supplies real entries:

```yaml
known_repos:
  - name: tslint
    path: "/absolute/path/to/tslint"
    expected_class: human
  - name: my-private-ai-repo
    path: "/absolute/path/to/your/private/repo"
    expected_class: ai
```

One well-tested public option for the `human` side, chosen specifically because it's almost entirely
TypeScript (the language with no real training data — see Known limitations):
[`palantir/tslint`](https://github.com/palantir/tslint) — archived, 2,895 commits spanning 2013-07-14 to
its last commit on 2021-03-25, safely before ChatGPT's public launch (Nov 2022) ruled out any LLM
assistance. 333 TypeScript source files, 99.4% TypeScript by byte count per the GitHub API.

The automated `pytest` suite only checks the `evaluate-repo` plumbing against a synthetic throwaway repo
(`tests/test_known_repo_eval.py`) — it doesn't depend on `configs/local.yaml` or any specific machine's
paths.

## Performance

The embedding step (`aicontrib embed`) is the only slow part -- it's a forward pass through a 110M-param
transformer for every row, and the MLP training itself is fast regardless of hardware. Measured throughput
on an 8-core CPU with realistic (non-trivial-length) code samples was **~1.6 rows/sec**, so the default
caps (6,300 rows total across splits) take roughly an hour. Runtime scales roughly linearly with
`per_class_cap` in `configs/default.yaml` -- raise it for a stronger model once the pipeline works
end-to-end, at a proportional time cost. A CUDA GPU (auto-detected) speeds this step up substantially;
exact speedup depends on the card.

## Known limitations

- **Trained on whole snippets, applied to diffs.** There's no diff-level ground-truth dataset publicly
  available, so `classify-commit` reconstructs the post-change text of each hunk and classifies that —
  expect commit-level accuracy to be noticeably below the snippet-level test metrics reported by
  `aicontrib evaluate`.
- **TypeScript isn't in the training data** — `.ts`/`.tsx` files reuse the JavaScript path, unvalidated.
- **Label semantics for AICD-Bench are inferred, not documented.** `aicontrib/data/sources.py`'s
  `_AICD_BENCH_LABEL_MAP` is a hypothesis based on manually reading sample rows, not an official mapping
  from the dataset authors.
- **No per-language breakdown.** AICD-Bench's public parquet has no `language` column (CodeMirage does),
  so we can't currently report or filter accuracy per programming language.
- **PR-level aggregation is out of scope for this phase.** The natural next step once commit-level
  predictions are validated is aggregating across a PR's commits.
