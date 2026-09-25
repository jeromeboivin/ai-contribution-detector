"""Zero-shot AI-code detection with Binoculars (Hans et al., 2024, arXiv:2401.12070) -- a prototype.

No training data: two language models that share a tokenizer read the code. The score is the
performer model's log-perplexity on the text divided by the cross-perplexity between the two
models (how surprised the performer is by the observer's next-token predictions). Machine-generated
text is unsurprising relative to what a model would expect anyway, so it scores LOWER.

The paper's decision threshold was fitted for its Falcon-7B pair and doesn't carry over to other
models, so this module only measures separation: the AUC between files of a known-human repo and
files of a known-AI repo (0.5 = no signal, 1.0 = perfect), next to the MLP classifier's AUC on the
same files when a trained checkpoint exists.

Files are read whole at HEAD (not as diffs) from the repos listed under known_repos.
"""
from __future__ import annotations

import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from aicontrib.device import describe_device, get_device
from aicontrib.diff.commit import run_git
from aicontrib.diff.known_repo_eval import evenly_spaced

POSITION_CHUNK = 128  # positions per step when comparing the two models' full-vocabulary distributions


def binoculars_scores(observer_logits: torch.Tensor, performer_logits: torch.Tensor, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor, chunk: int = POSITION_CHUNK) -> torch.Tensor:
    """One score per sequence: log-perplexity (performer) / log-cross-perplexity (observer vs performer).

    Logits are compared position by position in chunks: a full-vocabulary float32 copy of a
    512-token sequence is ~300 MB per model, too much at once on a 4 GB GPU."""
    targets = input_ids[:, 1:]
    mask = attention_mask[:, 1:].float()
    log_ppl = torch.zeros(input_ids.shape[0], device=input_ids.device)
    log_x_ppl = torch.zeros_like(log_ppl)
    for start in range(0, targets.shape[1], chunk):
        end = min(start + chunk, targets.shape[1])
        # Logits at position t predict token t + 1.
        perf_logp = performer_logits[:, start:end].float().log_softmax(-1)
        obs_p = observer_logits[:, start:end].float().softmax(-1)
        m = mask[:, start:end]
        log_ppl -= (perf_logp.gather(-1, targets[:, start:end, None]).squeeze(-1) * m).sum(1)
        log_x_ppl -= ((obs_p * perf_logp).sum(-1) * m).sum(1)
    return log_ppl / log_x_ppl


class BinocularsScorer:
    def __init__(self, bcfg: dict):
        self.device = get_device()
        print(f"Binoculars models running on: {describe_device(self.device)}")
        self.max_tokens = bcfg["max_tokens"]
        self.tokenizer = AutoTokenizer.from_pretrained(bcfg["observer"])
        # bf16 on GPU halves memory (two 0.5B models + logits must fit in 4 GB); fp32 on CPU.
        dtype = torch.float32
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.observer = self._load(bcfg["observer"], dtype)
        self.performer = self._load(bcfg["performer"], dtype)
        if self.observer.config.vocab_size != self.performer.config.vocab_size:
            raise ValueError(f"binoculars.observer and binoculars.performer must share a tokenizer "
                             f"({bcfg['observer']} and {bcfg['performer']} don't)")

    def _load(self, name: str, dtype: torch.dtype):
        model = AutoModelForCausalLM.from_pretrained(name)
        return model.to(device=self.device, dtype=dtype).eval()

    @torch.no_grad()
    def score(self, text: str) -> float:
        # One file at a time: no padding, and the logits of one sequence are all a 4 GB GPU can hold.
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_tokens).to(self.device)
        observer_logits = self.observer(**enc).logits
        performer_logits = self.performer(**enc).logits
        return binoculars_scores(observer_logits, performer_logits, enc["input_ids"], enc["attention_mask"]).item()


def sample_head_files(repo_path: str, extensions: dict[str, str], exclude: list[str], n: int) -> list[str]:
    """Up to n tracked source files, evenly spread over the sorted path list, minus excluded patterns
    (vendored and generated code: its authorship isn't the repo's)."""
    paths = [p for p in run_git(repo_path, "ls-files", "-z").split("\0") if p]
    keep = [p for p in paths if Path(p).suffix.lower() in extensions and not any(fnmatch(p, pat) for pat in exclude)]
    return evenly_spaced(sorted(keep), n)


def _auc(human: list[float], ai: list[float]) -> float:
    """P(a random AI file ranks above a random human file), for scores where higher = more AI-like."""
    return float(roc_auc_score([0] * len(human) + [1] * len(ai), human + ai))


def evaluate_binoculars(cfg: dict, files_per_repo: int | None = None, log: Callable[[str], None] = print) -> dict:
    bcfg = cfg["binoculars"]
    extensions = cfg["commit_classification"]["supported_extensions"]
    n = files_per_repo or bcfg["files_per_repo"]

    # 1. Files to score: long enough (per the Binoculars tokenizer) that the score isn't noise.
    tokenizer = AutoTokenizer.from_pretrained(bcfg["observer"])
    repos = []
    for entry in cfg.get("known_repos", []):
        name = entry.get("name") or Path(entry["path"]).resolve().name
        files, too_short = [], 0
        for path in sample_head_files(entry["path"], extensions, bcfg.get("exclude") or [], n):
            text = run_git(entry["path"], "show", f"HEAD:{path}").replace("\r\n", "\n")
            if len(tokenizer(text, truncation=True, max_length=bcfg["min_tokens"])["input_ids"]) < bcfg["min_tokens"]:
                too_short += 1
                continue
            files.append({"path": path, "language": extensions[Path(path).suffix.lower()], "text": text})
        repos.append({"name": name, "repo": entry["path"], "expected_class": entry["expected_class"],
                      "files": files, "n_files_too_short": too_short})
        log(f"[{name}] {len(files)} files to score ({too_short} skipped: under {bcfg['min_tokens']} tokens)")

    # 2. MLP baseline on the same files, if a model is trained. Done first and freed, so the GPU
    #    only ever holds one set of models.
    checkpoint = Path(cfg["paths"]["models_dir"]) / "mlp_classifier.pt"
    if checkpoint.exists():
        from aicontrib.diff.commit import CommitClassifier

        classifier = CommitClassifier()
        ai_index = classifier.class_names.index("ai")
        for r in repos:
            if r["files"]:
                probs = classifier._probabilities([f["text"] for f in r["files"]])
                for f, p in zip(r["files"], probs):
                    f["mlp_p_ai"] = float(p[ai_index])
        del classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        log(f"No MLP checkpoint at {checkpoint}: Binoculars results only, no MLP comparison")

    # 3. Binoculars.
    scorer = BinocularsScorer(bcfg)
    for r in repos:
        for f in tqdm(r["files"], desc=f"binoculars[{r['name']}]", unit="file"):
            f["binoculars"] = scorer.score(f["text"])

    # 4. Separation between every human repo and every AI repo.
    has_mlp = checkpoint.exists()
    pairs = []
    for h in (r for r in repos if r["expected_class"] == "human" and r["files"]):
        for a in (r for r in repos if r["expected_class"] == "ai" and r["files"]):
            pair = {"human": h["name"], "ai": a["name"],
                    # Lower Binoculars score = more AI-like, so negate it for the AUC.
                    "binoculars_auc": _auc([-f["binoculars"] for f in h["files"]], [-f["binoculars"] for f in a["files"]])}
            if has_mlp:
                pair["mlp_auc"] = _auc([f["mlp_p_ai"] for f in h["files"]], [f["mlp_p_ai"] for f in a["files"]])
            pairs.append(pair)

    for r in repos:
        scores = [f["binoculars"] for f in r["files"]]
        r["binoculars_quartiles"] = np.percentile(scores, [25, 50, 75]).tolist() if scores else None
        r["mlp_mean_p_ai"] = float(np.mean([f["mlp_p_ai"] for f in r["files"]])) if has_mlp and r["files"] else None
        for f in r["files"]:
            del f["text"]

    result = {"observer": bcfg["observer"], "performer": bcfg["performer"], "max_tokens": bcfg["max_tokens"],
              "repos": repos, "pairs": pairs}
    out_path = Path(cfg["paths"]["models_dir"]) / "binoculars_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["results_path"] = str(out_path)
    return result
