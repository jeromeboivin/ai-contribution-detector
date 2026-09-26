from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def _evaluate_repo_once(cfg: dict, repo_path: str, expected_class: str, name: str | None, samples: int, as_json: bool,
                        added_files_only: bool = False, since=None, until=None) -> None:
    import time
    from pathlib import Path

    from aicontrib.diff.known_repo_eval import evaluate_known_repo
    from aicontrib.monitor import append_known_repo_result

    result = evaluate_known_repo(repo_path, expected_class, n_samples=samples, added_files_only=added_files_only,
                                 since=since, until=until)
    name = name or Path(repo_path).resolve().name
    if added_files_only:
        name += " (added files only)"  # its own dashboard card and trend: a different measurement

    append_known_repo_result(
        Path(cfg["paths"]["models_dir"]),
        timestamp=time.time(),
        name=name,
        repo=result["repo"],
        expected_class=result["expected_class"],
        accuracy=result["accuracy"],
        mean_expected_class_probability=result["mean_expected_class_probability"],
        mean_expected_class_probability_by_language=result["mean_expected_class_probability_by_language"],
        mean_probabilities=result["mean_probabilities"],
        predicted_counts=result["predicted_counts"],
        added_files_only=added_files_only,
        n_commits_evaluated=result["n_commits_evaluated"],
        n_commits_skipped_no_supported_files=result["n_commits_skipped_no_supported_files"],
    )
    port = cfg["monitor"]["port"]
    print(f"[{name}] logged to the dashboard -- see http://127.0.0.1:{port} (run `aicontrib monitor` if it's not already open)")

    if as_json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"Repo: {result['repo']}")
        print(f"Expected class: {result['expected_class']}")
        if since or until:
            print(f"Commits from {since or 'the start'} to {until or 'now'}")
        if added_files_only:
            print("Scope: only files each commit adds (modified files ignored)")
        skip_reason = "no newly added supported-language files" if added_files_only else "no supported-language changes"
        print(
            f"Commits: {result['n_commits_evaluated']} evaluated, "
            f"{result['n_commits_skipped_no_supported_files']} skipped ({skip_reason}), "
            f"{result['n_commits_sampled']} sampled total"
        )
        print(f"Accuracy (argmax == expected): {result['accuracy']:.1%}")
        print("Predicted class counts: " + ", ".join(f"{c} {n}" for c, n in result["predicted_counts"].items()))
        print("Mean probability per class: " + ", ".join(f"{c} {p:.3f}" for c, p in result["mean_probabilities"].items()))
        print(f"Mean P({result['expected_class']}): {result['mean_expected_class_probability']:.3f}")
        print(f"Mean P({result['expected_class']}) by language:")
        for lang, prob in sorted(result["mean_expected_class_probability_by_language"].items()):
            print(f"  {lang}: {prob:.3f}")
        print()


def _print_binoculars(result: dict) -> None:
    print(f"\nBinoculars ({result['observer']} / {result['performer']}), whole files at HEAD. "
          "Lower score = more AI-like.")
    has_mlp = any(r["mlp_mean_p_ai"] is not None for r in result["repos"])
    for r in result["repos"]:
        if r.get("since") or r.get("until"):
            print(f"  {r['name']}: files created from {r.get('since') or 'the start'} to {r.get('until') or 'now'}")
    print(f"{'repo':<24}{'expected':<13}{'files':>6}{'score p25':>11}{'median':>9}{'p75':>9}"
          + (f"{'MLP mean P(ai)':>17}" if has_mlp else ""))
    for r in result["repos"]:
        q = r["binoculars_quartiles"] or [float("nan")] * 3
        mlp = f"{r['mlp_mean_p_ai']:>17.3f}" if r["mlp_mean_p_ai"] is not None else ""
        print(f"{r['name']:<24}{r['expected_class']:<13}{len(r['files']):>6}{q[0]:>11.3f}{q[1]:>9.3f}{q[2]:>9.3f}{mlp}")
    if not result["pairs"]:
        print("\nNo AUC: known_repos needs at least one `human` and one `ai` repo.")
    else:
        print("\nSeparation, file level (AUC: 0.5 = no signal, 1.0 = every AI file ranks above every human file):")
        for pair in result["pairs"]:
            mlp = f", MLP classifier {pair['mlp_auc']:.3f}" if "mlp_auc" in pair else ""
            print(f"  {pair['human']} (human) vs {pair['ai']} (ai): Binoculars {pair['binoculars_auc']:.3f}{mlp}")
    print(f"\nPer-file scores: {result['results_path']}")


def _print_generator_holdout(summary: dict) -> None:
    print(f"\nHeld-out generators, embedding representation '{summary['representation']}', "
          f"languages: {summary['languages']}")
    print("AUC of each generator's code vs human code (0.5 = no signal, 1.0 = perfect):")
    print(f"{'generator':<32}{'rows':>6}{'seen':>8}{'unseen':>9}{'drop':>8}")
    for r in summary["generators"]:
        print(f"{r['generator']:<32}{r['n_test']:>6}{r['seen_auc']:>8.3f}{r['unseen_auc']:>9.3f}"
              f"{r['seen_auc'] - r['unseen_auc']:>8.3f}")
    print(f"{'mean':<38}{summary['mean_seen_auc']:>8.3f}{summary['mean_unseen_auc']:>9.3f}"
          f"{summary['mean_seen_auc'] - summary['mean_unseen_auc']:>8.3f}")
    print("seen = trained with that generator's code; unseen = trained without it (as with a new AI model).")
    print(f"Results: {summary['results_path']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="aicontrib")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit", help="Print sample rows per raw dataset label to validate the label mapping")
    subparsers.add_parser(
        "language-census",
        help="(Maintainers) Rebuild the per-language statistics and AICD-Bench language index (~20 min)",
    )
    agent_parser = subparsers.add_parser(
        "build-agent-commits",
        help="Build training data from real commits signed by coding agents (clones repos from GitHub)",
    )
    agent_parser.add_argument("--max-repos", type=int, help="Override agent_commits.max_repos (e.g. 3 for a trial)")
    subparsers.add_parser("prepare", help="Stream and cache the labeled/capped train/val/test JSONL splits")
    embed_parser = subparsers.add_parser("embed", help="Compute and cache embeddings for each split")
    embed_parser.add_argument("--force", action="store_true", help="Recompute even if cached")
    train_parser = subparsers.add_parser("train", help="Train the MLP classifier on cached embeddings")
    train_parser.add_argument(
        "--resume", action="store_true",
        help="Continue from the saved best model instead of starting over (e.g. after raising early_stopping_patience)",
    )
    subparsers.add_parser("evaluate", help="Evaluate the trained classifier on the test split")
    subparsers.add_parser(
        "monitor", help="Serve the training dashboard standalone (train already starts one automatically)"
    )

    commit_parser = subparsers.add_parser("classify-commit", help="Classify a git commit's changed code")
    commit_parser.add_argument("repo_path", help="Path to a local git repository")
    commit_parser.add_argument("sha", help="Commit SHA to classify")

    eval_repo_parser = subparsers.add_parser(
        "evaluate-repo",
        help="Sanity-check the classifier against a local repo with known ground-truth authorship",
    )
    eval_repo_parser.add_argument("repo_path", nargs="?", help="Path to a local git repository")
    eval_repo_parser.add_argument(
        "expected_class", nargs="?", help="Known ground-truth class for this repo: human or ai (see classes.names)"
    )
    eval_repo_parser.add_argument(
        "--all", action="store_true", help="Run every repo listed under known_repos in configs/local.yaml instead"
    )
    eval_repo_parser.add_argument("--samples", type=int, default=50, help="Number of commits to sample (default 50)")
    eval_repo_parser.add_argument("--json", action="store_true", help="Print full per-commit results as JSON")
    eval_repo_parser.add_argument(
        "--name", help="Label for the dashboard (default: the repo directory's basename)"
    )
    eval_repo_parser.add_argument("--since", help="Only commits after this date, e.g. 2019-01-01 (not with --all)")
    eval_repo_parser.add_argument("--until", help="Only commits before this date, e.g. 2021-12-31 (not with --all)")
    eval_repo_parser.add_argument(
        "--added-files-only", action="store_true",
        help="Score only files each commit adds (whole files, like the training data); skip commits that add none",
    )

    subparsers.add_parser(
        "generator-holdout",
        help="How well the classifier catches code from AI models it never saw in training (CodeMirage)",
    )
    binoculars_parser = subparsers.add_parser(
        "binoculars",
        help="(Prototype) Zero-shot Binoculars detector: how well it separates your known_repos, vs the MLP",
    )
    binoculars_parser.add_argument(
        "--files", type=int, help="Files to sample per repo (default: binoculars.files_per_repo in the config)"
    )

    report_parser = subparsers.add_parser(
        "report", help="Classify every commit of a local repo and write a static HTML authorship timeline"
    )
    report_parser.add_argument("repo_path", help="Path to a local git repository")
    report_parser.add_argument(
        "-o", "--output", help="HTML file to write (default: ./<repo-name>-authorship-timeline.html)"
    )
    report_parser.add_argument(
        "--max-commits", type=int, help="Only analyze N commits, sampled evenly across history (quick preview)"
    )
    report_parser.add_argument(
        "--no-cache", action="store_true", help="Reclassify every commit instead of reusing cached results"
    )
    report_parser.add_argument("--since", help="Only commits from this date, e.g. 2018-01-01")
    report_parser.add_argument("--until", help="Only commits up to this date, e.g. 2022-11-30")

    args = parser.parse_args(argv)

    if args.command == "audit":
        from aicontrib.data.audit import audit

        audit()
    elif args.command == "language-census":
        import time

        from aicontrib.config import load_config
        from aicontrib.data.languages import run_census

        run_census(load_config(), log=lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True))
    elif args.command == "build-agent-commits":
        from aicontrib.config import load_config
        from aicontrib.data.agent_commits import build_agent_commits

        build_agent_commits(load_config(), max_repos=args.max_repos)
    elif args.command == "prepare":
        from aicontrib.data.prepare import prepare_all

        prepare_all()
    elif args.command == "embed":
        from aicontrib.features.embed import embed_all

        embed_all(force=args.force)
    elif args.command == "train":
        from aicontrib.model.train import train

        train(resume=args.resume)
    elif args.command == "evaluate":
        from aicontrib.model.evaluate import evaluate

        evaluate()
    elif args.command == "monitor":
        from pathlib import Path

        from aicontrib.config import load_config
        from aicontrib.monitor import serve_dashboard

        cfg = load_config()
        metrics_path = Path(cfg["paths"]["models_dir"]) / "metrics.jsonl"
        port = cfg["monitor"]["port"]
        print(f"Serving training dashboard at http://127.0.0.1:{port} (reading {metrics_path})")
        serve_dashboard(metrics_path, port)
    elif args.command == "classify-commit":
        from aicontrib.diff.commit import classify_commit

        result = classify_commit(args.repo_path, args.sha)
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif args.command == "evaluate-repo":
        from pathlib import Path

        from aicontrib.config import load_config

        cfg = load_config()

        if args.all:
            known_repos = cfg.get("known_repos", [])
            if not known_repos:
                print(
                    "No known_repos configured. Copy configs/local.example.yaml to configs/local.yaml "
                    "and add at least one entry (see README 'Real-world validation')."
                )
                return
            targets = [(r["path"], r["expected_class"], r.get("name"), r.get("since"), r.get("until")) for r in known_repos]
        elif args.repo_path and args.expected_class:
            targets = [(args.repo_path, args.expected_class, args.name, args.since, args.until)]
        else:
            print("Either pass <repo_path> <expected_class>, or use --all to run every configured known_repos entry.")
            return

        for repo_path, expected_class, name, since, until in targets:
            try:
                _evaluate_repo_once(cfg, repo_path, expected_class, name, args.samples, args.json, args.added_files_only,
                                    since, until)
            except Exception as exc:  # noqa: BLE001 - one bad entry (e.g. missing local path) shouldn't abort the rest
                print(f"[{name or repo_path}] skipped: {exc}")
                print()
    elif args.command == "generator-holdout":
        from aicontrib.config import load_config
        from aicontrib.model.generator_holdout import run_generator_holdout

        _print_generator_holdout(run_generator_holdout(load_config()))
    elif args.command == "binoculars":
        from aicontrib.binoculars import evaluate_binoculars
        from aicontrib.config import load_config

        cfg = load_config()
        if not cfg.get("known_repos"):
            sys.exit("No known_repos configured -- add them to configs/local.yaml (see README 'Real-world validation').")
        _print_binoculars(evaluate_binoculars(cfg, files_per_repo=args.files))
    elif args.command == "report":
        import subprocess
        from pathlib import Path

        from aicontrib.config import load_config
        from aicontrib.diff.timeline import write_report

        checkpoint = Path(load_config()["paths"]["models_dir"]) / "mlp_classifier.pt"
        if not checkpoint.exists():
            sys.exit(f"No trained model at {checkpoint} -- run `prepare`, `embed` and `train` first (see README).")
        try:
            out = write_report(args.repo_path, args.output, max_commits=args.max_commits, use_cache=not args.no_cache,
                               since=args.since, until=args.until)
        except subprocess.CalledProcessError as exc:
            sys.exit(f"git failed on {args.repo_path!r}: {exc.stderr.decode('utf-8', 'replace').strip()}")
        print(f"Report written to {out}")


def run() -> None:
    """Process entry point (the `aicontrib` command and `python -m aicontrib`).

    Ends with os._exit, skipping interpreter finalization: when a Hugging Face `datasets`
    stream is abandoned early (prepare stops once the class caps are reached; audit after a
    sample), a native read-ahead thread is still running, and finalizing with it alive aborts
    the process ("Fatal Python error: PyGILState_Release", exit code 134) -- after the work is
    done, but alarming and a failing exit code. Closing the stream doesn't stop that thread.
    All our output files are closed by their `with` blocks before we get here.
    """
    code = 0
    try:
        main()
    except SystemExit as exc:  # argparse errors and our sys.exit("message") calls
        if isinstance(exc.code, int) or exc.code is None:
            code = exc.code or 0
        else:
            print(exc.code, file=sys.stderr)
            code = 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        code = 130
    except BaseException:  # noqa: BLE001 - report it ourselves, then exit without finalization
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    run()
