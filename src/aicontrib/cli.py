from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def _evaluate_repo_once(cfg: dict, repo_path: str, expected_class: str, name: str | None, samples: int, as_json: bool) -> None:
    import time
    from pathlib import Path

    from aicontrib.diff.known_repo_eval import evaluate_known_repo
    from aicontrib.monitor import append_known_repo_result

    result = evaluate_known_repo(repo_path, expected_class, n_samples=samples)
    name = name or Path(repo_path).resolve().name

    append_known_repo_result(
        Path(cfg["paths"]["models_dir"]),
        timestamp=time.time(),
        name=name,
        repo=result["repo"],
        expected_class=result["expected_class"],
        accuracy=result["accuracy"],
        mean_expected_class_probability=result["mean_expected_class_probability"],
        mean_expected_class_probability_by_language=result["mean_expected_class_probability_by_language"],
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
        print(
            f"Commits: {result['n_commits_evaluated']} evaluated, "
            f"{result['n_commits_skipped_no_supported_files']} skipped (no supported-language changes), "
            f"{result['n_commits_sampled']} sampled total"
        )
        print(f"Accuracy (argmax == expected): {result['accuracy']:.1%}")
        print(f"Mean P({result['expected_class']}): {result['mean_expected_class_probability']:.3f}")
        print(f"Mean P({result['expected_class']}) by language:")
        for lang, prob in sorted(result["mean_expected_class_probability_by_language"].items()):
            print(f"  {lang}: {prob:.3f}")
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="aicontrib")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit", help="Print sample rows per raw dataset label to validate the label mapping")
    subparsers.add_parser(
        "language-census",
        help="(Maintainers) Rebuild the per-language statistics and AICD-Bench language index (~20 min)",
    )
    subparsers.add_parser("prepare", help="Stream and cache the labeled/capped train/val/test JSONL splits")
    embed_parser = subparsers.add_parser("embed", help="Compute and cache embeddings for each split")
    embed_parser.add_argument("--force", action="store_true", help="Recompute even if cached")
    subparsers.add_parser("train", help="Train the MLP classifier on cached embeddings")
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
        "expected_class", nargs="?", choices=["human", "co_authored", "ai"], help="Known ground-truth class for this repo"
    )
    eval_repo_parser.add_argument(
        "--all", action="store_true", help="Run every repo listed under known_repos in configs/local.yaml instead"
    )
    eval_repo_parser.add_argument("--samples", type=int, default=50, help="Number of commits to sample (default 50)")
    eval_repo_parser.add_argument("--json", action="store_true", help="Print full per-commit results as JSON")
    eval_repo_parser.add_argument(
        "--name", help="Label for the dashboard (default: the repo directory's basename)"
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

    args = parser.parse_args(argv)

    if args.command == "audit":
        from aicontrib.data.audit import audit

        audit()
    elif args.command == "language-census":
        import time

        from aicontrib.config import load_config
        from aicontrib.data.languages import run_census

        run_census(load_config(), log=lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True))
    elif args.command == "prepare":
        from aicontrib.data.prepare import prepare_all

        prepare_all()
    elif args.command == "embed":
        from aicontrib.features.embed import embed_all

        embed_all(force=args.force)
    elif args.command == "train":
        from aicontrib.model.train import train

        train()
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
            targets = [(r["path"], r["expected_class"], r.get("name")) for r in known_repos]
        elif args.repo_path and args.expected_class:
            targets = [(args.repo_path, args.expected_class, args.name)]
        else:
            print("Either pass <repo_path> <expected_class>, or use --all to run every configured known_repos entry.")
            return

        for repo_path, expected_class, name in targets:
            try:
                _evaluate_repo_once(cfg, repo_path, expected_class, name, args.samples, args.json)
            except Exception as exc:  # noqa: BLE001 - one bad entry (e.g. missing local path) shouldn't abort the rest
                print(f"[{name or repo_path}] skipped: {exc}")
                print()
    elif args.command == "report":
        import subprocess
        from pathlib import Path

        from aicontrib.config import load_config
        from aicontrib.diff.timeline import write_report

        checkpoint = Path(load_config()["paths"]["models_dir"]) / "mlp_classifier.pt"
        if not checkpoint.exists():
            sys.exit(f"No trained model at {checkpoint} -- run `prepare`, `embed` and `train` first (see README).")
        try:
            out = write_report(args.repo_path, args.output, max_commits=args.max_commits, use_cache=not args.no_cache)
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
