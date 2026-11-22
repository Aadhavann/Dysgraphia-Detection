"""Command-line entry point.

    python run.py                          # full study, Score-CAM, 2 seeds
    python run.py --cam gradcam            # faster, for iterating
    python run.py --stability-runs 1       # skip the CAM stability measurement
    python run.py --epochs 20 --seed 7     # shorter run, different seed
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from edi import PipelineConfig, run_pipeline


def parse_args(argv=None) -> argparse.Namespace:
    defaults = PipelineConfig()

    parser = argparse.ArgumentParser(
        description="Train the dysgraphia classifier, build the Explainable Dysgraphia Index and report held-out results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir, help="dataset root holding the two class folders")
    parser.add_argument("--results-dir", type=Path, default=defaults.results_dir, help="where figures, tables and checkpoints are written")
    parser.add_argument("--model-type", choices=("resnet18", "efficientnet_b0"), default=defaults.model_type)
    parser.add_argument("--cam", dest="cam_method", choices=("scorecam", "gradcam"), default="scorecam",
                        help="scorecam matches the published method; gradcam is much faster")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--test-size", type=float, default=defaults.test_size)
    parser.add_argument("--val-size", type=float, default=defaults.val_size)
    parser.add_argument("--scorecam-channels", type=int, default=defaults.scorecam_channels,
                        help="channels Score-CAM masks per image; higher is slower and marginally more faithful")
    parser.add_argument("--cv-folds", type=int, default=defaults.cv_folds, help="folds for out-of-fold head evaluation")
    parser.add_argument("--stability-runs", type=int, default=defaults.stability_runs,
                        help="models to train from different seeds; 2 or more enables a measured CAM stability IoU")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    config = replace(
        PipelineConfig(),
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        scorecam_channels=args.scorecam_channels,
        cv_folds=args.cv_folds,
        stability_runs=args.stability_runs,
    )

    if not config.data_dir.is_dir():
        print(f"error: data directory not found: {config.data_dir}", file=sys.stderr)
        return 2

    try:
        outcome = run_pipeline(
            config,
            cam_method=args.cam_method,
            write_figures=not args.no_figures,
            verbose=not args.quiet,
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    delta = outcome["manifest"]["fusion_minus_cnn_auc"]
    if delta is not None and delta <= 0:
        print(
            "\nnote: the EDI did not improve on the CNN alone in this run. "
            "That is a legitimate outcome and is recorded as-is in results_summary.txt.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
