"""End-to-end pipeline: train, explain, evaluate, report."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from . import __version__
from .cam import cam_stability_iou
from .config import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, PipelineConfig
from .data import DysgraphiaDataset, load_dataset, writer_grouped_split
from .evaluate import evaluate_all, format_report, fusion_improves_on_cnn, summarise
from .features import compute_edi_features, construct_edi
from .model import DysgraphiaClassifier, describe
from .train import TrainingHistory, resolve_device, set_seed, train_classifier
from .visualize import generate_case_studies, write_all_figures


def build_transforms(augment: bool):
    """Train-time augmentation is deliberately mild.

    Rotating or recolouring a handwriting sample much more than this changes the
    very features the EDI measures, so augmentation is kept to a small rotation
    and slight brightness/contrast jitter.
    """
    import torchvision.transforms as transforms

    common = [
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
    if augment:
        common += [
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    common += [
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ]
    return transforms.Compose(common)


def _environment_snapshot(device: torch.device) -> Dict[str, str]:
    snapshot = {
        "edi_version": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
    }
    try:
        import torchvision

        snapshot["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    if device.type == "cuda":
        snapshot["cuda_device"] = torch.cuda.get_device_name(0)
    return snapshot


def run_pipeline(
    config: Optional[PipelineConfig] = None,
    cam_method: str = "scorecam",
    write_figures: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run the full study and return everything worth archiving.

    When ``config.stability_runs`` is at least two, that many models are trained
    from different seeds and CAM stability is measured as the real mean IoU
    between their heatmaps. Setting it to one skips the measurement and reports
    it as not measured, rather than substituting a placeholder value.
    """
    config = config or PipelineConfig()
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = results_dir / "checkpoints"

    device = resolve_device()
    set_seed(config.seed)

    if verbose:
        print(f"device: {device}")
        print(f"cam method: {cam_method}")

    # 1. Data -----------------------------------------------------------------
    image_paths, labels, subjects = load_dataset(config.data_dir)
    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"no images found under {config.data_dir}. Expected subdirectories "
            f"'{Path(config.data_dir) / 'Low Potential Dysgraphia'}' and "
            f"'{Path(config.data_dir) / 'Potential Dysgraphia'}'."
        )

    train_split, val_split, test_split, split_report = writer_grouped_split(
        image_paths,
        labels,
        subjects,
        test_size=config.test_size,
        val_size=config.val_size,
        seed=config.seed,
    )

    if verbose:
        print("\ndata partition")
        print(split_report.as_text())

    train_transform = build_transforms(augment=True)
    eval_transform = build_transforms(augment=False)

    train_loader = DataLoader(
        DysgraphiaDataset(*train_split, train_transform), batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        DysgraphiaDataset(*val_split, eval_transform), batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    # Batch size 1 for the test loader: CAM is computed per image.
    test_loader = DataLoader(
        DysgraphiaDataset(*test_split, eval_transform), batch_size=1, shuffle=False, num_workers=0
    )

    # 2. Train ----------------------------------------------------------------
    runs: int = max(1, config.stability_runs)
    models: List[DysgraphiaClassifier] = []
    history: Optional[TrainingHistory] = None

    for run_index in range(runs):
        seed = config.seed + run_index
        set_seed(seed)
        if verbose:
            print(f"\ntraining model {run_index + 1}/{runs} (seed {seed}, {config.model_type})")

        model = DysgraphiaClassifier(model_type=config.model_type).to(device)
        if run_index == 0 and verbose:
            for line in describe(model):
                print(f"  {line}")

        model, run_history = train_classifier(
            model,
            train_loader,
            val_loader,
            device,
            config,
            checkpoint_path=checkpoints_dir / f"model_seed{seed}.pth",
            verbose=verbose,
        )
        models.append(model)
        if run_index == 0:
            history = run_history

    primary = models[0]
    assert history is not None
    if verbose:
        print(f"\nbest validation AUC {history.best_val_auc:.4f} at epoch {history.best_epoch}")

    # 3. Explain --------------------------------------------------------------
    if verbose:
        print(f"\ncomputing {cam_method} features on the test partition")
    features_df = compute_edi_features(
        primary,
        test_loader,
        device,
        cam_method=cam_method,
        num_channels=config.scorecam_channels,
        progress_every=10 if verbose else 0,
    )
    if features_df.empty:
        raise RuntimeError("no test images produced features; check the data directory and image readability")

    features_df, scaler = construct_edi(features_df, low_variance_eps=config.low_variance_eps)

    # 4. Evaluate -------------------------------------------------------------
    if verbose:
        print("\nevaluating heads out-of-fold")
    results = evaluate_all(features_df, cv_folds=config.cv_folds, seed=config.seed)

    # 5. CAM stability --------------------------------------------------------
    stability: Dict[str, object]
    if len(models) >= 2:
        if verbose:
            print(f"\nmeasuring CAM stability across {len(models)} independently trained models")
        test_tensors: List[torch.Tensor] = []
        for images, _, _ in test_loader:
            test_tensors.append(images[0:1].to(device))
        stability = cam_stability_iou(
            models,
            test_tensors,
            device,
            method=cam_method,
            threshold=config.cam_threshold,
            num_channels=config.scorecam_channels,
        )
    else:
        stability = {"mean_iou": None, "note": "stability_runs < 2; not measured"}
    if verbose:
        print(f"  {stability}")

    # 6. Artefacts ------------------------------------------------------------
    features_path = results_dir / "edi_features.csv"
    features_df.to_csv(features_path, index=False)

    summary_table = summarise(results)
    table_path = results_dir / "results_table.csv"
    summary_table.to_csv(table_path, index=False)

    torch.save(primary.state_dict(), results_dir / "trained_model.pth")

    run_manifest = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()},
        "cam_method": cam_method,
        "split": asdict(split_report),
        "environment": _environment_snapshot(device),
        "best_val_auc": history.best_val_auc,
        "best_epoch": history.best_epoch,
        "epochs_run": history.epochs_run,
        "stopped_epoch": history.stopped_epoch,
        "n_test_images": int(len(features_df)),
        "stability": {k: v for k, v in stability.items()},
        "results": {name: r.as_row() for name, r in results.items()},
        "fusion_minus_cnn_auc": fusion_improves_on_cnn(results),
    }
    manifest_path = results_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, default=str), encoding="utf-8")

    report_path = results_dir / "results_summary.txt"
    report_path.write_text(
        format_report(results, split_text=split_report.as_text(), stability=stability), encoding="utf-8"
    )

    figure_paths: Dict[str, Path] = {}
    case_paths: List[Path] = []
    if write_figures:
        if verbose:
            print("\nwriting figures")
        figure_paths = write_all_figures(features_df, history, results, results_dir)
        case_paths = generate_case_studies(
            primary,
            features_df,
            device,
            results_dir,
            cam_method=cam_method,
            num_channels=config.scorecam_channels,
        )
        figure_paths.update({path.stem: path for path in case_paths})

    if verbose:
        print("\n" + summary_table.to_string(index=False))
        delta = fusion_improves_on_cnn(results)
        if delta is not None:
            print(f"\nfusion vs CNN-only AUC: {delta:+.4f}")
        print(f"\nartefacts written to {results_dir}")

    return {
        "model": primary,
        "models": models,
        "features": features_df,
        "scaler": scaler,
        "results": results,
        "summary_table": summary_table,
        "stability": stability,
        "split_report": split_report,
        "history": history,
        "manifest": run_manifest,
        "figures": figure_paths,
        "paths": {
            "features": features_path,
            "table": table_path,
            "report": report_path,
            "manifest": manifest_path,
        },
    }
