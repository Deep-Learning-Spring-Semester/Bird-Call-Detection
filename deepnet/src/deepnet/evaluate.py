"""
DEEPNET Evaluation Script

Evaluate a trained model and generate a full set of presentation-quality plots:
  • Confusion matrix (normalized heatmap)
  • Per-species metrics bar chart (F1, precision, recall)
  • Accuracy vs. sample count scatter (data efficiency analysis)
  • Calibration reliability diagram (confidence vs. actual accuracy)
  • Training curves (if training history is available in the checkpoint)

Usage:
    # Basic evaluation (prints metrics only)
    uv run python -m deepnet.evaluate --checkpoint checkpoints/best_v1/best.pt

    # Full presentation package (all plots saved to results/best_v1/)
    uv run python -m deepnet.evaluate --checkpoint checkpoints/best_v1/best.pt --comprehensive

    # Save individual outputs
    uv run python -m deepnet.evaluate --checkpoint checkpoints/best_v1/best.pt \\
        --save-cm results/cm.png --save-metrics results/metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

from deepnet.dataset import build_dataloaders
from deepnet.metrics import MetricsTracker
from deepnet.model import DeepNet
from deepnet.utils import get_device, load_config


# ---------------------------------------------------------------------------
# Plot: Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm, class_names, save_path=None, normalize=True, top_k=None, title=None
):
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix (num_classes, num_classes)
        class_names: List of class names
        save_path: Path to save the figure (optional)
        normalize: Whether to normalize by row (show fractions)
        top_k: Only show top-k most confused classes (optional)
        title: Override plot title
    """
    if normalize:
        cm_plot = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_plot = np.nan_to_num(cm_plot)
        fmt = ".2f"
        cbar_label = "Fraction of true class"
    else:
        cm_plot = cm
        fmt = "d"
        cbar_label = "Count"

    names = [n.replace("_", " ") for n in class_names]

    # Filter to top-k most confused if specified
    if top_k and top_k < len(names):
        off_diagonal = cm.copy()
        np.fill_diagonal(off_diagonal, 0)
        errors_per_class = off_diagonal.sum(axis=1)
        top_indices = np.argsort(errors_per_class)[-top_k:]
        cm_plot = cm_plot[np.ix_(top_indices, top_indices)]
        names = [names[i] for i in top_indices]

    n = len(names)
    fig_size = max(10, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        cbar_kws={"label": cbar_label, "shrink": 0.8},
        ax=ax,
        linewidths=0.4,
        linecolor="#cccccc",
        annot_kws={"size": max(7, 11 - n // 3)},
    )

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)

    if title is None:
        title = "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"
        if top_k:
            title += f" — Top {top_k} Most Confused Species"
    ax.set_title(title, fontsize=13, pad=15, fontweight="bold")

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Confusion matrix saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Plot: Per-species metrics bar chart
# ---------------------------------------------------------------------------


def plot_per_species_metrics(per_class, save_path=None, title=None):
    """
    Horizontal bar chart of per-species F1, precision, and recall.

    Args:
        per_class: List of per-class metric dicts from MetricsTracker
        save_path: Path to save figure
        title: Override plot title
    """
    # Sort by F1 descending
    sorted_classes = sorted(per_class, key=lambda x: x["f1"], reverse=True)

    names = [c["class_name"].replace("_", " ") for c in sorted_classes]
    f1s = [c["f1"] for c in sorted_classes]
    precisions = [c["precision"] for c in sorted_classes]
    recalls = [c["recall"] for c in sorted_classes]
    supports = [c["support"] for c in sorted_classes]

    n = len(names)
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.45)))

    y = np.arange(n)
    bar_h = 0.25

    bars_f1 = ax.barh(y + bar_h, f1s, bar_h, label="F1", color="#2196F3", alpha=0.85)
    bars_p = ax.barh(
        y, precisions, bar_h, label="Precision", color="#4CAF50", alpha=0.85
    )
    bars_r = ax.barh(
        y - bar_h, recalls, bar_h, label="Recall", color="#FF9800", alpha=0.85
    )

    # Annotate support counts on the right
    for i, (support, f1) in enumerate(zip(supports, f1s)):
        ax.text(
            min(f1 + 0.02, 0.97),
            y[i] + bar_h,
            f"n={support}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#555555",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_xlim(0, 1.15)
    ax.set_title(
        title or "Per-Species Metrics (sorted by F1)", fontsize=13, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Per-species metrics saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Plot: Accuracy vs sample count scatter
# ---------------------------------------------------------------------------


def plot_accuracy_vs_support(per_class, save_path=None, title=None):
    """
    Scatter plot of per-class recall (accuracy) vs. training sample count.

    Helps visualise data-efficiency: do species with more samples perform better?
    """
    names = [c["class_name"].replace("_", " ") for c in per_class]
    recalls = np.array([c["recall"] for c in per_class])
    supports = np.array([c["support"] for c in per_class])

    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        supports,
        recalls * 100,
        s=80,
        alpha=0.8,
        c=recalls,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        edgecolors="k",
        linewidths=0.4,
    )

    # Annotate each point with the species name
    for name, x, y in zip(names, supports, recalls * 100):
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
            alpha=0.85,
        )

    # Trend line
    if len(supports) > 2:
        z = np.polyfit(supports, recalls * 100, 1)
        p = np.poly1d(z)
        x_line = np.linspace(supports.min(), supports.max(), 100)
        ax.plot(x_line, p(x_line), "b--", linewidth=1.2, alpha=0.6, label="Trend")

    plt.colorbar(scatter, ax=ax, label="Recall")
    ax.set_xlabel("Test samples per species (support)", fontsize=11)
    ax.set_ylabel("Recall / Class Accuracy (%)", fontsize=11)
    ax.set_title(
        title or "Class Accuracy vs. Sample Count", fontsize=13, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Accuracy vs support saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Plot: Calibration reliability diagram
# ---------------------------------------------------------------------------


def plot_calibration(outputs_np, targets_np, save_path=None, n_bins=10, title=None):
    """
    Reliability / calibration diagram.

    Buckets predictions by confidence (max softmax probability) and plots
    mean confidence vs. actual accuracy per bin.  A perfectly calibrated
    model lies on the diagonal.

    Args:
        outputs_np: Raw logits (N, C) as numpy array
        targets_np: Ground truth labels (N,) as numpy array
        save_path: Path to save figure
        n_bins: Number of calibration bins
        title: Override plot title
    """
    # Convert logits → probabilities
    exp_out = np.exp(outputs_np - outputs_np.max(axis=1, keepdims=True))
    probs = exp_out / exp_out.sum(axis=1, keepdims=True)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == targets_np).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(np.nan)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)

    # ECE (Expected Calibration Error)
    total = len(targets_np)
    ece = np.nansum(np.abs(bin_accs - bin_confs) * bin_counts / total)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    valid = ~np.isnan(bin_accs)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
    ax1.bar(
        bin_confs[valid],
        bin_accs[valid],
        width=(bin_edges[1] - bin_edges[0]) * 0.9,
        alpha=0.6,
        color="steelblue",
        label="Model",
    )
    ax1.plot(bin_confs[valid], bin_accs[valid], "ro-", markersize=5, linewidth=1.5)
    ax1.set_xlabel("Mean Confidence", fontsize=11)
    ax1.set_ylabel("Actual Accuracy", fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title(
        f"Reliability Diagram  (ECE = {ece:.3f})", fontsize=12, fontweight="bold"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Confidence histogram
    ax2.hist(confidences, bins=n_bins, color="steelblue", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Predicted Confidence", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Confidence Distribution", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Calibration diagram saved → {save_path}")

    return fig, ece


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def evaluate_model(model, test_loader, device, class_names):
    """
    Run full evaluation on test set.

    Returns:
        metrics dict (from MetricsTracker.compute()) plus raw 'outputs' and 'targets'
    """
    model.eval()
    tracker = MetricsTracker(num_classes=len(class_names), class_names=class_names)

    all_outputs = []
    all_targets = []

    print("\nEvaluating on test set...")
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluation"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            tracker.update(outputs, targets)
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    metrics = tracker.compute()
    metrics["outputs_np"] = np.concatenate(all_outputs, axis=0)
    metrics["targets_np"] = np.concatenate(all_targets, axis=0)
    return metrics


def print_metrics_report(metrics, class_names):
    """Print the full metrics report to stdout."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Overall metrics
    print("\nOverall Performance:")
    print(f"  Top-1 Accuracy:  {metrics['top1_accuracy']:.2f}%")
    print(f"  Top-3 Accuracy:  {metrics['top3_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy:  {metrics['top5_accuracy']:.2f}%")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")
    print(f"  Test samples:    {metrics['num_samples']}")

    # Per-species table
    print("\nPer-Species Metrics:")
    print(f"  {'Species':<32} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print(f"  {'-' * 55}")
    for cls in sorted(metrics["per_class"], key=lambda x: x["f1"], reverse=True):
        name = cls["class_name"].replace("_", " ")[:31]
        print(
            f"  {name:<32} {cls['precision']:>6.3f} {cls['recall']:>6.3f} "
            f"{cls['f1']:>6.3f} {cls['support']:>5}"
        )

    # Hardest classes
    print("\nHardest Classes (Lowest F1):")
    for i, cls in enumerate(metrics["hardest_classes"], 1):
        name = cls["class_name"].replace("_", " ")
        print(
            f"  {i}. {name:<30} F1: {cls['f1']:.3f}  "
            f"(P: {cls['precision']:.3f}, R: {cls['recall']:.3f}, N: {cls['support']})"
        )

    # Easiest classes
    print("\nEasiest Classes (Highest F1):")
    for i, cls in enumerate(metrics["easiest_classes"], 1):
        name = cls["class_name"].replace("_", " ")
        print(
            f"  {i}. {name:<30} F1: {cls['f1']:.3f}  "
            f"(P: {cls['precision']:.3f}, R: {cls['recall']:.3f}, N: {cls['support']})"
        )

    # Top-20 confused pairs
    print("\nTop-20 Most Confused Pairs:")
    for i, (true_cls, pred_cls, count, pct) in enumerate(
        metrics["confused_pairs"][:20], 1
    ):
        tc = true_cls.replace("_", " ")
        pc = pred_cls.replace("_", " ")
        print(f"  {i:2}. {tc:<28} → {pc:<28} {count:3d} ({pct:.1f}%)")

    print("\n" + "=" * 80)


def save_metrics_json(metrics, save_path):
    """Save metrics to JSON (presentation-ready)."""
    metrics_json = {
        "top1_accuracy": float(metrics["top1_accuracy"]),
        "top3_accuracy": float(metrics["top3_accuracy"]),
        "top5_accuracy": float(metrics["top5_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "num_samples": int(metrics["num_samples"]),
        "per_class": [
            {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in cls.items()
            }
            for cls in metrics["per_class"]
        ],
        "confused_pairs": [
            {
                "true_class": tc,
                "predicted_class": pc,
                "count": int(cnt),
                "percentage": float(pct),
            }
            for tc, pc, cnt, pct in metrics["confused_pairs"]
        ],
        "hardest_classes": metrics["hardest_classes"],
        "easiest_classes": metrics["easiest_classes"],
    }
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Metrics JSON saved → {save_path}")


def save_comprehensive_report(
    metrics, class_names, results_dir: Path, experiment_name: str
):
    """
    Generate and save the full Day-18 evaluation package.

    Saves:
      confusion_matrix.png
      confusion_matrix_normalized.png
      per_species_metrics.png
      accuracy_vs_support.png
      calibration.png
      metrics.json
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving comprehensive evaluation to: {results_dir}")

    # 1. Confusion matrix (normalized)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        save_path=results_dir / "confusion_matrix_normalized.png",
        normalize=True,
        title=f"Confusion Matrix (Normalized) — {experiment_name}",
    )
    plt.close("all")

    # 2. Confusion matrix (raw counts)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        save_path=results_dir / "confusion_matrix_counts.png",
        normalize=False,
        title=f"Confusion Matrix (Counts) — {experiment_name}",
    )
    plt.close("all")

    # 3. Per-species metrics bar chart
    plot_per_species_metrics(
        metrics["per_class"],
        save_path=results_dir / "per_species_metrics.png",
        title=f"Per-Species Metrics — {experiment_name}",
    )
    plt.close("all")

    # 4. Accuracy vs sample count
    plot_accuracy_vs_support(
        metrics["per_class"],
        save_path=results_dir / "accuracy_vs_support.png",
        title=f"Class Accuracy vs. Sample Count — {experiment_name}",
    )
    plt.close("all")

    # 5. Calibration reliability diagram
    _, ece = plot_calibration(
        metrics["outputs_np"],
        metrics["targets_np"],
        save_path=results_dir / "calibration.png",
        title=f"Calibration — {experiment_name}",
    )
    plt.close("all")
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")

    # 6. Metrics JSON
    save_metrics_json(metrics, results_dir / "metrics.json")

    print(f"\nAll plots saved to: {results_dir}")
    print("Files generated:")
    for f in sorted(results_dir.glob("*.png")):
        print(f"  {f.name}")
    for f in sorted(results_dir.glob("*.json")):
        print(f"  {f.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DEEPNET model — Day 18 comprehensive"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: use config from checkpoint)",
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Generate full Day-18 evaluation package (all plots + metrics JSON)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save comprehensive results (default: results/<experiment_name>)",
    )
    parser.add_argument(
        "--save-cm", type=str, default=None, help="Path to save confusion matrix plot"
    )
    parser.add_argument(
        "--save-metrics", type=str, default=None, help="Path to save metrics JSON"
    )
    parser.add_argument(
        "--normalize-cm",
        action="store_true",
        help="Normalize confusion matrix to show fractions",
    )
    parser.add_argument(
        "--top-k-cm",
        type=int,
        default=None,
        help="Only show top-k most confused classes in confusion matrix",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device (default: auto-detect)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return 1

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Resolve config
    if args.config:
        config = load_config(args.config)
    elif "config" in checkpoint:
        # Trainer stores a flattened config — reconstruct the nested form
        stored = checkpoint["config"]
        config = {
            "experiment_name": stored.get(
                "experiment_name", checkpoint_path.parent.name
            ),
            "data": {
                "batch_size": 32,
                "num_workers": 4,
                "use_augmented": False,
                "weighted_sampling": True,
            },
            "model": {
                "num_classes": 18,
                "dropout": stored.get("dropout", 0.5),
            },
            "training": {"seed": 42},
        }
    else:
        config = {
            "experiment_name": checkpoint_path.parent.name,
            "data": {
                "batch_size": 32,
                "num_workers": 4,
                "use_augmented": False,
                "weighted_sampling": True,
            },
            "model": {"num_classes": 18, "dropout": 0.5},
            "training": {"seed": 42},
        }

    experiment_name = config["experiment_name"]

    # Build test dataloader
    print("\nBuilding test dataloader...")
    data_config = {
        "batch_size": config["data"]["batch_size"],
        "num_workers": config["data"]["num_workers"],
        "use_augmented": False,
        "use_weighted_sampler": False,
        "seed": config["training"].get("seed", 42),
    }
    _, _, test_loader, label_map, _ = build_dataloaders(data_config)
    class_names = sorted(label_map.keys(), key=lambda k: label_map[k])
    num_classes = len(class_names)
    print(f"Test samples: {len(test_loader.dataset)}  |  Classes: {num_classes}")

    # Build model
    model = DeepNet(num_classes=num_classes, dropout=config["model"]["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Model loaded ({model.count_parameters():,} parameters)")

    # Evaluate
    metrics = evaluate_model(model, test_loader, device, class_names)

    # Print full report
    print_metrics_report(metrics, class_names)

    # Comprehensive package
    if args.comprehensive:
        results_dir = (
            Path(args.results_dir)
            if args.results_dir
            else Path("results") / experiment_name
        )
        save_comprehensive_report(metrics, class_names, results_dir, experiment_name)

    else:
        # Individual outputs requested
        if args.save_cm:
            plot_confusion_matrix(
                metrics["confusion_matrix"],
                class_names,
                save_path=args.save_cm,
                normalize=args.normalize_cm,
                top_k=args.top_k_cm,
            )
            plt.close("all")

        if args.save_metrics:
            save_metrics_json(metrics, args.save_metrics)

    return 0


if __name__ == "__main__":
    sys.exit(main())
