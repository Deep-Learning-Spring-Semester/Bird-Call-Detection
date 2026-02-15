"""
DEEPNET Evaluation Script

Evaluate a trained model on the test set and generate comprehensive metrics report.

Usage:
    uv run python -m deepnet.evaluate --checkpoint checkpoints/baseline_v1/best.pt
    uv run python -m deepnet.evaluate --checkpoint checkpoints/baseline_v1/best.pt --save-cm confusion_matrix.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

from deepnet.dataset import build_dataloaders
from deepnet.metrics import MetricsTracker
from deepnet.model import DeepNet
from deepnet.utils import get_device, load_config


def plot_confusion_matrix(cm, class_names, save_path=None, normalize=False, top_k=None):
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix (num_classes, num_classes)
        class_names: List of class names
        save_path: Path to save the figure (optional)
        normalize: Whether to normalize by row (show percentages)
        top_k: Only show top-k most confused classes (optional)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Handle division by zero

    num_classes = len(class_names)

    # Filter to top-k most confused if specified
    if top_k and top_k < num_classes:
        # Find classes with most errors
        off_diagonal = cm.copy()
        np.fill_diagonal(off_diagonal, 0)
        errors_per_class = off_diagonal.sum(axis=1)
        top_indices = np.argsort(errors_per_class)[-top_k:]

        cm = cm[np.ix_(top_indices, top_indices)]
        class_names = [class_names[i] for i in top_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'},
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    title = 'Confusion Matrix'
    if normalize:
        title += ' (Normalized)'
    if top_k:
        title += f' - Top {top_k} Most Confused Classes'
    ax.set_title(title, fontsize=14, pad=20)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")

    return fig


def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        class_names: List of class names

    Returns:
        metrics: Dict of evaluation metrics
    """
    model.eval()
    tracker = MetricsTracker(num_classes=len(class_names), class_names=class_names)

    print("\nEvaluating on test set...")
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluation"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            tracker.update(outputs, targets)

    metrics = tracker.compute()
    return metrics


def print_metrics_report(metrics):
    """
    Print comprehensive metrics report.

    Args:
        metrics: Dict of metrics from evaluate_model
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Overall metrics
    print("\n📊 Overall Performance:")
    print(f"  Top-1 Accuracy:  {metrics['top1_accuracy']:.2f}%")
    print(f"  Top-3 Accuracy:  {metrics['top3_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy:  {metrics['top5_accuracy']:.2f}%")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")
    print(f"  Test samples:    {metrics['num_samples']}")

    # Hardest classes
    print("\n🔴 Hardest Classes (Lowest F1):")
    for i, cls in enumerate(metrics['hardest_classes'], 1):
        print(f"  {i}. {cls['class_name']:<30} F1: {cls['f1']:.3f}  "
              f"(P: {cls['precision']:.3f}, R: {cls['recall']:.3f}, N: {cls['support']})")

    # Easiest classes
    print("\n🟢 Easiest Classes (Highest F1):")
    for i, cls in enumerate(metrics['easiest_classes'], 1):
        print(f"  {i}. {cls['class_name']:<30} F1: {cls['f1']:.3f}  "
              f"(P: {cls['precision']:.3f}, R: {cls['recall']:.3f}, N: {cls['support']})")

    # Most confused pairs
    print("\n🔄 Most Confused Class Pairs:")
    for i, (true_cls, pred_cls, count, pct) in enumerate(metrics['confused_pairs'][:10], 1):
        print(f"  {i}. {true_cls:<25} → {pred_cls:<25} "
              f"{count:3d} times ({pct:.1f}%)")

    print("\n" + "=" * 80)


def save_metrics_json(metrics, save_path):
    """
    Save metrics to JSON file.

    Args:
        metrics: Dict of metrics
        save_path: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = {
        'top1_accuracy': float(metrics['top1_accuracy']),
        'top3_accuracy': float(metrics['top3_accuracy']),
        'top5_accuracy': float(metrics['top5_accuracy']),
        'macro_f1': float(metrics['macro_f1']),
        'weighted_f1': float(metrics['weighted_f1']),
        'num_samples': int(metrics['num_samples']),
        'per_class': metrics['per_class'],
        'confused_pairs': [
            {
                'true_class': true_cls,
                'predicted_class': pred_cls,
                'count': int(count),
                'percentage': float(pct)
            }
            for true_cls, pred_cls, count, pct in metrics['confused_pairs']
        ],
        'hardest_classes': metrics['hardest_classes'],
        'easiest_classes': metrics['easiest_classes']
    }

    with open(save_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)

    print(f"✓ Metrics saved to {save_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate DEEPNET model on test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: use config from checkpoint)"
    )
    parser.add_argument(
        "--save-cm",
        type=str,
        default=None,
        help="Path to save confusion matrix plot (e.g., confusion_matrix.png)"
    )
    parser.add_argument(
        "--save-metrics",
        type=str,
        default=None,
        help="Path to save metrics JSON (e.g., metrics.json)"
    )
    parser.add_argument(
        "--normalize-cm",
        action="store_true",
        help="Normalize confusion matrix to show percentages"
    )
    parser.add_argument(
        "--top-k-cm",
        type=int,
        default=None,
        help="Only show top-k most confused classes in confusion matrix"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to run evaluation on (default: auto-detect)"
    )
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return 1

    # Get device
    if args.device:
        device = torch.device(args.device)
        print(f"Using specified device: {device}")
    else:
        device = get_device()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load config
    if args.config:
        config = load_config(args.config)
    elif 'config' in checkpoint:
        config = checkpoint['config']
        # Reconstruct full config from flattened trainer config
        config = {
            'experiment_name': config.get('experiment_name', 'unknown'),
            'data': {
                'batch_size': 32,
                'num_workers': 4,
                'use_augmented': False,
                'weighted_sampling': True
            },
            'model': {
                'num_classes': 18,
                'dropout': 0.5
            },
            'training': {'seed': 42}
        }
    else:
        print("Warning: No config found in checkpoint, using defaults")
        config = {
            'data': {'batch_size': 32, 'num_workers': 4, 'use_augmented': False, 'weighted_sampling': True},
            'model': {'num_classes': 18, 'dropout': 0.5},
            'training': {'seed': 42}
        }

    # Build test dataloader
    print("\nBuilding test dataloader...")
    data_config = {
        'batch_size': config['data']['batch_size'],
        'num_workers': config['data']['num_workers'],
        'use_augmented': False,  # Never use augmented data for test
        'use_weighted_sampler': False,  # No weighted sampling for test
        'seed': config['training'].get('seed', 42)
    }

    _, _, test_loader, label_map, _ = build_dataloaders(data_config)

    # Get class names (sorted by index)
    class_names = sorted(label_map.keys(), key=lambda k: label_map[k])
    num_classes = len(class_names)

    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {num_classes}")

    # Create model
    print("\nCreating model...")
    model = DeepNet(num_classes=num_classes, dropout=config['model']['dropout'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"✓ Model loaded ({model.count_parameters():,} parameters)")

    # Evaluate
    metrics = evaluate_model(model, test_loader, device, class_names)

    # Print report
    print_metrics_report(metrics)

    # Save confusion matrix plot
    if args.save_cm:
        save_path = Path(args.save_cm)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plot_confusion_matrix(
            metrics['confusion_matrix'],
            class_names,
            save_path=save_path,
            normalize=args.normalize_cm,
            top_k=args.top_k_cm
        )

    # Save metrics JSON
    if args.save_metrics:
        save_path = Path(args.save_metrics)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_metrics_json(metrics, save_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
