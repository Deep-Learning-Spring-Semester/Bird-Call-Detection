"""
Evaluation metrics for bird call classification.

Implements:
- Top-k accuracy (k=1, 3, 5)
- Macro and weighted F1 scores
- Per-class precision, recall, F1
- Confusion matrix computation and visualization
"""

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_topk_accuracy(outputs, targets, k=1):
    """
    Compute top-k accuracy.

    Args:
        outputs: Model predictions (batch, num_classes) - logits or probabilities
        targets: Ground truth labels (batch,) - integer class indices
        k: Top-k value (default: 1 for top-1 accuracy)

    Returns:
        accuracy: Top-k accuracy as percentage (0-100)
    """
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Get top-k predictions
    topk_preds = np.argsort(outputs, axis=1)[:, -k:]

    # Check if true label is in top-k predictions
    correct = np.array([targets[i] in topk_preds[i] for i in range(len(targets))])

    accuracy = 100.0 * correct.sum() / len(targets)
    return accuracy


def compute_f1_scores(predictions, targets, num_classes):
    """
    Compute macro and weighted F1 scores.

    Args:
        predictions: Predicted labels (N,)
        targets: Ground truth labels (N,)
        num_classes: Number of classes

    Returns:
        dict with keys:
            - macro_f1: Macro-averaged F1 (unweighted mean across classes)
            - weighted_f1: Weighted F1 (weighted by class support)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    macro_f1 = f1_score(targets, predictions, average='macro', zero_division=0)
    weighted_f1 = f1_score(targets, predictions, average='weighted', zero_division=0)

    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }


def compute_per_class_metrics(predictions, targets, class_names=None):
    """
    Compute per-class precision, recall, and F1 scores.

    Args:
        predictions: Predicted labels (N,)
        targets: Ground truth labels (N,)
        class_names: List of class names (optional, uses indices if None)

    Returns:
        List of dicts, one per class, with keys:
            - class_name: Class name or index
            - precision: Precision for this class
            - recall: Recall for this class
            - f1: F1 score for this class
            - support: Number of samples in this class
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )

    num_classes = len(precision)
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    per_class = []
    for i in range(num_classes):
        per_class.append({
            'class_name': class_names[i],
            'class_id': i,
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': int(support[i])
        })

    return per_class


def compute_confusion_matrix(predictions, targets, num_classes):
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted labels (N,)
        targets: Ground truth labels (N,)
        num_classes: Number of classes

    Returns:
        cm: Confusion matrix (num_classes, num_classes)
            cm[i, j] = number of samples with true label i predicted as j
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    cm = confusion_matrix(targets, predictions, labels=range(num_classes))
    return cm


def find_most_confused_pairs(cm, class_names=None, top_k=10):
    """
    Find the most confused class pairs from confusion matrix.

    Args:
        cm: Confusion matrix (num_classes, num_classes)
        class_names: List of class names (optional)
        top_k: Number of top confused pairs to return

    Returns:
        List of tuples (true_class, pred_class, count, percentage)
        sorted by count in descending order
    """
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    confused_pairs = []

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:  # Off-diagonal elements
                percentage = 100.0 * cm[i, j] / cm[i].sum() if cm[i].sum() > 0 else 0
                confused_pairs.append((
                    class_names[i],
                    class_names[j],
                    int(cm[i, j]),
                    percentage
                ))

    # Sort by count (descending)
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    return confused_pairs[:top_k]


def find_hardest_classes(per_class_metrics, top_k=5):
    """
    Find the hardest classes (lowest F1 scores).

    Args:
        per_class_metrics: List of per-class metric dicts from compute_per_class_metrics
        top_k: Number of hardest classes to return

    Returns:
        List of dicts for the hardest classes, sorted by F1 (ascending)
    """
    sorted_classes = sorted(per_class_metrics, key=lambda x: x['f1'])
    return sorted_classes[:top_k]


def find_easiest_classes(per_class_metrics, top_k=5):
    """
    Find the easiest classes (highest F1 scores).

    Args:
        per_class_metrics: List of per-class metric dicts from compute_per_class_metrics
        top_k: Number of easiest classes to return

    Returns:
        List of dicts for the easiest classes, sorted by F1 (descending)
    """
    sorted_classes = sorted(per_class_metrics, key=lambda x: x['f1'], reverse=True)
    return sorted_classes[:top_k]


class MetricsTracker:
    """
    Track and aggregate metrics during evaluation.

    Usage:
        tracker = MetricsTracker(num_classes=18)
        for batch in dataloader:
            outputs, targets = model(inputs), batch_targets
            tracker.update(outputs, targets)
        metrics = tracker.compute()
    """

    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.all_predictions = []
        self.all_targets = []
        self.all_outputs = []

    def update(self, outputs, targets):
        """
        Update tracker with a batch of predictions.

        Args:
            outputs: Model outputs (batch, num_classes) - logits or probabilities
            targets: Ground truth labels (batch,)
        """
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu()

        predictions = outputs.argmax(dim=1)

        self.all_predictions.append(predictions)
        self.all_targets.append(targets)
        self.all_outputs.append(outputs)

    def compute(self):
        """
        Compute all metrics from tracked predictions.

        Returns:
            dict with comprehensive metrics
        """
        # Concatenate all batches
        predictions = torch.cat(self.all_predictions).numpy()
        targets = torch.cat(self.all_targets).numpy()
        outputs = torch.cat(self.all_outputs).numpy()

        # Top-k accuracies
        top1_acc = compute_topk_accuracy(outputs, targets, k=1)
        top3_acc = compute_topk_accuracy(outputs, targets, k=3)
        top5_acc = compute_topk_accuracy(outputs, targets, k=5)

        # F1 scores
        f1_scores = compute_f1_scores(predictions, targets, self.num_classes)

        # Per-class metrics
        per_class = compute_per_class_metrics(predictions, targets, self.class_names)

        # Confusion matrix
        cm = compute_confusion_matrix(predictions, targets, self.num_classes)

        # Most confused pairs
        confused_pairs = find_most_confused_pairs(cm, self.class_names, top_k=10)

        # Hardest and easiest classes
        hardest = find_hardest_classes(per_class, top_k=5)
        easiest = find_easiest_classes(per_class, top_k=5)

        return {
            'top1_accuracy': top1_acc,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
            'macro_f1': f1_scores['macro_f1'],
            'weighted_f1': f1_scores['weighted_f1'],
            'per_class': per_class,
            'confusion_matrix': cm,
            'confused_pairs': confused_pairs,
            'hardest_classes': hardest,
            'easiest_classes': easiest,
            'num_samples': len(targets)
        }


if __name__ == "__main__":
    # Test metrics with sample data
    print("Testing metrics module...\n")

    np.random.seed(42)
    num_samples = 100
    num_classes = 18

    # Generate random predictions and targets
    outputs = np.random.randn(num_samples, num_classes)
    targets = np.random.randint(0, num_classes, num_samples)

    # Test top-k accuracy
    top1 = compute_topk_accuracy(outputs, targets, k=1)
    top3 = compute_topk_accuracy(outputs, targets, k=3)
    print(f"Top-1 accuracy: {top1:.2f}%")
    print(f"Top-3 accuracy: {top3:.2f}%")

    # Test F1 scores
    predictions = outputs.argmax(axis=1)
    f1_scores = compute_f1_scores(predictions, targets, num_classes)
    print(f"\nMacro F1: {f1_scores['macro_f1']:.4f}")
    print(f"Weighted F1: {f1_scores['weighted_f1']:.4f}")

    # Test MetricsTracker
    tracker = MetricsTracker(num_classes=num_classes)
    tracker.update(torch.tensor(outputs), torch.tensor(targets))
    metrics = tracker.compute()

    print("\nMetricsTracker results:")
    print(f"  Top-1: {metrics['top1_accuracy']:.2f}%")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Samples: {metrics['num_samples']}")

    print("\n✓ Metrics module test complete!")
