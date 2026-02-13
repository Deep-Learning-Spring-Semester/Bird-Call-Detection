"""
Loss functions for bird call classification.

Implements three loss functions optimized for multi-class classification with class imbalance:
- LabelSmoothingCE: Cross-entropy with label smoothing to prevent overconfident predictions
- FocalLoss: Focuses learning on hard examples by down-weighting easy ones
- WeightedCE: Cross-entropy with class weights to address imbalanced datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Label smoothing prevents the model from becoming overconfident by distributing
    a small amount of probability mass to incorrect classes.

    Args:
        smoothing: Label smoothing factor (default: 0.1)
            - 0.0: no smoothing (standard cross-entropy)
            - 0.1: target becomes 0.9 for correct class, 0.1/(num_classes-1) for others
    """

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions (batch, num_classes)
            targets: Ground truth labels (batch,) - integer class indices

        Returns:
            loss: Scalar loss value
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # One-hot encode targets
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)

        # Apply label smoothing
        targets_smooth = targets_one_hot * self.confidence + (1 - targets_one_hot) * (self.smoothing / (num_classes - 1))

        # Compute loss
        loss = (-targets_smooth * log_probs).sum(dim=-1).mean()

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal Loss down-weights easy examples and focuses training on hard negatives.
    Particularly effective for datasets with severe class imbalance.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)

    Args:
        alpha: Weighting factor in range [0, 1] to balance positive/negative examples
               Can also be a tensor of per-class weights (default: None)
        gamma: Focusing parameter >= 0 (default: 2.0)
            - gamma=0: equivalent to cross-entropy
            - gamma=2: standard focal loss
            - Higher gamma increases focus on hard examples
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions (batch, num_classes)
            targets: Ground truth labels (batch,) - integer class indices

        Returns:
            loss: Scalar loss value
        """
        # Compute cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Get probabilities for the target class
        probs = F.softmax(logits, dim=-1)
        targets_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal term: (1 - p_t)^gamma
        focal_weight = (1 - targets_probs) ** self.gamma

        # Apply focal weight
        loss = focal_weight * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_weight = self.alpha
            else:
                # Per-class alpha weights
                alpha_weight = self.alpha.gather(0, targets)
            loss = alpha_weight * loss

        return loss.mean()


class WeightedCE(nn.Module):
    """
    Cross-entropy loss with class weights.

    Assigns higher weights to underrepresented classes to address class imbalance.

    Args:
        weights: Tensor of per-class weights (num_classes,)
            Typically computed as inverse frequency or inverse sqrt frequency
    """

    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions (batch, num_classes)
            targets: Ground truth labels (batch,) - integer class indices

        Returns:
            loss: Scalar loss value
        """
        return F.cross_entropy(logits, targets, weight=self.weights)


def compute_class_weights(dataset, method='inverse', device='cpu'):
    """
    Compute class weights from dataset for weighted loss functions.

    Args:
        dataset: PyTorch dataset with labels accessible via dataset.labels
        method: Weight computation method
            - 'inverse': 1 / frequency (emphasizes rare classes heavily)
            - 'inverse_sqrt': 1 / sqrt(frequency) (balanced emphasis)
            - 'effective_samples': (1-beta) / (1-beta^n) for beta=0.9999
        device: Device to place weights tensor on

    Returns:
        weights: Tensor of per-class weights (num_classes,)
    """
    # Count samples per class
    labels = torch.tensor([dataset.labels[i] for i in range(len(dataset))])
    num_classes = len(torch.unique(labels))
    class_counts = torch.bincount(labels, minlength=num_classes).float()

    if method == 'inverse':
        weights = 1.0 / class_counts
    elif method == 'inverse_sqrt':
        weights = 1.0 / torch.sqrt(class_counts)
    elif method == 'effective_samples':
        # Effective number of samples (Cui et al. 2019)
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights to sum to num_classes (keeps loss scale similar to unweighted)
    weights = weights * num_classes / weights.sum()

    return weights.to(device)


def test_losses():
    """Test loss functions with sample data."""
    print("Testing loss functions...")

    batch_size = 4
    num_classes = 18

    # Sample logits and targets
    torch.manual_seed(42)
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    print(f"Logits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Targets: {targets.tolist()}")

    # Test LabelSmoothingCE
    ls_loss = LabelSmoothingCE(smoothing=0.1)
    ls_output = ls_loss(logits, targets)
    print(f"\nLabelSmoothingCE (smoothing=0.1): {ls_output.item():.4f}")

    # Test FocalLoss
    focal_loss = FocalLoss(gamma=2.0)
    focal_output = focal_loss(logits, targets)
    print(f"FocalLoss (gamma=2.0): {focal_output.item():.4f}")

    # Test WeightedCE
    weights = torch.ones(num_classes)
    weights[0] = 2.0  # Give class 0 higher weight
    weighted_ce = WeightedCE(weights=weights)
    wce_output = weighted_ce(logits, targets)
    print(f"WeightedCE (class 0 weight=2.0): {wce_output.item():.4f}")

    # Test standard CE for comparison
    ce_output = F.cross_entropy(logits, targets)
    print(f"Standard CE (comparison): {ce_output.item():.4f}")

    print("\n✓ All loss functions tested successfully!")


if __name__ == "__main__":
    test_losses()
