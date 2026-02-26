"""
Training infrastructure for DEEPNET bird call classifier.

Trainer class handles:
- Training and validation loops with progress bars
- Optimizer and learning rate scheduling (Cosine, OneCycle, ReduceLROnPlateau)
- Early stopping to prevent overfitting
- TensorBoard logging for experiment tracking
- Checkpoint saving and loading
- Training curve plots saved at end of every run
- Automatic test set evaluation after training completes
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from deepnet.metrics import MetricsTracker


class Trainer:
    """
    Trainer for bird call classification model.

    Manages the full training lifecycle including optimization, validation,
    early stopping, logging, checkpointing, training curve plots, and
    final test set evaluation.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        config,
        device,
        checkpoint_dir="checkpoints",
        log_dir="runs",
        results_dir="results",
        test_loader=None,
        class_names=None,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            config: Training configuration dict with keys:
                - lr: Learning rate
                - weight_decay: L2 regularization
                - num_epochs: Maximum training epochs
                - patience: Early stopping patience
                - experiment_name: Name for logging/checkpoints
                - scheduler: 'cosine' (default), 'onecycle', or 'plateau'
            device: Device to train on ('cuda', 'mps', or 'cpu')
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            results_dir: Directory to save plots and metrics JSON
            test_loader: Optional test DataLoader — evaluated at end of training
            class_names: List of class names in label-index order (for test eval)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.device = device
        self.test_loader = test_loader
        self.class_names = class_names

        lr = config['lr']
        num_epochs = config['num_epochs']
        min_lr = config.get('min_lr', 1e-6)
        weight_decay = config.get('weight_decay', 0.01)
        scheduler_type = config.get('scheduler', 'cosine')

        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Setup scheduler
        self.scheduler_type = scheduler_type
        self.scheduler_per_batch = False

        if scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=lr,
                steps_per_epoch=len(train_loader),
                epochs=num_epochs,
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1e4,
            )
            self.scheduler_per_batch = True
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=5,
                factor=0.5,
                min_lr=min_lr,
            )
        else:  # default: cosine
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=min_lr,
            )

        # Early stopping
        self.patience = config.get('patience', 10)
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) / config['experiment_name']
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Results directory for plots and metrics
        self.results_dir = Path(results_dir) / config['experiment_name']
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard logging
        self.log_dir = Path(log_dir) / config['experiment_name']
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.lr_history = []

    def train_epoch(self):
        """
        Train for one epoch.

        Returns:
            avg_loss: Average training loss for the epoch
            accuracy: Training accuracy for the epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # OneCycleLR steps every batch
            if self.scheduler_per_batch:
                self.scheduler.step()

            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%'
            })

            # Log to TensorBoard (every 10 batches)
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/batch_loss', loss.item(), self.global_step)

            self.global_step += 1

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self):
        """
        Validate the model on validation set.

        Returns:
            avg_loss: Average validation loss
            accuracy: Validation accuracy
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]  ")

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Track metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                avg_loss = running_loss / (batch_idx + 1)
                accuracy = 100.0 * correct / total
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{accuracy:.2f}%'
                })

        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self):
        """
        Full training loop with early stopping and checkpointing.

        At the end of training:
        - Saves training/validation curve plots to results/
        - Runs test set evaluation if test_loader was provided

        Returns:
            history: Dict with training history (losses, accuracies)
        """
        num_epochs = self.config['num_epochs']
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Scheduler: {self.scheduler_type}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"TensorBoard logs: {self.log_dir}")
        print(f"Early stopping patience: {self.patience}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Epoch-level LR step (not for OneCycleLR which steps per batch)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)

            if not self.scheduler_per_batch:
                if self.scheduler_type == 'plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log to TensorBoard
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/train_acc', train_acc, epoch)
            self.writer.add_scalar('epoch/val_acc', val_acc, epoch)
            self.writer.add_scalar('epoch/lr', current_lr, epoch)

            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1}/{num_epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                self.save_checkpoint(is_best=True, metrics={'val_loss': val_loss, 'val_acc': val_acc})
                print(f"  ✓ New best model! Val loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epoch(s)")

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(is_best=False, metrics={'val_loss': val_loss, 'val_acc': val_acc})

            print()

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {max(self.val_accuracies):.2f}%")

        self.writer.close()

        # Save training curve plots
        self._save_training_plots()

        # Run test set evaluation at end of training
        if self.test_loader is not None:
            self._run_test_evaluation()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': max(self.val_accuracies),
        }

    def _save_training_plots(self):
        """Save training/validation loss and accuracy curves as PNG."""
        save_path = self.results_dir / 'training_curves.png'
        epochs = range(1, len(self.train_losses) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss curves
        ax = axes[0]
        ax.plot(epochs, self.train_losses, 'b-o', label='Train', linewidth=2, markersize=3)
        ax.plot(epochs, self.val_losses, 'r-o', label='Val', linewidth=2, markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss Curves', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Accuracy curves
        ax = axes[1]
        ax.plot(epochs, self.train_accuracies, 'b-o', label='Train', linewidth=2, markersize=3)
        ax.plot(epochs, self.val_accuracies, 'r-o', label='Val', linewidth=2, markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy Curves', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Learning rate curve
        ax = axes[2]
        ax.plot(epochs, self.lr_history, 'g-o', linewidth=2, markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=13)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        experiment_name = self.config['experiment_name']
        best_val = max(self.val_accuracies)
        fig.suptitle(
            f"Training Curves — {experiment_name}  |  Best Val Acc: {best_val:.1f}%",
            fontsize=14,
            fontweight='bold',
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Training curves saved → {save_path}")

    def _run_test_evaluation(self):
        """Evaluate on test set, print metrics, and save results to results dir."""
        print("\n" + "=" * 70)
        print("TEST SET EVALUATION (end of training)")
        print("=" * 70)

        # Determine class names
        num_classes = self.model.classifier[-1].out_features
        class_names = self.class_names or [f"Class_{i}" for i in range(num_classes)]

        self.model.eval()
        tracker = MetricsTracker(num_classes=len(class_names), class_names=class_names)

        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Test evaluation"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                tracker.update(outputs, targets)

        metrics = tracker.compute()

        # Print summary
        print(f"\n  Top-1 Accuracy:  {metrics['top1_accuracy']:.2f}%")
        print(f"  Top-3 Accuracy:  {metrics['top3_accuracy']:.2f}%")
        print(f"  Top-5 Accuracy:  {metrics['top5_accuracy']:.2f}%")
        print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")
        print(f"  Test samples:    {metrics['num_samples']}")

        print("\n  Hardest classes (lowest F1):")
        for cls in metrics['hardest_classes']:
            print(f"    {cls['class_name']:<30} F1: {cls['f1']:.3f}  "
                  f"(P: {cls['precision']:.3f}, R: {cls['recall']:.3f}, N: {cls['support']})")

        print("\n  Most confused pairs:")
        for true_cls, pred_cls, count, pct in metrics['confused_pairs'][:5]:
            print(f"    {true_cls:<25} → {pred_cls:<25} {count:3d} times ({pct:.1f}%)")

        # Save metrics JSON
        metrics_path = self.results_dir / 'test_metrics.json'
        metrics_json = {
            'experiment': self.config['experiment_name'],
            'top1_accuracy': float(metrics['top1_accuracy']),
            'top3_accuracy': float(metrics['top3_accuracy']),
            'top5_accuracy': float(metrics['top5_accuracy']),
            'macro_f1': float(metrics['macro_f1']),
            'weighted_f1': float(metrics['weighted_f1']),
            'num_samples': int(metrics['num_samples']),
            'per_class': [
                {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                 for k, v in cls.items()}
                for cls in metrics['per_class']
            ],
            'confused_pairs': [
                {'true_class': tc, 'predicted_class': pc, 'count': int(cnt), 'percentage': float(pct)}
                for tc, pc, cnt, pct in metrics['confused_pairs']
            ],
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f"\n  Test metrics saved → {metrics_path}")

        # Save confusion matrix plot
        self._save_confusion_matrix(metrics['confusion_matrix'], class_names)

        print("=" * 70)

    def _save_confusion_matrix(self, cm, class_names):
        """Save normalized confusion matrix heatmap."""
        try:
            import seaborn as sns
        except ImportError:
            print("  seaborn not installed — skipping confusion matrix plot")
            return

        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        # Short names for display
        short_names = [n.replace('_', ' ') for n in class_names]

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=short_names,
            yticklabels=short_names,
            cbar_kws={'label': 'Fraction'},
            ax=ax,
            linewidths=0.5,
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(
            f"Confusion Matrix (Normalized) — {self.config['experiment_name']}",
            fontsize=13,
            pad=15,
        )
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        save_path = self.results_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Confusion matrix saved → {save_path}")

    def save_checkpoint(self, is_best=False, metrics=None):
        """
        Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far
            metrics: Dict of metrics to save with checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'metrics': metrics or {}
        }

        if is_best:
            path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, path)
            print(f"  Saved best checkpoint to {path}")
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch + 1}.pt'
            torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Resumed from epoch {self.current_epoch + 1}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


if __name__ == "__main__":
    # Quick test of trainer initialization
    print("Trainer module loaded successfully!")
    print("Use train.py to run training experiments.")
