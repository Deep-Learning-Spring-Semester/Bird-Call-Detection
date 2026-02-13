"""
Training infrastructure for DEEPNET bird call classifier.

Trainer class handles:
- Training and validation loops with progress bars
- Optimizer and learning rate scheduling
- Early stopping to prevent overfitting
- TensorBoard logging for experiment tracking
- Checkpoint saving and loading
"""

import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    """
    Trainer for bird call classification model.

    Manages the full training lifecycle including optimization, validation,
    early stopping, logging, and checkpointing.
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
        log_dir="runs"
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
            device: Device to train on ('cuda', 'mps', or 'cpu')
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.device = device

        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.01)
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )

        # Early stopping
        self.patience = config.get('patience', 10)
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) / config['experiment_name']
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard logging
        self.log_dir = Path(log_dir) / config['experiment_name']
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

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

        Returns:
            history: Dict with training history (losses, accuracies)
        """
        num_epochs = self.config['num_epochs']
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"TensorBoard logs: {self.log_dir}")
        print(f"Early stopping patience: {self.patience}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Learning rate step
            current_lr = self.optimizer.param_groups[0]['lr']
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

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': max(self.val_accuracies)
        }

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
