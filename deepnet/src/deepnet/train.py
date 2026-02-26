"""
DEEPNET Training Script

CLI entry point for training bird call classification models.

Usage:
    uv run python -m deepnet.train --config src/deepnet/configs/baseline.yaml
    uv run python -m deepnet.train --config src/deepnet/configs/baseline.yaml --resume checkpoints/baseline_v1/best.pt
"""

import argparse
import sys
from pathlib import Path

import torch

from deepnet.dataset import build_dataloaders
from deepnet.losses import FocalLoss, LabelSmoothingCE, WeightedCE, compute_class_weights
from deepnet.model import DeepNet
from deepnet.trainer import Trainer
from deepnet.transforms import build_spec_augment
from deepnet.utils import get_device, load_config, set_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train DEEPNET bird call classifier")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to train on (default: auto-detect)"
    )
    return parser.parse_args()


def setup_loss_function(config, device, train_dataset=None):
    """
    Set up loss function based on config.

    Args:
        config: Configuration dictionary
        device: Device to place loss on
        train_dataset: Training dataset (needed for weighted loss)

    Returns:
        criterion: Loss function
    """
    loss_type = config['loss']['type']

    if loss_type == 'label_smoothing':
        smoothing = config['loss']['smoothing']
        criterion = LabelSmoothingCE(smoothing=smoothing)
        print(f"Loss: LabelSmoothingCE (smoothing={smoothing})")

    elif loss_type == 'focal':
        gamma = config['loss']['gamma']
        criterion = FocalLoss(gamma=gamma)
        print(f"Loss: FocalLoss (gamma={gamma})")

    elif loss_type == 'weighted_ce':
        if train_dataset is None:
            raise ValueError("train_dataset required for weighted CE loss")
        method = config['loss'].get('class_weights_method', 'inverse_sqrt')
        weights = compute_class_weights(train_dataset, method=method, device=device)
        criterion = WeightedCE(weights=weights)
        print(f"Loss: WeightedCE (method={method})")
        print(f"Class weights range: [{weights.min():.3f}, {weights.max():.3f}]")

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return criterion


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    print(f"\nLoading config from {args.config}...")
    config = load_config(args.config)
    print(f"Experiment: {config['experiment_name']}")

    # Set random seed
    seed = config['training'].get('seed', 42)
    set_seed(seed)

    # Get device
    if args.device:
        device = torch.device(args.device)
        print(f"Using specified device: {device}")
    else:
        device = get_device()

    # Build dataloaders
    print("\nBuilding dataloaders...")

    # Flatten config for build_dataloaders
    data_config = {
        'batch_size': config['data']['batch_size'],
        'num_workers': config['data']['num_workers'],
        'use_augmented': config['data'].get('use_augmented', False),
        'use_weighted_sampler': config['data'].get('weighted_sampling', True),
        'seed': config['training'].get('seed', 42)
    }

    # Build spectrogram transform for training (val/test are never augmented)
    aug_config = config.get('augmentation', {})
    train_transform = build_spec_augment(aug_config)
    if train_transform is not None:
        print(f"SpecAugment enabled: freq_mask={aug_config.get('freq_mask_param', 15)}, "
              f"time_mask={aug_config.get('time_mask_param', 25)}, "
              f"num_masks={aug_config.get('num_masks', 2)}")
    else:
        print("No spectrogram augmentation")

    train_loader, val_loader, test_loader, label_map, class_weights = build_dataloaders(
        data_config, train_transform=train_transform
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Number of classes: {len(label_map)}")

    # Create model
    print("\nCreating model...")
    num_classes = config['model']['num_classes']
    dropout = config['model']['dropout']
    model = DeepNet(num_classes=num_classes, dropout=dropout)
    print(f"Model: DeepNet ({model.count_parameters():,} parameters)")

    # Setup loss function
    print("\nSetting up loss function...")
    criterion = setup_loss_function(config, device, train_dataset=train_loader.dataset)

    # Flatten config for trainer
    trainer_config = {
        'experiment_name': config['experiment_name'],
        'lr': config['optimizer']['lr'],
        'weight_decay': config['optimizer']['weight_decay'],
        'min_lr': config['optimizer'].get('min_lr', 1e-6),
        'num_epochs': config['training']['num_epochs'],
        'patience': config['training']['patience'],
        'scheduler': config['optimizer'].get('scheduler', 'cosine'),
    }

    # Class names sorted by label index (for test evaluation report)
    class_names = sorted(label_map.keys(), key=lambda k: label_map[k])

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=trainer_config,
        device=device,
        checkpoint_dir=config['paths'].get('checkpoints', 'checkpoints'),
        log_dir=config['paths'].get('logs', 'runs'),
        results_dir=config['paths'].get('results', 'results'),
        test_loader=test_loader,
        class_names=class_names,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    try:
        history = trainer.train()

        # Print final summary
        print("\n" + "=" * 80)
        print("Training Summary")
        print("=" * 80)
        print(f"Best validation loss: {history['best_val_loss']:.4f}")
        print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
        print(f"Final checkpoint: {trainer.checkpoint_dir / 'best.pt'}")
        print(f"Training curves:  {trainer.results_dir / 'training_curves.png'}")
        print(f"Test metrics:     {trainer.results_dir / 'test_metrics.json'}")
        print(f"\nView training progress:")
        print(f"  tensorboard --logdir {trainer.log_dir.parent}")
        print("=" * 80)

        return 0

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print(f"Latest checkpoint saved to: {trainer.checkpoint_dir}")
        return 1

    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
