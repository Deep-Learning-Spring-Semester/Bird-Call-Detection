"""
DEEPNET Hyperparameter Sweep — Days 12-13

Sequential hyperparameter optimisation (not grid search).
Each stage fixes the best value found so far and sweeps the next dimension.

Stages (each run = SWEEP_EPOCHS, no early stopping):
  1. Learning rate:   [0.0005, 0.001, 0.002, 0.005]
  2. Weight decay:    [0.001, 0.01, 0.05]
  3. Dropout:         [0.3, 0.5, 0.7]
  4. Batch size:      [16, 32, 64]
  5. Scheduler:       [cosine, onecycle, plateau]

After the sweep the best config is written to configs/best.yaml and optionally
trained for the full number of epochs with test evaluation.

Usage (from deepnet/):
    # Run full sweep
    uv run python -m deepnet.hparam_sweep

    # Run sweep starting from a specific config
    uv run python -m deepnet.hparam_sweep --base-config src/deepnet/configs/audio_augmented.yaml

    # Sweep specific stages only
    uv run python -m deepnet.hparam_sweep --stages lr dropout scheduler

    # After sweep, train best config for 50 epochs
    uv run python -m deepnet.hparam_sweep --train-best
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import yaml

from deepnet.dataset import build_dataloaders
from deepnet.losses import FocalLoss, LabelSmoothingCE, WeightedCE, compute_class_weights
from deepnet.model import DeepNet
from deepnet.trainer import Trainer
from deepnet.transforms import build_spec_augment
from deepnet.utils import get_device, load_config, set_seed

SWEEP_EPOCHS = 20       # Epochs per sweep run
SWEEP_PATIENCE = 999    # Effectively disable early stopping during sweep


# ---------------------------------------------------------------------------
# Sweep stage definitions
# ---------------------------------------------------------------------------

STAGES = {
    'lr': {
        'values': [0.0005, 0.001, 0.002, 0.005],
        'config_path': ['optimizer', 'lr'],
        'description': 'Learning rate',
        'format': '{:.4f}',
    },
    'weight_decay': {
        'values': [0.001, 0.01, 0.05],
        'config_path': ['optimizer', 'weight_decay'],
        'description': 'Weight decay',
        'format': '{:.3f}',
    },
    'dropout': {
        'values': [0.3, 0.5, 0.7],
        'config_path': ['model', 'dropout'],
        'description': 'Dropout rate',
        'format': '{:.1f}',
    },
    'batch_size': {
        'values': [16, 32],
        'config_path': ['data', 'batch_size'],
        'description': 'Batch size',
        'format': '{}',
    },
    'scheduler': {
        'values': ['cosine', 'onecycle', 'plateau'],
        'config_path': ['optimizer', 'scheduler'],
        'description': 'LR scheduler',
        'format': '{}',
    },
}

DEFAULT_STAGE_ORDER = ['lr', 'weight_decay', 'dropout', 'batch_size', 'scheduler']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_nested(d: dict, path: list, value):
    """Set a value in a nested dict via a key-path list."""
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


def _get_nested(d: dict, path: list):
    """Get a value from a nested dict via a key-path list."""
    for key in path:
        d = d[key]
    return d


def _make_loss(config, device, train_dataset=None):
    loss_type = config['loss']['type']
    if loss_type == 'label_smoothing':
        return LabelSmoothingCE(smoothing=config['loss']['smoothing'])
    elif loss_type == 'focal':
        return FocalLoss(gamma=config['loss']['gamma'])
    elif loss_type == 'weighted_ce':
        method = config['loss'].get('class_weights_method', 'inverse_sqrt')
        weights = compute_class_weights(train_dataset, method=method, device=device)
        return WeightedCE(weights=weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def run_single(
    full_config: dict,
    experiment_name: str,
    device,
    num_epochs: int = SWEEP_EPOCHS,
    patience: int = SWEEP_PATIENCE,
    test_loader=None,
    class_names=None,
) -> dict:
    """
    Run a single training experiment.

    Returns:
        dict with 'val_acc', 'val_loss', 'experiment_name', 'epochs_run'
    """
    config = copy.deepcopy(full_config)
    config['experiment_name'] = experiment_name
    config['training']['num_epochs'] = num_epochs
    config['training']['patience'] = patience

    seed = config['training'].get('seed', 42)
    set_seed(seed)

    # Build dataloaders
    data_config = {
        'batch_size': config['data']['batch_size'],
        'num_workers': config['data']['num_workers'],
        'use_augmented': config['data'].get('use_augmented', True),
        'use_weighted_sampler': config['data'].get('weighted_sampling', True),
        'seed': seed,
    }
    aug_config = config.get('augmentation', {})
    train_transform = build_spec_augment(aug_config)

    train_loader, val_loader, test_loader_built, label_map, _ = build_dataloaders(
        data_config, train_transform=train_transform
    )

    # Use provided test_loader if given, else the freshly built one
    tl = test_loader if test_loader is not None else test_loader_built
    cn = class_names if class_names is not None else sorted(
        label_map.keys(), key=lambda k: label_map[k]
    )

    # Model
    num_classes = config['model']['num_classes']
    dropout = config['model']['dropout']
    model = DeepNet(num_classes=num_classes, dropout=dropout)

    # Loss
    criterion = _make_loss(config, device, train_loader.dataset)

    # Trainer config
    trainer_config = {
        'experiment_name': experiment_name,
        'lr': config['optimizer']['lr'],
        'weight_decay': config['optimizer']['weight_decay'],
        'min_lr': config['optimizer'].get('min_lr', 1e-6),
        'num_epochs': num_epochs,
        'patience': patience,
        'scheduler': config['optimizer'].get('scheduler', 'cosine'),
    }

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
        test_loader=tl if num_epochs >= 50 else None,  # test eval only for full runs
        class_names=cn,
    )

    history = trainer.train()

    return {
        'experiment_name': experiment_name,
        'val_acc': history['best_val_acc'],
        'val_loss': history['best_val_loss'],
        'epochs_run': len(history['train_losses']),
    }


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------

def run_stage(
    stage_name: str,
    base_config: dict,
    device,
    all_results: list,
) -> tuple:
    """
    Run a single sweep stage.

    Returns:
        (best_value, stage_results)
    """
    stage = STAGES[stage_name]
    values = stage['values']
    config_path = stage['config_path']
    description = stage['description']
    fmt = stage['format']

    print(f"\n{'=' * 70}")
    print(f"SWEEP STAGE: {description}")
    print(f"{'=' * 70}")
    current = _get_nested(base_config, config_path)
    print(f"Current value: {current}")
    print(f"Testing: {values}\n")

    stage_results = []

    for value in values:
        config = copy.deepcopy(base_config)
        _set_nested(config, config_path, value)

        val_str = fmt.format(value)
        exp_name = f"sweep_{stage_name}_{val_str}"
        print(f"\n--- {description}={val_str} ---")

        result = run_single(config, exp_name, device)
        result['value'] = value
        result['stage'] = stage_name
        stage_results.append(result)
        all_results.append(result)

        print(f"  Val acc: {result['val_acc']:.2f}%  |  Val loss: {result['val_loss']:.4f}")

    # Print stage summary table
    print(f"\n{description} sweep results:")
    print(f"  {'Value':<15} {'Val Acc':>10} {'Val Loss':>10}")
    print(f"  {'-' * 35}")
    for r in sorted(stage_results, key=lambda x: x['val_acc'], reverse=True):
        marker = " ← best" if r == max(stage_results, key=lambda x: x['val_acc']) else ""
        print(f"  {fmt.format(r['value']):<15} {r['val_acc']:>9.2f}%  {r['val_loss']:>10.4f}{marker}")

    best = max(stage_results, key=lambda x: x['val_acc'])
    return best['value'], stage_results


# ---------------------------------------------------------------------------
# Best config writer
# ---------------------------------------------------------------------------

def write_best_yaml(config: dict, save_path: Path):
    """Write the best config as best.yaml with full 50-epoch training settings."""
    best = copy.deepcopy(config)
    best['experiment_name'] = 'best_v1'
    best['training']['num_epochs'] = 50
    best['training']['patience'] = 15

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(best, f, default_flow_style=False, sort_keys=False)
    print(f"\nBest config written to {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DEEPNET hyperparameter sweep (Days 12-13)")
    parser.add_argument(
        '--base-config',
        type=str,
        default='src/deepnet/configs/audio_augmented.yaml',
        help='Base config to start from (default: audio_augmented.yaml)',
    )
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=list(STAGES.keys()),
        default=DEFAULT_STAGE_ORDER,
        help='Which stages to run (default: all in order)',
    )
    parser.add_argument(
        '--sweep-epochs',
        type=int,
        default=SWEEP_EPOCHS,
        help=f'Epochs per sweep run (default: {SWEEP_EPOCHS})',
    )
    parser.add_argument(
        '--train-best',
        action='store_true',
        help='After sweep, train best config for full 50 epochs with test evaluation',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'mps'],
        help='Device (default: auto-detect)',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\nLoading base config: {args.base_config}")
    base_config = load_config(args.base_config)

    # Ensure optimizer.scheduler key exists
    if 'scheduler' not in base_config.get('optimizer', {}):
        base_config['optimizer']['scheduler'] = 'cosine'

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print(f"Device: {device}")
    print(f"Sweep epochs per run: {args.sweep_epochs}")
    print(f"Stages: {args.stages}")

    best_config = copy.deepcopy(base_config)
    all_results = []

    # Run each stage sequentially
    for stage_name in args.stages:
        best_value, _ = run_stage(
            stage_name, best_config, device, all_results
        )
        # Apply best value to running config
        _set_nested(best_config, STAGES[stage_name]['config_path'], best_value)
        print(f"\nBest {STAGES[stage_name]['description']}: {best_value}")

    # Print final summary
    print(f"\n{'=' * 70}")
    print("SWEEP COMPLETE — Final best hyperparameters:")
    print(f"{'=' * 70}")
    for stage_name in args.stages:
        path = STAGES[stage_name]['config_path']
        value = _get_nested(best_config, path)
        fmt = STAGES[stage_name]['format']
        print(f"  {STAGES[stage_name]['description']:<20}: {fmt.format(value)}")

    # Print all results sorted by val_acc
    print(f"\n{'All sweep runs (sorted by val acc)':}")
    print(f"  {'Experiment':<45} {'Val Acc':>10} {'Val Loss':>10}")
    print(f"  {'-' * 65}")
    for r in sorted(all_results, key=lambda x: x['val_acc'], reverse=True):
        print(f"  {r['experiment_name']:<45} {r['val_acc']:>9.2f}%  {r['val_loss']:>10.4f}")

    # Save all sweep results JSON
    results_path = Path('results') / 'sweep_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSweep results saved to {results_path}")

    # Write best.yaml
    best_yaml_path = Path('src/deepnet/configs/best.yaml')
    write_best_yaml(best_config, best_yaml_path)

    # Optionally train best config for full epochs
    if args.train_best:
        print(f"\n{'=' * 70}")
        print("TRAINING BEST CONFIG (50 epochs, with test evaluation)...")
        print(f"{'=' * 70}")

        result = run_single(
            best_config,
            experiment_name='best_v1',
            device=device,
            num_epochs=50,
            patience=15,
        )
        print(f"\nFinal best model: val_acc={result['val_acc']:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
