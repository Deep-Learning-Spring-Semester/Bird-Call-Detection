"""
Utility functions for DEEPNET training.

Includes:
- Device selection (MPS/CUDA/CPU)
- Random seed setting for reproducibility
- Checkpoint I/O helpers
- Configuration loading
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def get_device(prefer_mps=True):
    """
    Get the best available device for PyTorch.

    Priority: MPS (Apple Silicon) → CUDA (NVIDIA GPU) → CPU

    Args:
        prefer_mps: Whether to prefer MPS over CUDA if both available

    Returns:
        device: torch.device object
    """
    if prefer_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    print(f"Random seed set to {seed}")


def load_config(config_path):
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        config: Dictionary of configuration values
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def flatten_config(config, parent_key='', sep='_'):
    """
    Flatten nested configuration dictionary.

    Converts {'training': {'lr': 0.001}} to {'training_lr': 0.001}

    Args:
        config: Nested configuration dictionary
        parent_key: Parent key for recursion
        sep: Separator for nested keys

    Returns:
        flat_config: Flattened dictionary
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint on

    Returns:
        checkpoint: Full checkpoint dictionary with metadata
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"✓ Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")

    return checkpoint


def save_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, epoch=None, metrics=None, config=None):
    """
    Save model checkpoint.

    Args:
        checkpoint_path: Path to save checkpoint
        model: Model to save
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        epoch: Current epoch number
        metrics: Dictionary of metrics to save
        config: Configuration dictionary
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics or {},
        'config': config or {}
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Create parent directory if needed
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved to {checkpoint_path}")


def count_parameters(model, trainable_only=True):
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters

    Returns:
        num_params: Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_time(seconds):
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        time_str: Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


if __name__ == "__main__":
    # Test utilities
    print("Testing DEEPNET utilities...\n")

    # Test device selection
    device = get_device()
    print(f"Selected device: {device}\n")

    # Test seed setting
    set_seed(42)

    # Test config loading (if baseline.yaml exists)
    config_path = Path(__file__).parent / "configs" / "baseline.yaml"
    if config_path.exists():
        config = load_config(config_path)
        print(f"\nLoaded config: {config['experiment_name']}")
        flat_config = flatten_config(config)
        print(f"Flattened config keys: {list(flat_config.keys())[:5]}...")

    print("\n✓ Utilities test complete!")
