"""PyTorch Dataset and DataLoader factory for bird call spectrograms.

Loads pre-computed mel-spectrogram .pt tensors, performs stratified
train/val/test splits, and builds DataLoaders with weighted sampling.

Usage (from deepnet/):
    uv run python -m deepnet.dataset          # prints split stats
"""

import json
from pathlib import Path

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import torch.backends.mps

from deepnet.utils import DATA_ROOT, CONFIGS_DIR, setup_logging

MEL_TENSORS_DIR = DATA_ROOT / "processed" / "mel_tensors"
CLEANED_TENSORS_DIR = MEL_TENSORS_DIR / "cleaned"
AUGMENTED_TENSORS_DIR = MEL_TENSORS_DIR / "augmented"

EXCLUDE_SPECIES = {"Identity_unknown"}

log = setup_logging("dataset")


def build_label_map(species_dirs: list[Path]) -> dict[str, int]:
    """Build sorted species → index mapping, save to label_map.json."""
    species_names = sorted(
        d.name for d in species_dirs if d.name not in EXCLUDE_SPECIES
    )
    label_map = {name: idx for idx, name in enumerate(species_names)}

    out_path = CONFIGS_DIR / "label_map.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(label_map, f, indent=2)
    log.info(f"Label map saved: {len(label_map)} species → {out_path}")

    return label_map


def load_label_map() -> dict[str, int]:
    """Load existing label map from configs/label_map.json."""
    path = CONFIGS_DIR / "label_map.json"
    with open(path) as f:
        return json.load(f)


class BirdCallDataset(Dataset):
    """Dataset that loads pre-computed mel-spectrogram .pt tensors.

    Each item returns (spectrogram, label) where spectrogram is
    shape (1, 128, 216) and label is an integer class index.
    """

    def __init__(
        self,
        file_paths: list[Path],
        labels: list[int],
        transform=None,
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        spec = torch.load(self.file_paths[idx], weights_only=True)
        label = self.labels[idx]
        if self.transform:
            spec = self.transform(spec)
        return spec, label


def _scan_tensors(tensor_dir: Path, label_map: dict[str, int]) -> tuple[list[Path], list[int]]:
    """Scan a tensor directory and return (paths, labels) for known species."""
    paths = []
    labels = []
    for species_dir in sorted(tensor_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        species = species_dir.name
        if species not in label_map:
            continue
        label = label_map[species]
        for pt_file in sorted(species_dir.glob("*.pt")):
            paths.append(pt_file)
            labels.append(label)
    return paths, labels


def build_dataloaders(
    config: dict | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int], torch.Tensor]:
    """Build train/val/test DataLoaders with stratified splitting.

    Args:
        config: Optional dict with keys:
            - batch_size (int, default 32)
            - num_workers (int, default 4)
            - val_ratio (float, default 0.15)
            - test_ratio (float, default 0.15)
            - use_augmented (bool, default True) — add augmented data to train
            - use_weighted_sampler (bool, default True)
            - seed (int, default 42)

    Returns:
        (train_loader, val_loader, test_loader, label_map, class_weights)
    """
    config = config or {}
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    val_ratio = config.get("val_ratio", 0.15)
    test_ratio = config.get("test_ratio", 0.15)
    use_augmented = config.get("use_augmented", True)
    use_weighted_sampler = config.get("use_weighted_sampler", True)
    seed = config.get("seed", 42)

    # Build or load label map
    species_dirs = [d for d in CLEANED_TENSORS_DIR.iterdir() if d.is_dir()]
    label_map = build_label_map(species_dirs)
    num_classes = len(label_map)

    # Scan cleaned tensors (used for all splits)
    clean_paths, clean_labels = _scan_tensors(CLEANED_TENSORS_DIR, label_map)
    log.info(f"Cleaned tensors: {len(clean_paths)} files, {num_classes} classes")

    # Stratified split: first split off test, then split remainder into train/val
    splitter_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    remaining_idx, test_idx = next(
        splitter_test.split(clean_paths, clean_labels)
    )

    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
    splitter_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio_adjusted, random_state=seed
    )
    remaining_labels = [clean_labels[i] for i in remaining_idx]
    remaining_paths = [clean_paths[i] for i in remaining_idx]
    train_sub_idx, val_sub_idx = next(
        splitter_val.split(remaining_paths, remaining_labels)
    )

    train_idx = [remaining_idx[i] for i in train_sub_idx]
    val_idx = [remaining_idx[i] for i in val_sub_idx]

    # Build path/label lists for each split
    train_paths = [clean_paths[i] for i in train_idx]
    train_labels = [clean_labels[i] for i in train_idx]
    val_paths = [clean_paths[i] for i in val_idx]
    val_labels = [clean_labels[i] for i in val_idx]
    test_paths = [clean_paths[i] for i in test_idx]
    test_labels = [clean_labels[i] for i in test_idx]

    # Optionally add augmented data to training set only
    if use_augmented and AUGMENTED_TENSORS_DIR.exists():
        # Only include augmented versions of training set files
        train_clean_stems = {p.stem for p in train_paths}
        aug_paths, aug_labels = _scan_tensors(AUGMENTED_TENSORS_DIR, label_map)

        # Match augmented files to their clean originals by checking if the
        # clean stem (after removing the aug_ prefix) is in the training set
        added = 0
        for ap, al in zip(aug_paths, aug_labels):
            # Augmented files are named like aug_noise_XC12345.pt
            # The corresponding clean file is clean_XC12345.pt
            aug_stem = ap.stem
            # Strip the augmentation prefix to get the original recording name
            # Format: aug_{type}_{original_name}
            parts = aug_stem.split("_", 2)
            if len(parts) >= 3:
                original_stem = "clean_" + parts[2]
            else:
                original_stem = aug_stem
            if original_stem in train_clean_stems:
                train_paths.append(ap)
                train_labels.append(al)
                added += 1
        log.info(f"Added {added} augmented tensors to training set")

    # Compute class weights (inverse frequency)
    class_counts = torch.zeros(num_classes)
    for lbl in train_labels:
        class_counts[lbl] += 1
    class_weights = 1.0 / torch.clamp(class_counts, min=1.0)
    class_weights = class_weights / class_weights.sum() * num_classes

    log.info(f"Split sizes — train: {len(train_paths)}, val: {len(val_paths)}, test: {len(test_paths)}")
    log.info(f"Class counts (train): min={int(class_counts.min())}, max={int(class_counts.max())}")

    # Create datasets
    train_dataset = BirdCallDataset(train_paths, train_labels)
    val_dataset = BirdCallDataset(val_paths, val_labels)
    test_dataset = BirdCallDataset(test_paths, test_labels)

    # Weighted sampler for training
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sample_weights = [class_weights[lbl].item() for lbl in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True,
        )
        shuffle = False  # mutually exclusive with sampler

    # pin_memory only useful for CUDA, not MPS
    pin = not torch.backends.mps.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader, label_map, class_weights


if __name__ == "__main__":
    train_loader, val_loader, test_loader, label_map, class_weights = build_dataloaders()

    print(f"\nLabel map ({len(label_map)} classes):")
    for species, idx in label_map.items():
        print(f"  {idx:2d}: {species}")

    print(f"\nClass weights: {class_weights}")

    # Sample a batch
    specs, labels = next(iter(train_loader))
    print(f"\nSample batch — specs: {specs.shape}, labels: {labels.shape}")
    print(f"  spec range: [{specs.min():.2f}, {specs.max():.2f}]")
    print(f"  labels: {labels[:8].tolist()}")
