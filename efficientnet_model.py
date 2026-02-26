"""
Bird call spectrogram classifier — trained FROM SCRATCH (no pretrained weights).

Why the architecture is completely different from the EfficientNet transfer learning version:
- No ImageNet weights means the model must learn ALL features from your 2,109 images.
- With ~33 samples/class across 63 classes, a deep model will overfit immediately.
- Solution: a compact custom CNN with very aggressive regularization, trained with
  heavy augmentation and a long warm cosine LR schedule.

Architecture: Custom lightweight CNN (not EfficientNet)
- 4 conv blocks with increasing filters: 32 → 64 → 128 → 256
- Each block: Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool → Dropout
- Global Average Pooling (much better than Flatten for small data — fewer parameters)
- Slim head: 256 → num_classes
- Total params: ~2M (vs 8M for EfficientNetB2 — far more appropriate for this dataset size)

Training strategy:
- Warmup + cosine decay LR schedule
- Very heavy augmentation (mixup + standard transforms)
- High dropout throughout (0.3 in conv blocks, 0.5 in head)
- L2 regularization on all Conv and Dense layers
- Label smoothing 0.1
- Long training: 300 epochs with early stopping patience=40
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse
from datetime import datetime
import pickle

# --- CONFIGURATION ---
SPECTROGRAM_DIR = Path("data/processed/spectrograms/cleaned")
AUGMENTED_SPEC_DIR = Path("data/processed/spectrograms/augmented")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

IMG_SIZE = (128, 128)    # Smaller than 260x260 — less parameters, faster training,
                          # and 128px is plenty of resolution for spectrograms
BATCH_SIZE = 16
EPOCHS = 300
WARMUP_EPOCHS = 10
INITIAL_LR = 1e-3
MIN_LR = 1e-6
L2_REG = 2e-4


# ------------------------------------------------------------------ #
# DATA LOADING
# ------------------------------------------------------------------ #

def load_spectrogram_dataset(spec_dir: Path, use_augmented: bool = False,
                              max_per_class: int = None):
    images, labels = [], []
    print(f"Loading spectrograms from {spec_dir}...")
    if not spec_dir.exists():
        raise FileNotFoundError(f"Not found: {spec_dir}")

    species_folders = sorted([f for f in spec_dir.iterdir() if f.is_dir()])
    if not species_folders:
        raise ValueError(f"No species folders in {spec_dir}")
    print(f"Found {len(species_folders)} species")

    for folder in species_folders:
        files = list(folder.glob("*.png"))
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            images.append(str(f))
            labels.append(folder.name)
        print(f"  {folder.name}: {len(files)} spectrograms")

    if use_augmented and AUGMENTED_SPEC_DIR.exists():
        print(f"\nLoading augmented spectrograms from {AUGMENTED_SPEC_DIR}...")
        for folder in sorted(AUGMENTED_SPEC_DIR.iterdir()):
            if not folder.is_dir():
                continue
            files = list(folder.glob("*.png"))
            if max_per_class:
                files = files[:max_per_class]
            for f in files:
                images.append(str(f))
                labels.append(folder.name)
            print(f"  {folder.name}: {len(files)} augmented")

    class_names = sorted(set(labels))
    print(f"\nTotal: {len(images)} samples, {len(class_names)} classes")
    return images, labels, class_names


def load_image(path: str):
    """Load and normalize to [0, 1]. No pretrained preprocessing needed."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img


# ------------------------------------------------------------------ #
# AUGMENTATION
# ------------------------------------------------------------------ #

def augment(image):
    """
    Strong augmentation for from-scratch training.
    From-scratch models need more augmentation variety than fine-tuned models.
    """
    # Geometric
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)   # Frequency flip — OK for some bird calls

    # Photometric
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)

    # Random crop + resize (zoom simulation)
    h, w = IMG_SIZE
    crop_h = tf.random.uniform([], int(h * 0.80), h, dtype=tf.int32)
    crop_w = tf.random.uniform([], int(w * 0.80), w, dtype=tf.int32)
    offset_h = tf.random.uniform([], 0, h - crop_h + 1, dtype=tf.int32)
    offset_w = tf.random.uniform([], 0, w - crop_w + 1, dtype=tf.int32)
    image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, crop_h, crop_w)
    image = tf.image.resize(image, IMG_SIZE)

    # SpecAugment: time masking (horizontal band)
    t_width = tf.random.uniform([], 5, 25, dtype=tf.int32)
    t_start = tf.random.uniform([], 0, w - t_width, dtype=tf.int32)
    paddings = tf.stack([
        tf.constant([0, 0]),           # height: no pad
        tf.stack([t_start, w - t_start - t_width]),  # width: mask band
        tf.constant([0, 0])            # channels
    ])
    left  = image[:, :t_start, :]
    mid   = tf.zeros([h, t_width, 3])
    right = image[:, t_start + t_width:, :]
    image = tf.concat([left, mid, right], axis=1)

    # SpecAugment: frequency masking (vertical band)
    f_width = tf.random.uniform([], 5, 20, dtype=tf.int32)
    f_start = tf.random.uniform([], 0, h - f_width, dtype=tf.int32)
    top    = image[:f_start, :, :]
    mid_f  = tf.zeros([f_width, w, 3])
    bottom = image[f_start + f_width:, :, :]
    image  = tf.concat([top, mid_f, bottom], axis=0)

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def mixup(images, labels, alpha=0.3):
    """
    Mixup augmentation — blends pairs of training examples.
    Especially powerful for from-scratch training on small datasets.
    """
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform([], alpha, 1.0)
    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
    return mixed_images, mixed_labels


def create_dataset(image_paths, labels, num_classes, batch_size=BATCH_SIZE,
                   shuffle=True, augment_data=False, use_mixup=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda x, y: (load_image(x), tf.one_hot(y, num_classes)),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment_data:
        dataset = dataset.map(
            lambda x, y: (augment(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size)

    if use_mixup:
        dataset = dataset.map(
            lambda x, y: mixup(x, y, alpha=0.3),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    return dataset.prefetch(tf.data.AUTOTUNE)


# ------------------------------------------------------------------ #
# MODEL — Custom lightweight CNN (no pretrained weights)
# ------------------------------------------------------------------ #

def conv_block(x, filters, dropout_rate, l2):
    """Two conv layers + BN + MaxPool + Dropout."""
    x = layers.Conv2D(filters, 3, padding='same',
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, padding='same',
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def build_scratch_model(num_classes: int, dropout_conv: float = 0.3,
                         dropout_head: float = 0.5, l2: float = L2_REG):
    """
    Compact custom CNN trained from scratch.

    Filter progression: 32 → 64 → 128 → 256
    Input 128x128 → 64 → 32 → 16 → 8 → GAP → Dense(256) → softmax

    ~2M parameters — appropriate for 2,109 training samples.
    """
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = inputs

    # Block 1: learn basic edges/textures
    x = conv_block(x, filters=32,  dropout_rate=dropout_conv * 0.5, l2=l2)
    # Block 2: learn frequency/time patterns
    x = conv_block(x, filters=64,  dropout_rate=dropout_conv * 0.75, l2=l2)
    # Block 3: learn call shapes
    x = conv_block(x, filters=128, dropout_rate=dropout_conv, l2=l2)
    # Block 4: high-level call features
    x = conv_block(x, filters=256, dropout_rate=dropout_conv, l2=l2)

    # Global Average Pooling — 8x8x256 → 256. Much better than Flatten for small data.
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu',
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_head)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


# ------------------------------------------------------------------ #
# LR SCHEDULE: linear warmup → cosine decay
# ------------------------------------------------------------------ #

class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup for `warmup_steps`, then cosine decay to `min_lr`.
    Warmup prevents early large gradients from destroying random init.
    """
    def __init__(self, initial_lr, min_lr, warmup_steps, total_steps):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        initial = tf.cast(self.initial_lr, tf.float32)
        minimum = tf.cast(self.min_lr, tf.float32)

        # Linear warmup
        warmup_lr = initial * (step / warmup)

        # Cosine decay after warmup
        progress = (step - warmup) / tf.maximum(total - warmup, 1.0)
        cosine_lr = minimum + 0.5 * (initial - minimum) * (
            1.0 + tf.cos(tf.constant(3.14159265358979) * progress)
        )

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            'initial_lr': self.initial_lr,
            'min_lr': self.min_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps
        }


# ------------------------------------------------------------------ #
# CALLBACKS / PLOTTING / SAVING
# ------------------------------------------------------------------ #

def get_callbacks(exp_dir: Path, patience: int = 40):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(exp_dir / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename=str(exp_dir / 'training_log.csv'),
            append=False
        ),
    ]


def plot_training_history(history, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, train_key, val_key, title, ylabel in [
        (axes[0], 'accuracy', 'val_accuracy', 'Accuracy', 'Accuracy'),
        (axes[1], 'loss',     'val_loss',     'Loss',     'Loss'),
    ]:
        ax.plot(history.history[train_key], label='Train')
        ax.plot(history.history[val_key],   label='Validation')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def save_info(class_names, label_encoder, history, exp_dir: Path):
    info = {
        'architecture': 'custom_scratch_cnn',
        'pretrained': False,
        'num_classes': len(class_names),
        'class_names': class_names,
        'img_size': list(IMG_SIZE),
        'batch_size': BATCH_SIZE,
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(exp_dir / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    with open(exp_dir / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Info + label encoder saved to {exp_dir}")


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #

def main():
    tf.keras.backend.clear_session()

    parser = argparse.ArgumentParser(
        description="Train a from-scratch CNN for bird call spectrogram classification"
    )
    parser.add_argument("--spec-dir",      type=Path,  default=SPECTROGRAM_DIR)
    parser.add_argument("--use-augmented", action="store_true")
    parser.add_argument("--max-per-class", type=int,   default=None)
    parser.add_argument("--epochs",        type=int,   default=EPOCHS)
    parser.add_argument("--batch-size",    type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",            type=float, default=INITIAL_LR)
    parser.add_argument("--min-lr",        type=float, default=MIN_LR)
    parser.add_argument("--warmup-epochs", type=int,   default=WARMUP_EPOCHS)
    parser.add_argument("--dropout-conv",  type=float, default=0.3)
    parser.add_argument("--dropout-head",  type=float, default=0.5)
    parser.add_argument("--val-split",     type=float, default=0.2)
    parser.add_argument("--augment-train", action="store_true", default=True)
    parser.add_argument("--no-augment",    action="store_false", dest="augment_train")
    parser.add_argument("--mixup",         action="store_true", default=True,
                        help="Use mixup augmentation (default: True)")
    parser.add_argument("--no-mixup",      action="store_false", dest="mixup")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience",      type=int,   default=40)
    parser.add_argument("--model-dir",     type=Path,  default=MODEL_DIR)
    parser.add_argument("--results-dir",   type=Path,  default=RESULTS_DIR)
    parser.add_argument("--name",          type=str,   default=None)

    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.name = f"scratch_cnn_{timestamp}"

    exp_dir = args.results_dir / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Bird Call Classification — FROM-SCRATCH CNN")
    print("=" * 70)
    print(f"Experiment:       {args.name}")
    print(f"Architecture:     Custom CNN (32→64→128→256 conv blocks, ~2M params)")
    print(f"Pretrained:       NO — all weights initialised randomly")
    print(f"Image size:       {IMG_SIZE}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Epochs:           {args.epochs}  (early stop patience={args.patience})")
    print(f"LR schedule:      warmup {args.warmup_epochs} epochs → cosine decay to {args.min_lr}")
    print(f"Augmentation:     {'ON' if args.augment_train else 'OFF'}")
    print(f"Mixup:            {'ON' if args.mixup else 'OFF'}")
    print(f"Label smoothing:  {args.label_smoothing}")
    print("=" * 70)

    # Load data
    image_paths, labels, class_names = load_spectrogram_dataset(
        args.spec_dir, use_augmented=args.use_augmented, max_per_class=args.max_per_class
    )
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(class_names)

    avg_per_class = len(image_paths) / num_classes
    print(f"\nDataset: {len(image_paths)} samples | {num_classes} classes | "
          f"~{avg_per_class:.0f} samples/class")

    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, encoded_labels,
        test_size=args.val_split,
        stratify=encoded_labels,
        random_state=42
    )
    print(f"Split: {len(X_train)} train / {len(X_val)} val\n")

    train_dataset = create_dataset(
        X_train, y_train, num_classes=num_classes,
        batch_size=args.batch_size, shuffle=True,
        augment_data=args.augment_train, use_mixup=args.mixup
    )
    val_dataset = create_dataset(
        X_val, y_val, num_classes=num_classes,
        batch_size=args.batch_size, shuffle=False,
        augment_data=False, use_mixup=False
    )

    # Build model
    model = build_scratch_model(
        num_classes=num_classes,
        dropout_conv=args.dropout_conv,
        dropout_head=args.dropout_head,
        l2=L2_REG
    )

    # LR schedule
    steps_per_epoch = max(len(X_train) // args.batch_size, 1)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    lr_schedule = WarmupCosineDecay(
        initial_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=L2_REG),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=['accuracy']
    )

    model.summary()
    print(f"\nTrainable parameters: {model.count_params():,}")
    print(f"\nStarting training...")
    print("=" * 70)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=get_callbacks(exp_dir, patience=args.patience),
        verbose=1
    )

    # Evaluate
    print(f"\n{'=' * 70}")
    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f"Final Validation Loss:     {val_loss:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"Best  Validation Accuracy: {max(history.history['val_accuracy']):.4f}")

    # Save
    model.save(str(exp_dir / 'final_model.keras'))
    plot_training_history(history, exp_dir / 'training_history.png')
    save_info(class_names, label_encoder, history, exp_dir)

    print(f"\n{'=' * 70}")
    print(f"All results saved to: {exp_dir}")
    print(f"Best val accuracy: {max(history.history['val_accuracy']):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    print("RUNNING FILE:", __file__)
    main()
