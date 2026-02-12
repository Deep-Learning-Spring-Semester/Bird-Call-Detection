"""
EfficientNet-based bird call classifier training script.
Trains on spectrograms generated from audio recordings.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
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

IMG_SIZE = (224, 224)  # EfficientNet input size
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001


def load_spectrogram_dataset(spec_dir: Path, use_augmented: bool = False, max_per_class: int = None):
    """
    Load spectrogram images and their labels.
    
    Args:
        spec_dir: Directory containing species folders with spectrograms
        use_augmented: Whether to include augmented spectrograms
        max_per_class: Maximum samples per species (None for all)
    
    Returns:
        images: List of image paths
        labels: List of species labels
        class_names: List of unique species names
    """
    images = []
    labels = []
    
    print(f"Loading spectrograms from {spec_dir}...")
    
    if not spec_dir.exists():
        raise FileNotFoundError(f"Spectrogram directory not found: {spec_dir}")
    
    # Get all species folders
    species_folders = [f for f in spec_dir.iterdir() if f.is_dir()]
    
    if not species_folders:
        raise ValueError(f"No species folders found in {spec_dir}")
    
    print(f"Found {len(species_folders)} species")
    
    for species_folder in sorted(species_folders):
        species_name = species_folder.name
        
        # Get all PNG spectrograms
        spec_files = list(species_folder.glob("*.png"))
        
        if max_per_class:
            spec_files = spec_files[:max_per_class]
        
        for spec_file in spec_files:
            images.append(str(spec_file))
            labels.append(species_name)
        
        print(f"  {species_name}: {len(spec_files)} spectrograms")
    
    # Load augmented data if requested
    if use_augmented and AUGMENTED_SPEC_DIR.exists():
        print(f"\nLoading augmented spectrograms from {AUGMENTED_SPEC_DIR}...")
        
        for species_folder in sorted(AUGMENTED_SPEC_DIR.iterdir()):
            if not species_folder.is_dir():
                continue
                
            species_name = species_folder.name
            aug_files = list(species_folder.glob("*.png"))
            
            if max_per_class:
                aug_files = aug_files[:max_per_class]
            
            for spec_file in aug_files:
                images.append(str(spec_file))
                labels.append(species_name)
            
            print(f"  {species_name}: {len(aug_files)} augmented spectrograms")
    
    class_names = sorted(list(set(labels)))
    print(f"\nTotal samples: {len(images)}")
    print(f"Total classes: {len(class_names)}")
    
    return images, labels, class_names


def load_and_preprocess_image(image_path: str, img_size: tuple = IMG_SIZE):
    """Load and preprocess a single spectrogram image."""
    # Read image
    img = tf.io.read_file(image_path)
    
    # Decode image - try to force RGB
    try:
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
    except:
        img = tf.image.decode_png(img, channels=3)
    
    # Ensure RGB format (in case of grayscale)
    if img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)
    elif img.shape[-1] == 4:  # RGBA
        img = img[:, :, :3]  # Drop alpha channel
    
    # Set shape explicitly
    img.set_shape([None, None, 3])
    
    # Resize
    img = tf.image.resize(img, img_size)
    
    # Normalize to [0, 1]
    img = img / 255.0
    
    return img


def create_dataset(image_paths, labels, batch_size=BATCH_SIZE, shuffle=True, augment=False):
    """
    Create a tf.data.Dataset from image paths and labels.
    
    Args:
        image_paths: List of image file paths
        labels: List of encoded labels
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        augment: Whether to apply data augmentation
    
    Returns:
        TensorFlow dataset
    """
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    
    # Load and preprocess images
    dataset = dataset.map(
        lambda x, y: (load_and_preprocess_image(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Data augmentation for training
    if augment:
        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
        ])
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def build_efficientnet_model(num_classes: int, model_name: str = "B0", 
                             freeze_base: bool = True, dropout_rate: float = 0.3):
    """
    Build EfficientNet-based classification model.
    
    Args:
        num_classes: Number of bird species classes
        model_name: EfficientNet variant ("B0", "B1", "B2")
        freeze_base: Whether to freeze base model weights
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    # Select EfficientNet variant
    if model_name == "B0":
        base_model = EfficientNetB0(include_top=False, weights='imagenet', 
                                    input_shape=(224, 224, 3))
    elif model_name == "B1":
        base_model = EfficientNetB1(include_top=False, weights='imagenet', 
                                    input_shape=(224, 224, 3))
    elif model_name == "B2":
        base_model = EfficientNetB2(include_top=False, weights='imagenet', 
                                    input_shape=(224, 224, 3))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Freeze base model
    if freeze_base:
        base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


def plot_training_history(history, save_path: Path):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to {save_path}")


def save_training_info(class_names, label_encoder, model_name, history, save_dir: Path):
    """Save training information and metadata."""
    info = {
        'model_name': model_name,
        'num_classes': len(class_names),
        'class_names': class_names,
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save as JSON
    info_path = save_dir / 'training_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    # Save label encoder
    encoder_path = save_dir / 'label_encoder.pkl'
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Training info saved to {info_path}")
    print(f"Label encoder saved to {encoder_path}")


def main():
    tf.keras.backend.clear_session()
    parser = argparse.ArgumentParser(
        description="Train EfficientNet model for bird call classification"
    )
    
    # Data options
    parser.add_argument("--spec-dir", type=Path, default=SPECTROGRAM_DIR,
                       help="Directory with cleaned spectrograms")
    parser.add_argument("--use-augmented", action="store_true",
                       help="Include augmented spectrograms in training")
    parser.add_argument("--max-per-class", type=int, default=None,
                       help="Maximum samples per class")
    
    # Model options
    parser.add_argument("--model", choices=["B0", "B1", "B2"], default="B0",
                       help="EfficientNet variant (default: B0)")
    parser.add_argument("--freeze-base", action="store_true", default=True,
                       help="Freeze base model weights (transfer learning)")
    parser.add_argument("--unfreeze-base", action="store_false", dest="freeze_base",
                       help="Unfreeze base model (fine-tuning)")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate (default: 0.3)")
    
    # Training options
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                       help=f"Number of epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                       help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split ratio (default: 0.2)")
    parser.add_argument("--augment-train", action="store_true",
                       help="Apply data augmentation during training")
    
    # Output options
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                       help=f"Model save directory (default: {MODEL_DIR})")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                       help=f"Results save directory (default: {RESULTS_DIR})")
    parser.add_argument("--name", type=str, default=None,
                       help="Experiment name (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Create output directories
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment name
    if args.name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.name = f"efficientnet_{args.model}_{timestamp}"
    
    exp_dir = args.results_dir / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 70)
    print(f"Bird Call Classification Training")
    print(f"=" * 70)
    print(f"Experiment: {args.name}")
    print(f"Model: EfficientNet-{args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"=" * 70)
    
    # Load dataset
    image_paths, labels, class_names = load_spectrogram_dataset(
        args.spec_dir,
        use_augmented=args.use_augmented,
        max_per_class=args.max_per_class
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(class_names)
    
    print(f"\nDataset loaded:")
    print(f"  Total samples: {len(image_paths)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, encoded_labels,
        test_size=args.val_split,
        stratify=encoded_labels,
        random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Create datasets
    train_dataset = create_dataset(
        X_train, y_train,
        batch_size=args.batch_size,
        shuffle=True,
        augment=args.augment_train
    )
    
    val_dataset = create_dataset(
        X_val, y_val,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False
    )
    
    # Build model
    print(f"\nBuilding EfficientNet-{args.model} model...")
    model = build_efficientnet_model(
        num_classes=num_classes,
        model_name=args.model,
        freeze_base=args.freeze_base,
        dropout_rate=args.dropout
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nModel summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=exp_dir / 'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename=exp_dir / 'training_log.csv',
            separator=',',
            append=False
        )
    ]
    
    # Train model
    print(f"\nStarting training...")
    print(f"=" * 70)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n" + "=" * 70)
    print(f"Training complete!")
    print(f"=" * 70)
    
    # Evaluate on validation set
    print(f"\nEvaluating on validation set...")
    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Save final model
    final_model_path = exp_dir / 'final_model.keras'
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Plot training history
    plot_path = exp_dir / 'training_history.png'
    plot_training_history(history, plot_path)
    
    # Save training info
    save_training_info(class_names, label_encoder, args.model, history, exp_dir)
    
    print(f"\n" + "=" * 70)
    print(f"All results saved to: {exp_dir}")
    print(f"=" * 70)
    print(f"\nBest validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final validation accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    print("RUNNING FILE:", __file__)

    main()
