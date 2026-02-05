"""
Audio processor for bird call recordings.
Performs cleaning, augmentation, and spectrogram generation on downloaded audio files.
Supports parallel processing for faster execution.
"""

import os
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import argparse
from typing import Tuple, List
from scipy import signal

# --- CONFIGURATION ---
INPUT_ROOT = Path("data/recordings")
OUTPUT_ROOT = Path("data/processed")
CLEANED_DIR = OUTPUT_ROOT / "cleaned"
AUGMENTED_DIR = OUTPUT_ROOT / "augmented"
SPECTROGRAM_DIR = OUTPUT_ROOT / "spectrograms"

SAMPLE_RATE = 22050
DURATION = 5  # Seconds
TARGET_LENGTH = SAMPLE_RATE * DURATION

# Calculate safe number of workers (leave 1 core free for your OS)
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)


def add_noise(y: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """Add white noise to audio signal."""
    noise = np.random.randn(len(y))
    augmented_data = y + noise_level * noise
    return augmented_data


def time_shift(y: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
    """Shift audio in time by a random amount."""
    shift = np.random.randint(int(SAMPLE_RATE * shift_max))
    direction = np.random.choice([-1, 1])
    shift = shift * direction
    return np.roll(y, shift)


def pitch_shift(y: np.ndarray, sr: int, n_steps: float = None) -> np.ndarray:
    """Shift pitch of audio by random semitones."""
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def time_stretch(y: np.ndarray, rate: float = None) -> np.ndarray:
    """Stretch or compress audio in time."""
    if rate is None:
        rate = np.random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(y, rate=rate)


def add_background_noise(y: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
    """Add colored background noise."""
    # Generate pink noise (1/f noise)
    noise = np.random.randn(len(y))
    # Simple pink noise approximation
    b, a = signal.butter(1, 0.1)
    noise = signal.lfilter(b, a, noise)
    noise = noise / np.max(np.abs(noise))  # Normalize
    return y + noise_factor * noise


def clean_audio(y: np.ndarray, sr: int,
                noise_reduce: bool = True,
                normalize: bool = True,
                trim_silence: bool = True) -> np.ndarray:
    """
    Clean audio by removing noise, normalizing, and trimming silence.
    
    Args:
        y: Audio signal
        sr: Sample rate
        noise_reduce: Apply noise reduction
        normalize: Normalize audio levels
        trim_silence: Trim silence from beginning and end
    
    Returns:
        Cleaned audio signal
    """
    # Trim silence
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=20)
    
    # Normalize
    if normalize:
        y = librosa.util.normalize(y)
    
    # Basic noise reduction using spectral gating
    if noise_reduce:
        # Compute spectrogram
        D = librosa.stft(y)
        magnitude, phase = librosa.magphase(D)
        
        # Estimate noise floor from quieter frames
        noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
        
        # Apply soft gating
        mask = magnitude > (noise_floor * 2)
        magnitude_clean = magnitude * mask
        
        # Reconstruct
        D_clean = magnitude_clean * phase
        y = librosa.istft(D_clean)
    
    return y


def pad_or_truncate(y: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or truncate audio to target length."""
    if len(y) > target_length:
        return y[:target_length]
    else:
        padding = target_length - len(y)
        return np.pad(y, (0, padding), 'wrap')


def generate_spectrogram(y: np.ndarray, sr: int, output_path: Path,
                        n_fft: int = 2048,
                        hop_length: int = 512,
                        n_mels: int = 128) -> bool:
    """
    Generate and save a mel spectrogram.
    
    Args:
        y: Audio signal
        sr: Sample rate
        output_path: Path to save spectrogram
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
    
    Returns:
        True if successful
    """
    try:
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Create figure
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            mel_spec_db, sr=sr, hop_length=hop_length,
            x_axis='time', y_axis='mel', cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram - {output_path.stem}')
        plt.tight_layout()
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return True
        
    except Exception as e:
        print(f"    Failed to generate spectrogram: {e}")
        plt.close()
        return False


def process_single_file(args: Tuple) -> str:
    """
    Process a single audio file with all augmentations.
    
    Args:
        args: Tuple of (file_path, relative_path, options)
    
    Returns:
        Status message
    """
    file_path, relative_path, options = args
    
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Clean audio
        y_clean = clean_audio(
            y, sr,
            noise_reduce=options['noise_reduce'],
            normalize=True,
            trim_silence=True
        )
        
        # Pad or truncate to target length
        y_clean = pad_or_truncate(y_clean, TARGET_LENGTH)
        
        # Prepare output directories
        species_folder = relative_path.parent
        filename = relative_path.stem
        
        outputs_created = []
        
        # 1. Save cleaned version
        if options['save_cleaned']:
            clean_dir = CLEANED_DIR / species_folder
            clean_dir.mkdir(parents=True, exist_ok=True)
            clean_path = clean_dir / f"clean_{filename}.wav"
            sf.write(str(clean_path), y_clean, sr)
            outputs_created.append("cleaned")
            
            # Generate spectrogram for cleaned audio
            if options['generate_spectrograms']:
                spec_dir = SPECTROGRAM_DIR / "cleaned" / species_folder
                spec_dir.mkdir(parents=True, exist_ok=True)
                spec_path = spec_dir / f"clean_{filename}.png"
                if generate_spectrogram(y_clean, sr, spec_path):
                    outputs_created.append("clean_spec")
        
        # 2. Generate augmentations
        if options['augment']:
            aug_base_dir = AUGMENTED_DIR / species_folder
            aug_base_dir.mkdir(parents=True, exist_ok=True)
            
            augmentations = []
            
            # Add noise
            if options['aug_noise']:
                y_noisy = add_noise(y_clean, noise_level=0.005)
                augmentations.append(('noise', y_noisy))
            
            # Time shift
            if options['aug_time_shift']:
                y_shifted = time_shift(y_clean)
                augmentations.append(('timeshift', y_shifted))
            
            # Pitch shift
            if options['aug_pitch']:
                y_pitched = pitch_shift(y_clean, sr)
                augmentations.append(('pitch', y_pitched))
            
            # Time stretch
            if options['aug_stretch']:
                y_stretched = time_stretch(y_clean)
                y_stretched = pad_or_truncate(y_stretched, TARGET_LENGTH)
                augmentations.append(('stretch', y_stretched))
            
            # Background noise
            if options['aug_background']:
                y_bg = add_background_noise(y_clean)
                augmentations.append(('background', y_bg))
            
            # Save augmented versions
            for aug_name, y_aug in augmentations:
                aug_path = aug_base_dir / f"aug_{aug_name}_{filename}.wav"
                sf.write(str(aug_path), y_aug, sr)
                outputs_created.append(f"aug_{aug_name}")
                
                # Generate spectrogram for augmented audio
                if options['generate_spectrograms']:
                    spec_dir = SPECTROGRAM_DIR / "augmented" / species_folder
                    spec_dir.mkdir(parents=True, exist_ok=True)
                    spec_path = spec_dir / f"aug_{aug_name}_{filename}.png"
                    if generate_spectrogram(y_aug, sr, spec_path):
                        outputs_created.append(f"aug_{aug_name}_spec")
        
        return f"✓ {relative_path.name}: {', '.join(outputs_created)}"
        
    except Exception as e:
        return f"✗ {relative_path.name}: {e}"


def collect_audio_files(input_root: Path, max_per_species: int = None) -> List[Tuple]:
    """Collect all audio files to process."""
    all_files = []
    
    if not input_root.exists():
        print(f"Error: Input directory '{input_root}' not found.")
        return all_files
    
    # Walk through species folders
    for root, dirs, files in os.walk(input_root):
        mp3_files = [f for f in files if f.lower().endswith('.mp3')]
        
        # Apply per-species limit if specified
        if max_per_species:
            mp3_files = mp3_files[:max_per_species]
        
        for file in mp3_files:
            full_path = Path(root) / file
            relative_path = full_path.relative_to(input_root)
            all_files.append((full_path, relative_path))
    
    return all_files


def main():
    parser = argparse.ArgumentParser(
        description="Process bird call audio: clean, augment, and generate spectrograms"
    )
    
    # Input/Output
    parser.add_argument("--input-dir", type=Path, default=INPUT_ROOT,
                       help=f"Input directory with recordings (default: {INPUT_ROOT})")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT,
                       help=f"Output directory (default: {OUTPUT_ROOT})")
    parser.add_argument("--max-per-species", type=int, default=None,
                       help="Maximum files to process per species")
    
    # Processing options
    parser.add_argument("--no-clean", action="store_true",
                       help="Skip saving cleaned audio files")
    parser.add_argument("--no-noise-reduce", action="store_true",
                       help="Skip noise reduction in cleaning")
    parser.add_argument("--spectrograms", action="store_true",
                       help="Generate mel spectrograms")
    
    # Augmentation options
    parser.add_argument("--augment", action="store_true",
                       help="Enable data augmentation")
    parser.add_argument("--aug-noise", action="store_true",
                       help="Add white noise augmentation")
    parser.add_argument("--aug-time-shift", action="store_true",
                       help="Add time shift augmentation")
    parser.add_argument("--aug-pitch", action="store_true",
                       help="Add pitch shift augmentation")
    parser.add_argument("--aug-stretch", action="store_true",
                       help="Add time stretch augmentation")
    parser.add_argument("--aug-background", action="store_true",
                       help="Add background noise augmentation")
    parser.add_argument("--aug-all", action="store_true",
                       help="Enable all augmentation types")
    
    # Performance
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                       help=f"Number of parallel workers (default: {MAX_WORKERS})")
    parser.add_argument("--sequential", action="store_true",
                       help="Process files sequentially (no parallelization)")
    
    args = parser.parse_args()
    
    # Update global paths if custom output directory specified
    global CLEANED_DIR, AUGMENTED_DIR, SPECTROGRAM_DIR
    if args.output_dir != OUTPUT_ROOT:
        CLEANED_DIR = args.output_dir / "cleaned"
        AUGMENTED_DIR = args.output_dir / "augmented"
        SPECTROGRAM_DIR = args.output_dir / "spectrograms"
    
    # Set augmentation flags
    if args.aug_all:
        args.aug_noise = True
        args.aug_time_shift = True
        args.aug_pitch = True
        args.aug_stretch = True
        args.aug_background = True
    
    # Build options dict
    options = {
        'save_cleaned': not args.no_clean,
        'noise_reduce': not args.no_noise_reduce,
        'generate_spectrograms': args.spectrograms,
        'augment': args.augment or args.aug_all,
        'aug_noise': args.aug_noise,
        'aug_time_shift': args.aug_time_shift,
        'aug_pitch': args.aug_pitch,
        'aug_stretch': args.aug_stretch,
        'aug_background': args.aug_background,
    }
    
    print(f"Audio Processor for Bird Calls")
    print(f"=" * 50)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {args.workers if not args.sequential else 1}")
    print(f"Options:")
    print(f"  - Save cleaned: {options['save_cleaned']}")
    print(f"  - Noise reduction: {options['noise_reduce']}")
    print(f"  - Generate spectrograms: {options['generate_spectrograms']}")
    print(f"  - Augmentation: {options['augment']}")
    if options['augment']:
        print(f"    • Noise: {options['aug_noise']}")
        print(f"    • Time shift: {options['aug_time_shift']}")
        print(f"    • Pitch shift: {options['aug_pitch']}")
        print(f"    • Time stretch: {options['aug_stretch']}")
        print(f"    • Background noise: {options['aug_background']}")
    print(f"=" * 50)
    
    # Collect files
    print("\nScanning for audio files...")
    file_list = collect_audio_files(args.input_dir, args.max_per_species)
    
    if not file_list:
        print("No audio files found!")
        return
    
    total_files = len(file_list)
    print(f"Found {total_files} files to process.\n")
    
    # Prepare tasks
    tasks = [(fp, rp, options) for fp, rp in file_list]
    
    # Process files
    if args.sequential:
        print("Processing sequentially...")
        for i, task in enumerate(tasks, 1):
            result = process_single_file(task)
            if i % 10 == 0 or i == total_files:
                print(f"[{i}/{total_files}] {result}")
    else:
        print(f"Processing in parallel with {args.workers} workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(process_single_file, tasks)
            
            for i, result in enumerate(results, 1):
                if i % 10 == 0 or i == total_files:
                    print(f"[{i}/{total_files}] {result}")
    
    print("\n" + "=" * 50)
    print("Processing complete!")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
