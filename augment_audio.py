# augment_audio.py
import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
INPUT_ROOT = Path("data/recordings")
OUTPUT_ROOT = Path("data/augmented")

SAMPLE_RATE = 22050
DURATION = 5  # Seconds

def add_noise(y, noise_level=0.005):
    """Adds white noise (static) to the audio."""
    noise = np.random.randn(len(y))
    augmented_data = y + noise_level * noise
    return augmented_data

def process_file(file_path, relative_path):
    try:
        # 1. LOAD & CLEAN
        # Load audio (automatically converts to mono/resamples)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Normalize volume
        y = librosa.util.normalize(y)

        # 2. PAD OR TRUNCATE
        target_len = SAMPLE_RATE * DURATION
        if len(y) > target_len:
            y = y[:target_len]
        else:
            padding = target_len - len(y)
            y = np.pad(y, (0, padding), 'wrap')

        # 3. AUGMENT
        y_noisy = add_noise(y)
        
        # 4. PREPARE OUTPUT PATHS
        # Create the species folder in the output directory
        species_folder = OUTPUT_ROOT / relative_path.parent
        species_folder.mkdir(parents=True, exist_ok=True)
        
        filename = relative_path.name
        
        # Save "Clean" version (Processed but no extra noise)
        clean_path = species_folder / f"clean_{filename}"
        sf.write(clean_path, y, sr)
        
        # Save "Augmented" version (With noise)
        aug_path = species_folder / f"aug_{filename}"
        sf.write(aug_path, y_noisy, sr)
        
        print(f"Processed: {relative_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# --- MAIN EXECUTION ---
print(f"Scanning for files in {INPUT_ROOT}...")

if not INPUT_ROOT.exists():
    print(f"Error: Could not find '{INPUT_ROOT}'. Make sure you are in the 'Bird-Call-Detection-main' folder.")
else:
    # Walk through all species folders
    count = 0
    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.lower().endswith('.mp3'):
                # Get the full path to the file
                full_path = Path(root) / file
                
                # Get path relative to input root (e.g., "American_Coot/XC123.mp3")
                relative_path = full_path.relative_to(INPUT_ROOT)
                
                process_file(full_path, relative_path)
                count += 1

    print(f"Done! Processed {count} files. Output is in '{OUTPUT_ROOT}'.")