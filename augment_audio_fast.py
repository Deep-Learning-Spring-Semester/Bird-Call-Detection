# fast_augment.py
import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# --- CONFIGURATION ---
INPUT_ROOT = Path("data/recordings")
OUTPUT_ROOT = Path("data/augmented")
SAMPLE_RATE = 22050
DURATION = 5  # Seconds

# Calculate safe number of workers (leave 1 core free for your OS)
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)

def add_noise(y, noise_level=0.005):
    noise = np.random.randn(len(y))
    augmented_data = y + noise_level * noise
    return augmented_data

def process_single_file(file_info):
    """
    Worker function that processes one file independently.
    We pass a tuple (file_path, relative_path) to keep it simple for the executor.
    """
    file_path, relative_path = file_info
    
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
        os.makedirs(species_folder, exist_ok=True)
        
        filename = relative_path.name
        
        # Save "Clean" version
        clean_path = species_folder / f"clean_{filename}"
        sf.write(str(clean_path), y, sr)
        
        # Save "Augmented" version
        aug_path = species_folder / f"aug_{filename}"
        sf.write(str(aug_path), y_noisy, sr)
        
        return f"Processed: {filename}"

    except Exception as e:
        return f"ERROR on {filename}: {e}"

def main():
    print(f"Scanning files in {INPUT_ROOT}...")
    
    if not INPUT_ROOT.exists():
        print(f"Error: Could not find '{INPUT_ROOT}'.")
        return

    # 1. Gather all tasks first
    all_tasks = []
    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.lower().endswith('.mp3'):
                full_path = Path(root) / file
                relative_path = full_path.relative_to(INPUT_ROOT)
                all_tasks.append((full_path, relative_path))

    total_files = len(all_tasks)
    print(f"Found {total_files} files. Starting processing on {MAX_WORKERS} CPU cores...")

    # 2. Run Parallel Processing
    # This creates a pool of worker processes to eat through the list
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        results = executor.map(process_single_file, all_tasks)
        
        # Print results as they finish
        count = 0
        for result in results:
            count += 1
            if count % 10 == 0: # Only print every 10th file to reduce clutter
                print(f"[{count}/{total_files}] ... {result}")

    print("Done! All audio processed.")

if __name__ == "__main__":
    main()