"""
Precompute log-mel-spectrogram tensors from WAV files for faster loading and resume capabilities
"""

from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from deepnet.utils import DATA_ROOT, setup_logging

SAMPLE_RATE = 22050
DURATION = 5  # We limit to 5 seconds
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
TARGET_LENGTH = SAMPLE_RATE * DURATION
# Expected time frames: ceil(TARGET_LENGTH / HOP_LENGTH) + 1 = 216
EXPECTED_TIME_FRAMES = 216

CLEANED_DIR = DATA_ROOT / "processed" / "cleaned"
AUGMENTED_DIR = DATA_ROOT / "processed" / "augmented"
MEL_TENSORS_DIR = DATA_ROOT / "processed" / "mel_tensors"

log = setup_logging("preprocess")


def wav_to_mel(
    wav_path: Path, mel_transform: torchaudio.transforms.MelSpectrogram
) -> torch.Tensor:
    """Load a WAV file and convert to a log-mel-spectrogram tensor.

    Returns tensor of shape (1, N_MELS, EXPECTED_TIME_FRAMES).
    """
    data, sr = sf.read(wav_path, dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # (1, samples)
    else:
        waveform = waveform.T  # (channels, samples)

    # Resample if needed
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Pad or truncate to TARGET_LENGTH
    if waveform.shape[1] < TARGET_LENGTH:
        padding = TARGET_LENGTH - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif waveform.shape[1] > TARGET_LENGTH:
        waveform = waveform[:, :TARGET_LENGTH]

    # Compute mel spectrogram
    mel_spec = mel_transform(waveform)  # (1, N_MELS, time_frames)

    # Convert to log scale (dB)
    mel_spec = torch.clamp(mel_spec, min=1e-10)
    mel_spec = torch.log10(mel_spec)

    # Ensure exact time dimension
    if mel_spec.shape[2] > EXPECTED_TIME_FRAMES:
        mel_spec = mel_spec[:, :, :EXPECTED_TIME_FRAMES]
    elif mel_spec.shape[2] < EXPECTED_TIME_FRAMES:
        padding = EXPECTED_TIME_FRAMES - mel_spec.shape[2]
        mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))

    return mel_spec


def process_directory(
    source_dir: Path,
    output_dir: Path,
    mel_transform: torchaudio.transforms.MelSpectrogram,
) -> int:
    """Convert all WAVs in source_dir/{species}/*.wav to .pt tensors."""
    if not source_dir.exists():
        log.warning(f"Source directory does not exist: {source_dir}")
        return 0

    wav_files = sorted(source_dir.rglob("*.wav"))
    if not wav_files:
        log.warning(f"No WAV files found in {source_dir}")
        return 0

    converted = 0
    failed = 0
    for wav_path in tqdm(wav_files, desc=f"Processing {source_dir.name}"):
        # Preserve species subdirectory structure
        relative = wav_path.relative_to(source_dir)
        out_path = output_dir / relative.with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            converted += 1
            continue

        try:
            mel_tensor = wav_to_mel(wav_path, mel_transform)
            torch.save(mel_tensor, out_path)
            converted += 1
        except Exception as e:
            log.warning(f"Failed to process {wav_path.name}: {e}")
            failed += 1

    if failed:
        log.warning(f"{failed} files failed in {source_dir.name}")
    return converted


def main() -> None:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    log.info("Pre-computing mel-spectrogram tensors")
    log.info(f"  n_fft={N_FFT}, hop_length={HOP_LENGTH}, n_mels={N_MELS}")
    log.info(f"  Expected tensor shape: (1, {N_MELS}, {EXPECTED_TIME_FRAMES})")

    # Process cleaned WAVs
    cleaned_out = MEL_TENSORS_DIR / "cleaned"
    n_cleaned = process_directory(CLEANED_DIR, cleaned_out, mel_transform)
    log.info(f"Cleaned: {n_cleaned} tensors saved to {cleaned_out}")

    # Process augmented WAVs
    augmented_out = MEL_TENSORS_DIR / "augmented"
    n_augmented = process_directory(AUGMENTED_DIR, augmented_out, mel_transform)
    log.info(f"Augmented: {n_augmented} tensors saved to {augmented_out}")

    # Verify a sample
    sample = next(cleaned_out.rglob("*.pt"))
    t = torch.load(sample, weights_only=True)
    log.info(
        f"Sample tensor shape: {t.shape}, dtype: {t.dtype}, range: [{t.min():.2f}, {t.max():.2f}]"
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
