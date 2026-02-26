"""Spectrogram-domain data augmentation transforms.

Implements SpecAugment (frequency/time masking), random time shift,
and Gaussian noise for training bird call classifiers.

Usage:
    from deepnet.transforms import build_spec_augment
    transform = build_spec_augment(config['augmentation'])
"""

import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """SpecAugment: frequency and time masking for spectrogram augmentation.

    Applies random rectangular masks along frequency and time axes,
    replacing masked values with zero (mean of log-mel spectrogram).

    Reference: Park et al. "SpecAugment: A Simple Data Augmentation Method
    for Automatic Speech Recognition" (2019)

    Args:
        freq_mask_param: Maximum number of frequency bands to mask (F parameter)
        time_mask_param: Maximum number of time frames to mask (T parameter)
        num_freq_masks: Number of frequency masks to apply
        num_time_masks: Number of time masks to apply
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 25,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to a spectrogram tensor.

        Args:
            spec: Tensor of shape (C, F, T) — channels, freq bins, time frames

        Returns:
            Augmented spectrogram of same shape
        """
        spec = spec.clone()
        _, num_freq, num_time = spec.shape

        # Apply frequency masks
        for _ in range(self.num_freq_masks):
            if self.freq_mask_param > 0 and num_freq > 1:
                f = int(torch.randint(0, min(self.freq_mask_param + 1, num_freq), (1,)).item())
                if f > 0:
                    f0 = int(torch.randint(0, num_freq - f + 1, (1,)).item())
                    spec[:, f0:f0 + f, :] = 0.0

        # Apply time masks
        for _ in range(self.num_time_masks):
            if self.time_mask_param > 0 and num_time > 1:
                t = int(torch.randint(0, min(self.time_mask_param + 1, num_time), (1,)).item())
                if t > 0:
                    t0 = int(torch.randint(0, num_time - t + 1, (1,)).item())
                    spec[:, :, t0:t0 + t] = 0.0

        return spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"freq_mask_param={self.freq_mask_param}, "
            f"time_mask_param={self.time_mask_param}, "
            f"num_freq_masks={self.num_freq_masks}, "
            f"num_time_masks={self.num_time_masks})"
        )


class RandomTimeShift(nn.Module):
    """Randomly shifts spectrogram along the time axis with wrap-around.

    Useful for making the model invariant to slight temporal offsets.

    Args:
        max_shift: Maximum number of frames to shift (in either direction)
    """

    def __init__(self, max_shift: int = 20):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply random time shift.

        Args:
            spec: Tensor of shape (C, F, T)

        Returns:
            Time-shifted spectrogram of same shape (wrapped)
        """
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
        return torch.roll(spec, shifts=shift, dims=-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_shift={self.max_shift})"


class GaussianNoise(nn.Module):
    """Adds Gaussian noise to the spectrogram.

    Simulates sensor noise and environmental interference.

    Args:
        std: Standard deviation of the Gaussian noise
    """

    def __init__(self, std: float = 0.01):
        super().__init__()
        self.std = std

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise.

        Args:
            spec: Tensor of shape (C, F, T)

        Returns:
            Noisy spectrogram of same shape
        """
        noise = torch.randn_like(spec) * self.std
        return spec + noise

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"


def build_spec_augment(aug_config: dict) -> nn.Module | None:
    """Build a spectrogram transform pipeline from config.

    Args:
        aug_config: Dict with augmentation settings:
            - spec_augment (bool): Enable SpecAugment
            - freq_mask_param (int): Max freq bands to mask (default 15)
            - time_mask_param (int): Max time frames to mask (default 25)
            - num_masks (int): Number of masks of each type (default 2)
            - time_shift (bool): Enable random time shift (default False)
            - max_shift (int): Max time shift in frames (default 20)
            - gaussian_noise (bool): Enable Gaussian noise (default False)
            - noise_std (float): Noise standard deviation (default 0.01)

    Returns:
        Composed transform module, or None if no augmentation enabled
    """
    if not aug_config.get('spec_augment', False):
        # Check if any other augment is enabled
        has_other = (
            aug_config.get('time_shift', False)
            or aug_config.get('gaussian_noise', False)
        )
        if not has_other:
            return None

    transforms: list[nn.Module] = []

    if aug_config.get('spec_augment', False):
        transforms.append(SpecAugment(
            freq_mask_param=aug_config.get('freq_mask_param', 15),
            time_mask_param=aug_config.get('time_mask_param', 25),
            num_freq_masks=aug_config.get('num_masks', 2),
            num_time_masks=aug_config.get('num_masks', 2),
        ))

    if aug_config.get('time_shift', False):
        transforms.append(RandomTimeShift(
            max_shift=aug_config.get('max_shift', 20),
        ))

    if aug_config.get('gaussian_noise', False):
        transforms.append(GaussianNoise(
            std=aug_config.get('noise_std', 0.01),
        ))

    if not transforms:
        return None

    if len(transforms) == 1:
        return transforms[0]

    return nn.Sequential(*transforms)


if __name__ == "__main__":
    print("Testing transforms...")
    spec = torch.randn(1, 128, 216)

    # Test SpecAugment
    sa = SpecAugment(freq_mask_param=15, time_mask_param=25, num_freq_masks=2, num_time_masks=2)
    out = sa(spec)
    assert out.shape == spec.shape, f"Shape mismatch: {out.shape}"
    print(f"SpecAugment: {spec.shape} → {out.shape}")

    # Test RandomTimeShift
    rts = RandomTimeShift(max_shift=20)
    out = rts(spec)
    assert out.shape == spec.shape
    print(f"RandomTimeShift: {spec.shape} → {out.shape}")

    # Test GaussianNoise
    gn = GaussianNoise(std=0.01)
    out = gn(spec)
    assert out.shape == spec.shape
    print(f"GaussianNoise: {spec.shape} → {out.shape}")

    # Test build_spec_augment
    config = {
        'spec_augment': True,
        'freq_mask_param': 15,
        'time_mask_param': 25,
        'num_masks': 2,
        'time_shift': True,
        'gaussian_noise': True,
    }
    pipeline = build_spec_augment(config)
    assert pipeline is not None
    out = pipeline(spec)
    assert out.shape == spec.shape
    print(f"Full pipeline: {spec.shape} → {out.shape}")
    print(f"Pipeline: {pipeline}")

    print("\nAll transforms tested successfully!")
