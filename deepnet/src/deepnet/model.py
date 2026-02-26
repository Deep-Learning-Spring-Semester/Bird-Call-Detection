"""
DEEPNET: Deep Neural Network for Bird Call Classification

Architecture: Custom deep CNN with Dual-Path Blocks for frequency and temporal feature extraction.
- StemBlock: Initial feature extraction with 2 conv layers
- DualPathBlock: Asymmetric kernels for frequency (5,1) and temporal (1,5) patterns with fusion
- DeepNet: Full model combining stem, dual-path blocks, and classifier

Input: Mel-spectrograms (1, 128, 216) - 1 channel, 128 mel bands, 216 time frames
Output: Logits (batch_size, num_classes) - 18 species classifications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StemBlock(nn.Module):
    """Initial feature extraction block with 2 convolutional layers."""

    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, 1, 128, 216)
        Returns:
            Output tensor (batch, 32, 64, 108)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):
    """
    Dual-path block with separate frequency and temporal branches.

    - Frequency branch: (5, 1) kernels for mel band patterns
    - Temporal branch: (1, 5) kernels for time evolution
    - Fusion: Concatenate branches and project to out_channels
    - Shortcut: Projection if dimensions change, identity otherwise
    """

    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__()
        mid_channels = out_channels // 2

        # Frequency branch - captures patterns across mel bands
        self.freq_conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(5, 1), padding=(2, 0), bias=False)
        self.freq_bn1 = nn.BatchNorm2d(mid_channels)
        self.freq_conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(5, 1), padding=(2, 0), bias=False)
        self.freq_bn2 = nn.BatchNorm2d(mid_channels)

        # Temporal branch - captures patterns across time frames
        self.temp_conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.temp_bn1 = nn.BatchNorm2d(mid_channels)
        self.temp_conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.temp_bn2 = nn.BatchNorm2d(mid_channels)

        # Fusion layer - combine both paths
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(out_channels)

        # Downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Projection shortcut if dimensions change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, in_channels, H, W)
        Returns:
            Output tensor (batch, out_channels, H//2, W//2)
        """
        # Frequency branch
        freq = self.freq_conv1(x)
        freq = self.freq_bn1(freq)
        freq = F.relu(freq)
        freq = self.freq_conv2(freq)
        freq = self.freq_bn2(freq)
        freq = F.relu(freq)

        # Temporal branch
        temp = self.temp_conv1(x)
        temp = self.temp_bn1(temp)
        temp = F.relu(temp)
        temp = self.temp_conv2(temp)
        temp = self.temp_bn2(temp)
        temp = F.relu(temp)

        # Fuse both paths
        fused = torch.cat([freq, temp], dim=1)
        fused = self.fusion(fused)
        fused = self.fusion_bn(fused)
        fused = F.relu(fused)
        fused = self.dropout(fused)
        fused = self.pool(fused)

        # Shortcut connection
        shortcut = self.shortcut(x)

        # Residual addition
        out = fused + shortcut
        return out


class DeepNet(nn.Module):
    """
    DEEPNET: Deep Neural Network for Bird Call Classification

    Architecture: Stem → DualPathBlock1 → DualPathBlock2 → Global Avg Pool → Classifier
    Total parameters: ~2.1M

    Input: (batch, 1, 128, 216) - mel-spectrograms
    Output: (batch, num_classes) - logits for 18 species
    """

    def __init__(self, num_classes=18, dropout=0.5):
        super().__init__()

        # Stem block: (1, 128, 216) -> (32, 64, 108)
        self.stem = StemBlock(in_channels=1, out_channels=32)

        # DualPathBlock 1: (32, 64, 108) -> (128, 32, 54)
        self.block1 = DualPathBlock(in_channels=32, out_channels=128, dropout=dropout)

        # DualPathBlock 2: (128, 32, 54) -> (256, 16, 27)
        self.block2 = DualPathBlock(in_channels=128, out_channels=256, dropout=dropout)

        # Global average pooling: (256, 16, 27) -> (256,)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head: (256,) -> (num_classes,)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

        self.num_classes = num_classes
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input mel-spectrograms (batch, 1, 128, 216)

        Returns:
            logits: Class logits (batch, num_classes)
        """
        # Stem
        x = self.stem(x)  # (batch, 32, 64, 108)

        # Dual-path blocks
        x = self.block1(x)  # (batch, 128, 32, 54)
        x = self.block2(x)  # (batch, 256, 16, 27)

        # Global pooling
        x = self.global_pool(x)  # (batch, 256, 1, 1)
        x = torch.flatten(x, 1)  # (batch, 256)

        # Classifier
        logits = self.classifier(x)  # (batch, num_classes)

        return logits

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """Test model with sample input to verify shapes."""
    print("Testing DEEPNET architecture...")

    # Create model
    model = DeepNet(num_classes=18, dropout=0.5)
    print(f"Model created with {model.count_parameters():,} parameters")

    # Test with sample batch
    batch_size = 2
    x = torch.randn(batch_size, 1, 128, 216)
    print(f"Input shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        logits = model(x)

    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, 18), f"Expected (2, 18), got {logits.shape}"

    print("✓ Model test passed!")

    # Print architecture summary
    print("\nArchitecture summary:")
    print(f"  Stem: Conv(1→32) → Conv(32→32) → Pool")
    print(f"  Block1: DualPath(32→128) with frequency + temporal branches")
    print(f"  Block2: DualPath(128→256) with frequency + temporal branches")
    print(f"  Classifier: GlobalAvgPool → Linear(256→18)")
    print(f"  Total parameters: {model.count_parameters():,}")


if __name__ == "__main__":
    test_model()
