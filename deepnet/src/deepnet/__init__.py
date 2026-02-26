"""
DEEPNET: Deep Neural Network for Bird Call Classification

A custom CNN architecture for classifying bird calls from mel-spectrograms.
"""

__version__ = "0.1.0"

from deepnet.model import DeepNet, DualPathBlock, StemBlock
from deepnet.dataset import BirdCallDataset, build_dataloaders
from deepnet.losses import FocalLoss, LabelSmoothingCE, WeightedCE
from deepnet.utils import get_device, set_seed, load_config

__all__ = [
    "DeepNet",
    "StemBlock",
    "DualPathBlock",
    "BirdCallDataset",
    "build_dataloaders",
    "LabelSmoothingCE",
    "FocalLoss",
    "WeightedCE",
    "get_device",
    "set_seed",
    "load_config",
]
