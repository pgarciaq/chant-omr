"""PyTorch dataset for (image, GABC) training pairs."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset


class ChantOMRDataset(Dataset):
    """Dataset of (rendered score image, GABC notation) pairs.

    Each sample is loaded from a pre-rendered directory structure:
        data/rendered/
            00001.png
            00001.gabc
            00002.png
            00002.gabc
            ...

    During training, augmentation is applied on-the-fly to the images.
    """

    def __init__(
        self,
        data_dir: Path,
        tokenizer=None,
        augment: bool = True,
        target_width: int = 1050,
        target_height: int = 1485,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.augment = augment
        self.target_width = target_width
        self.target_height = target_height
        self.samples = sorted(self.data_dir.glob("*.gabc"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raise NotImplementedError("Dataset loading not yet implemented")
