"""DocTamper LMDB PyTorch Dataset implementation.

Provides lazy-initialized, multi-worker safe access to DocTamper LMDB datasets
with synchronized spatial transforms and mask normalization.
"""

import io
import os
import random
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class DocTamperLMDBDataset(Dataset):
    """PyTorch Dataset for DocTamper LMDB subsets.

    Features:
    - Lazy LMDB opening per worker process (safe for DataLoader multi-processing)
    - O(1) direct key indexing using 'image-%09d' and 'label-%09d'
    - Dynamic mask normalization: (mask > 0).astype(np.float32)
    - Synchronized joint spatial transformations for image and mask
    """

    def __init__(
        self,
        lmdb_dir: str,
        is_train: bool = False,
        horizontal_flip_prob: float = 0.5,
        target_size: tuple = (512, 512),
        normalize_mean: tuple = (0.485, 0.456, 0.406),
        normalize_std: tuple = (0.229, 0.224, 0.225),
    ):
        """Args:

        lmdb_dir: Path to the directory containing data.mdb and lock.mdb.
        is_train: If True, enables data augmentations (e.g. random horizontal
        flip).
        horizontal_flip_prob: Probability of horizontal flip if is_train is
        True.
        target_size: Expected image and mask (H, W). Default (512, 512).
        normalize_mean: ImageNet mean for RGB normalization.
        normalize_std: ImageNet std for RGB normalization.
        """
        super().__init__()
        self.lmdb_dir = os.path.abspath(lmdb_dir)
        self.is_train = is_train
        self.horizontal_flip_prob = horizontal_flip_prob
        self.target_size = target_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        if not os.path.exists(self.lmdb_dir):
            raise FileNotFoundError(f"LMDB directory not found: {self.lmdb_dir}")

        # Probe number of samples once during init using a temporary read-only handle
        temp_env = lmdb.open(
            self.lmdb_dir,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_dbs=1,
        )
        with temp_env.begin(write=False) as txn:
            num_samples_raw = txn.get(b"num-samples")
            if num_samples_raw is not None:
                self.num_samples = int(num_samples_raw.decode("utf-8"))
            else:
                self.num_samples = temp_env.stat()["entries"] // 2
        temp_env.close()

        # Lazy environment handles per worker
        self.env = None
        self.txn = None

    def _init_db(self):
        """Lazily initialize LMDB handle for current process/worker."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_dir,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_dbs=1,
            )
            self.txn = self.env.begin(write=False)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.num_samples}"
            )

        self._init_db()

        img_key = f"image-{idx:09d}".encode("utf-8")
        mask_key = f"label-{idx:09d}".encode("utf-8")

        img_bytes = self.txn.get(img_key)
        mask_bytes = self.txn.get(mask_key)

        if img_bytes is None or mask_bytes is None:
            raise KeyError(f"Failed to retrieve sample index {idx} from LMDB")

        # Decode directly from memory byte streams
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        mask_pil = Image.open(io.BytesIO(mask_bytes)).convert("L")

        # Verify dimensions
        if img_pil.size != (
            self.target_size[1],
            self.target_size[0],
        ) or mask_pil.size != (self.target_size[1], self.target_size[0]):
            img_pil = img_pil.resize(
                (self.target_size[1], self.target_size[0]),
                resample=Image.Resampling.BILINEAR,
            )
            mask_pil = mask_pil.resize(
                (self.target_size[1], self.target_size[0]),
                resample=Image.Resampling.NEAREST,
            )

        # Synchronized Geometric Augmentations
        if self.is_train and self.horizontal_flip_prob > 0:
            if random.random() < self.horizontal_flip_prob:
                img_pil = TF.hflip(img_pil)
                mask_pil = TF.hflip(mask_pil)

        # Convert Image to Tensor [3, H, W] in [0, 1]
        img_tensor = TF.to_tensor(img_pil)

        # Convert Mask: strictly binary float [1, H, W]
        mask_np = (np.asarray(mask_pil, dtype=np.float32) > 0.0).astype(
            np.float32
        )
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

        # Normalized RGB Tensor for pretrained backbone
        normalized_img = TF.normalize(
            img_tensor, mean=self.normalize_mean, std=self.normalize_std
        )

        return {
            "image": normalized_img,  # [3, H, W] normalized
            "raw_image": img_tensor,  # [3, H, W] in [0, 1] for SRM / DCT or visualization
            "mask": mask_tensor,  # [1, H, W] binary float {0.0, 1.0}
            "index": idx,
        }

    def __del__(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
