from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF

SYNTH_ROOT = PROJECT_ROOT / "Images"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_WORKERS = 0

SYNTH_CLASSES = [
    "background",
    "bedroom",
    "bathroom",
    "kitchen",
    "wall",
    "door",
    "window",
    "front_door",
    "balcony",
    "living_space",
    "lift",
    "corridor",
    "apartment",
    "master_bedroom",
]

NUM_CLASSES = len(SYNTH_CLASSES)
IDX_TO_CLASS = {i: name for i, name in enumerate(SYNTH_CLASSES)}
CLASS_NAMES = SYNTH_CLASSES


class SyntheticFloorplanSegDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_size=(256, 256),
        use_bw=False,
        augment=None,
        normalize=True,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "floor_plans"
        self.mask_dir = self.root_dir / "seg_masks"
        self.image_size = image_size
        self.use_bw = use_bw
        self.augment = augment
        self.normalize = normalize

        suffix = "_bw.png" if use_bw else ".png"
        self.image_paths = sorted([
            p for p in self.image_dir.glob(f"image_*{suffix}")
            if not p.name.endswith("_bw.png") or use_bw
        ])

        self.samples = []
        for img_path in self.image_paths:
            idx = img_path.stem.replace("image_", "").replace("_bw", "")
            mask_path = self.mask_dir / f"mask_{idx}.npy"
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No image/mask pairs found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = np.load(mask_path).astype(np.int64)

        image = TF.resize(image, self.image_size)
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = TF.resize(mask, self.image_size, interpolation=Image.NEAREST)

        image = np.array(image).astype(np.float32)
        mask = np.array(mask).astype(np.int64)

        if self.normalize:
            image = image / 255.0

        if self.augment is not None:
            augmented = self.augment(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).long()

        return image, mask


full_dataset = SyntheticFloorplanSegDataset(
    root_dir=SYNTH_ROOT,
    image_size=IMAGE_SIZE,
    use_bw=False,
    augment=None,
)

valid_fraction = 0.2
valid_size = max(1, int(len(full_dataset) * valid_fraction))
train_size = len(full_dataset) - valid_size

train_dataset, valid_dataset = random_split(
    full_dataset,
    [train_size, valid_size],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
)

images, masks = next(iter(train_loader))
print("Images:", images.shape)
print("Masks:", masks.shape)
print("Mask labels:", torch.unique(masks))
print("Num classes:", NUM_CLASSES)
