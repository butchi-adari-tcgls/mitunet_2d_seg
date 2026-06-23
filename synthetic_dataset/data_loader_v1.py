import random
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYNTH_ROOT = PROJECT_ROOT / "Images"
print(SYNTH_ROOT)

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_WORKERS = 0
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# for v1
TARGET_OLD_TO_NEW = {
    7: 1,  # wall
    5: 2,  # door
    8: 2,  # front_door -> door
    6: 3,  # window
}

SYNTH_CLASSES = [
    "background",
    "wall",
    "door",
    "window",
    # "front_door",
]

NUM_CLASSES = len(SYNTH_CLASSES)
CLASS_NAMES = SYNTH_CLASSES
IDX_TO_CLASS = {i: name for i, name in enumerate(SYNTH_CLASSES)}

COLOR_TO_NEW_ID = {
    (40, 40, 40): 1,        # wall
    (160, 82, 45): 2,       # door
    (255, 50, 50): 2,       # front_door -> door
    (135, 206, 250): 3,     # window
}


class DataAugmentation:
    def __init__(
        self,
        image_size=(256, 256),
        min_ops=1,
        max_ops=None,   # None means it can randomly apply all ops
        p=1.0,
    ):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.image_size = image_size  # (H, W)
        self.min_ops = min_ops
        self.max_ops = max_ops
        self.p = p

        self.ops = [
            self.random_horizontal_flip,
            self.random_vertical_flip,
            self.random_rotate_90,
            self.random_small_rotation,
            self.random_scale_translate,
            self.random_perspective,
            self.random_line_thickness,
            self.random_blur,
            self.random_noise,
            self.random_brightness_contrast,
            self.random_gamma,
            self.random_grayscale,
            self.random_threshold,
            self.random_downscale_upscale,
            self.random_jpeg_compression,
            self.random_clutter,
        ]

    def __call__(self, image, mask):
        """
        image: H,W,3 numpy image
        mask:  H,W integer class mask

        Returns dict so it works with:
        augmented = self.augment(image=image, mask=mask)
        """

        return_float_01 = (
            image.dtype != np.uint8 and np.max(image) <= 1.0
        )

        image = self._to_uint8(image)
        mask = mask.astype(np.uint8)

        if random.random() > self.p:
            return {
                "image": image.astype(np.float32) / 255.0 if return_float_01 else image,
                "mask": mask.astype(np.int64),
            }

        ops = self.ops.copy()
        random.shuffle(ops)

        max_ops = len(ops) if self.max_ops is None else min(self.max_ops, len(ops))
        min_ops = min(self.min_ops, max_ops)
        n_ops = random.randint(min_ops, max_ops)

        for op in ops[:n_ops]:
            image, mask = op(image, mask)

        image, mask = self._resize_back(image, mask)

        if return_float_01:
            image = image.astype(np.float32) / 255.0

        return {
            "image": image,
            "mask": mask.astype(np.int64),
        }

    def _to_uint8(self, image):
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        if image.shape[-1] == 4:
            image = image[..., :3]

        if image.dtype == np.uint8:
            return image

        if np.max(image) <= 1.0:
            image = image * 255.0

        return np.clip(image, 0, 255).astype(np.uint8)

    def _resize_back(self, image, mask):
        h, w = self.image_size

        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        return image, mask

    def _warp_affine(self, image, mask, matrix):
        h, w = image.shape[:2]

        image = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        mask = cv2.warpAffine(
            mask,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return image, mask

    def random_horizontal_flip(self, image, mask):
        image = np.ascontiguousarray(np.fliplr(image))
        mask = np.ascontiguousarray(np.fliplr(mask))
        return image, mask

    def random_vertical_flip(self, image, mask):
        image = np.ascontiguousarray(np.flipud(image))
        mask = np.ascontiguousarray(np.flipud(mask))
        return image, mask

    def random_rotate_90(self, image, mask):
        k = random.choice([1, 2, 3])
        image = np.rot90(image, k).copy()
        mask = np.rot90(mask, k).copy()
        return image, mask

    def random_small_rotation(self, image, mask):
        h, w = image.shape[:2]

        angle = random.uniform(-8, 8)
        scale = random.uniform(0.95, 1.05)

        matrix = cv2.getRotationMatrix2D(
            center=(w / 2, h / 2),
            angle=angle,
            scale=scale,
        )

        return self._warp_affine(image, mask, matrix)

    def random_scale_translate(self, image, mask):
        h, w = image.shape[:2]

        scale = random.uniform(0.75, 1.25)
        tx = random.uniform(-0.08, 0.08) * w
        ty = random.uniform(-0.08, 0.08) * h

        matrix = np.array([
            [scale, 0, (1 - scale) * w / 2 + tx],
            [0, scale, (1 - scale) * h / 2 + ty],
        ], dtype=np.float32)

        return self._warp_affine(image, mask, matrix)

    def random_perspective(self, image, mask):
        h, w = image.shape[:2]

        max_dx = int(w * random.uniform(0.02, 0.07))
        max_dy = int(h * random.uniform(0.02, 0.07))

        src = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ])

        dst = np.float32([
            [random.randint(0, max_dx), random.randint(0, max_dy)],
            [w - 1 - random.randint(0, max_dx), random.randint(0, max_dy)],
            [w - 1 - random.randint(0, max_dx), h - 1 - random.randint(0, max_dy)],
            [random.randint(0, max_dx), h - 1 - random.randint(0, max_dy)],
        ])

        matrix = cv2.getPerspectiveTransform(src, dst)

        image = cv2.warpPerspective(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        mask = cv2.warpPerspective(
            mask,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return image, mask

    def random_line_thickness(self, image, mask):
        mode = random.choice(["thicken", "thin"])
        ksize = random.choice([2, 3])
        kernel = np.ones((ksize, ksize), np.uint8)

        # Image: black/dark floorplan lines
        if mode == "thicken":
            image = cv2.erode(image, kernel, iterations=1)
        else:
            image = cv2.dilate(image, kernel, iterations=1)

        # Mask: class regions
        new_mask = np.zeros_like(mask)

        # wall first, then window/door overwrite
        for cls in [1, 3, 2]:
            cls_mask = (mask == cls).astype(np.uint8)

            if mode == "thicken":
                cls_mask = cv2.dilate(cls_mask, kernel, iterations=1)
            else:
                cls_mask = cv2.erode(cls_mask, kernel, iterations=1)

            new_mask[cls_mask > 0] = cls

        return image, new_mask

    def random_blur(self, image, mask):
        blur_type = random.choice(["gaussian", "median", "motion"])

        if blur_type == "gaussian":
            k = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (k, k), 0)

        elif blur_type == "median":
            k = random.choice([3, 5])
            image = cv2.medianBlur(image, k)

        else:
            k = random.choice([3, 5])
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0 / k
            image = cv2.filter2D(image, -1, kernel)

        return image, mask

    def random_noise(self, image, mask):
        noise_type = random.choice(["gaussian", "salt_pepper"])

        if noise_type == "gaussian":
            sigma = random.uniform(5, 22)
            noise = np.random.normal(0, sigma, image.shape)
            image = image.astype(np.float32) + noise
            image = np.clip(image, 0, 255).astype(np.uint8)

        else:
            h, w = image.shape[:2]
            amount = random.uniform(0.001, 0.008)
            n = int(amount * h * w)

            ys = np.random.randint(0, h, n)
            xs = np.random.randint(0, w, n)

            value = random.choice([0, 255])
            image[ys, xs] = value

        return image, mask

    def random_brightness_contrast(self, image, mask):
        alpha = random.uniform(0.55, 1.55)  # contrast
        beta = random.uniform(-45, 45)      # brightness

        image = image.astype(np.float32) * alpha + beta
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image, mask

    def random_gamma(self, image, mask):
        gamma = random.uniform(0.6, 1.6)

        image_float = image.astype(np.float32) / 255.0
        image_float = np.power(image_float, gamma)
        image = np.clip(image_float * 255.0, 0, 255).astype(np.uint8)

        return image, mask

    def random_grayscale(self, image, mask):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.stack([gray, gray, gray], axis=-1)
        return image, mask

    def random_threshold(self, image, mask):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        thresh = random.randint(110, 210)

        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        image = np.stack([bw, bw, bw], axis=-1)

        return image, mask

    def random_downscale_upscale(self, image, mask):
        h, w = image.shape[:2]

        scale = random.uniform(0.45, 0.85)
        small_w = max(16, int(w * scale))
        small_h = max(16, int(h * scale))

        image = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        return image, mask

    def random_jpeg_compression(self, image, mask):
        quality = random.randint(30, 85)

        pil_img = Image.fromarray(image)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        image = np.array(Image.open(buffer).convert("RGB")).astype(np.uint8)

        return image, mask

    def random_clutter(self, image, mask):
        """
        Adds random text/furniture-like clutter to image only.
        Mask is unchanged.
        """
        h, w = image.shape[:2]
        image = image.copy()

        count = random.randint(3, 12)

        for _ in range(count):
            color_value = random.randint(80, 210)
            color = (color_value, color_value, color_value)

            x1 = random.randint(0, w - 1)
            y1 = random.randint(0, h - 1)

            if random.random() < 0.5:
                x2 = min(w - 1, x1 + random.randint(5, 40))
                y2 = min(h - 1, y1 + random.randint(2, 15))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=-1)
            else:
                x2 = min(w - 1, x1 + random.randint(10, 80))
                y2 = min(h - 1, y1 + random.randint(-20, 20))
                thickness = random.choice([1, 1, 2])
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)

        return image, mask


class SyntheticFloorplanSegDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_size=(256, 256),
        use_bw=False,
        augment=None,
        normalize=True,
    ):

        use_bw = True
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
            # mask_path = self.mask_dir / f"mask_{idx}.npy"
            mask_path = self.mask_dir / f"mask_{idx}.png"
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No image/mask pairs found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("RGB")

        image = TF.resize(image, self.image_size)
        mask_img = TF.resize(
            mask_img,
            self.image_size,
            interpolation=Image.NEAREST
        )

        image = np.array(image).astype(np.float32)
        mask_rgb = np.array(mask_img)

        # Everything starts as background
        mask = np.zeros(mask_rgb.shape[:2], dtype=np.int64)

        COLOR_TO_NEW_ID = {
            (40, 40, 40): 1,        # wall
            (120, 120, 120): 1,     # wall, if gray wall appears
            (160, 82, 45): 2,       # door
            (255, 50, 50): 2,       # front_door -> door
            (135, 206, 250): 3,     # window
        }

        for color, class_id in COLOR_TO_NEW_ID.items():
            color = np.array(color, dtype=np.uint8)
            mask[np.all(mask_rgb == color, axis=-1)] = class_id

        if self.normalize:
            image = image / 255.0

        if self.augment is not None:
            augmented = self.augment(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure contiguous arrays before tensor conversion (augmentation
        # ops like flips/rot90 can leave non-contiguous views).
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).long()

        return image, mask


def build_dataloaders(
    root_dir=SYNTH_ROOT,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    valid_fraction=0.2,
    use_bw=False,
    seed=SEED,
):
    """
    Builds train/valid loaders.

    Augmentation is applied to the TRAIN split only. Validation uses a clean
    (un-augmented) view of the same data, split by index so there is no leakage.
    """

    augment = DataAugmentation(
        image_size=image_size,
        min_ops=1,
        max_ops=4,     # cap how many ops stack per sample (None = all 16)
        p=0.9,         # probability any augmentation is applied at all
    )

    # Two views over the same files: train gets augmentation, valid does not.
    train_base = SyntheticFloorplanSegDataset(
        root_dir=root_dir,
        image_size=image_size,
        use_bw=use_bw,
        augment=augment,
    )
    valid_base = SyntheticFloorplanSegDataset(
        root_dir=root_dir,
        image_size=image_size,
        use_bw=use_bw,
        augment=None,
    )

    n = len(train_base)
    valid_size = max(1, int(n * valid_fraction))
    train_size = n - valid_size

    # Deterministic, non-overlapping index split shared by both views.
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    train_idx = perm[:train_size]
    valid_idx = perm[train_size:]

    train_dataset = Subset(train_base, train_idx)
    valid_dataset = Subset(valid_base, valid_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
    )

    return train_loader, valid_loader


if __name__ == "__main__":
    train_loader, valid_loader = build_dataloaders(
        root_dir=SYNTH_ROOT,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        valid_fraction=0.2,
        use_bw=False,
        seed=SEED,
    )

    print("Train batches:", len(train_loader))
    print("Valid batches:", len(valid_loader))

    images, masks = next(iter(train_loader))
    print("Images:", images.shape)
    print("Masks:", masks.shape)
    print("Mask labels:", torch.unique(masks))
    print("Num classes:", NUM_CLASSES)