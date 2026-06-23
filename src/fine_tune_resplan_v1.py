import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

try:
    import albumentations as A
except ImportError:
    A = None

import segmentation_models_pytorch as smp
from shapely import affinity
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry

try:
    from shapely import make_valid
except ImportError:
    from shapely.validation import make_valid

PROJECT_ROOT = Path("/home/ubuntu/mitunet")
sys.path.insert(0, str(PROJECT_ROOT))

import src.data_loader as data_loader
from src.data_loader import ResPlanSegmentationDataset
from src.model.model import build_mitunet

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)
print(f"Running on: {DEVICE}")

def _make_valid_geometry(geom):
    if not isinstance(geom, BaseGeometry) or geom.is_empty:
        return geom

    try:
        if geom.is_valid:
            return geom
        return make_valid(geom)
    except GEOSException:
        return geom.buffer(0)


def safe_fit_plan_to_canvas(plan, shape=(256, 256), padding=8):
    h, w = shape
    cleaned_plan = {}
    geoms = []

    for key, value in plan.items():
        if isinstance(value, BaseGeometry):
            value = _make_valid_geometry(value)
            cleaned_plan[key] = value
            if not value.is_empty:
                geoms.append(value)
        else:
            cleaned_plan[key] = value

    bounds = [geom.bounds for geom in geoms if len(geom.bounds) == 4]
    if not bounds:
        return cleaned_plan

    minx = min(bound[0] for bound in bounds)
    miny = min(bound[1] for bound in bounds)
    maxx = max(bound[2] for bound in bounds)
    maxy = max(bound[3] for bound in bounds)

    plan_w = max(maxx - minx, 1e-6)
    plan_h = max(maxy - miny, 1e-6)
    scale = min((w - 2 * padding) / plan_w, (h - 2 * padding) / plan_h)

    fitted = {}
    for key, value in cleaned_plan.items():
        if isinstance(value, BaseGeometry):
            if value.is_empty:
                fitted[key] = value
                continue
            geom = affinity.translate(value, xoff=-minx, yoff=-miny)
            geom = affinity.scale(geom, xfact=scale, yfact=scale, origin=(0, 0))
            geom = affinity.translate(geom, xoff=padding, yoff=padding)
            fitted[key] = geom
        else:
            fitted[key] = value

    return fitted


# Patch only this notebook session. This avoids Shapely unary_union failures from invalid polygons
# without changing src/data_loader.py.
data_loader.fit_plan_to_canvas = safe_fit_plan_to_canvas
print("Using notebook-safe fit_plan_to_canvas for invalid geometries")


PKL_PATH = PROJECT_ROOT / "ResPlan.pkl"
IMAGE_SIZE = (256, 256)
PADDING = 8
VALID_FRACTION = 0.2
BATCH_SIZE = 8
NUM_WORKERS = 0

ENCODER_NAME = "mit_b4"
IN_CHANNELS = 3
ENCODER_WEIGHTS = "imagenet"
WARM_START_CHECKPOINT = PROJECT_ROOT / "mitunet.pth"

EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4

SAVE_DIR = PROJECT_ROOT / "checkpoints" / "resplan_v1"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

if A is not None:
    train_augment = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])
else:
    train_augment = None

base_dataset = ResPlanSegmentationDataset(
    pkl_path=str(PKL_PATH),
    shape=IMAGE_SIZE,
    padding=PADDING,
    augment=None,
)

num_samples = len(base_dataset)
valid_size = max(1, int(num_samples * VALID_FRACTION))
train_size = num_samples - valid_size
indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(SEED)).tolist()
train_indices = indices[:train_size]
valid_indices = indices[train_size:]

train_base_dataset = ResPlanSegmentationDataset(
    pkl_path=str(PKL_PATH),
    shape=IMAGE_SIZE,
    classes=base_dataset.classes,
    padding=PADDING,
    augment=train_augment,
)
valid_base_dataset = ResPlanSegmentationDataset(
    pkl_path=str(PKL_PATH),
    shape=IMAGE_SIZE,
    classes=base_dataset.classes,
    padding=PADDING,
    augment=None,
)

train_dataset = Subset(train_base_dataset, train_indices)
valid_dataset = Subset(valid_base_dataset, valid_indices)

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

NUM_CLASSES = base_dataset.num_classes
CLASS_NAMES = ["background"] + base_dataset.classes
IDX_TO_CLASS = {0: "background", **base_dataset.idx_to_class}

print("Classes:", CLASS_NAMES)
print("Class mapping:", {name: idx for idx, name in IDX_TO_CLASS.items()})
print("Num classes:", NUM_CLASSES)
print(f"Train samples: {len(train_dataset)} | Valid samples: {len(valid_dataset)}")

images, masks = next(iter(train_loader))
print("Images:", tuple(images.shape))
print("Masks:", tuple(masks.shape))
print("Mask labels:", torch.unique(masks).tolist())

def load_matching_weights(model, checkpoint_path, device=DEVICE):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Warm-start checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in checkpoint.items()
        if key in model_state and model_state[key].shape == value.shape
    }

    model_state.update(compatible)
    model.load_state_dict(model_state)

    skipped = sorted(set(checkpoint) - set(compatible))
    print(f"Loaded {len(compatible)} tensors from {checkpoint_path.name}")
    print(f"Skipped {len(skipped)} tensors with missing keys or different shapes")


model = build_mitunet(
    encoder_name=ENCODER_NAME,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=IN_CHANNELS,
    classes=NUM_CLASSES,
    decoder_attention_type="scse",
    checkpoint_path=None,
    device=DEVICE,
)

load_matching_weights(model, WARM_START_CHECKPOINT, device=DEVICE)

ce_loss = nn.CrossEntropyLoss()
dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

# NOTE: needs to add tversky loss


def loss_fn(logits, masks):
    return ce_loss(logits, masks) + dice_loss(logits, masks)


optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3,
)

print(f"Model output classes: {NUM_CLASSES}")
print(f"Training input channels: {IN_CHANNELS}")
def prepare_images(images):
    images = images.to(DEVICE, non_blocking=True)
    if images.shape[1] == IN_CHANNELS:
        return images
    if images.shape[1] == 1 and IN_CHANNELS == 3:
        return images.repeat(1, 3, 1, 1)
    raise ValueError(f"Dataset returned {images.shape[1]} channels, but model expects {IN_CHANNELS}")


def empty_stats():
    foreground_classes = max(NUM_CLASSES - 1, 1)
    return {
        "tp": torch.zeros(foreground_classes, dtype=torch.float64),
        "fp": torch.zeros(foreground_classes, dtype=torch.float64),
        "fn": torch.zeros(foreground_classes, dtype=torch.float64),
        "correct": 0,
        "pixels": 0,
    }


def update_stats(stats, logits, masks):
    preds = logits.argmax(dim=1)
    tp, fp, fn, _ = smp.metrics.get_stats(
        preds.detach().cpu(),
        masks.detach().cpu(),
        mode="multiclass",
        num_classes=NUM_CLASSES,
    )

    if NUM_CLASSES > 1:
        stats["tp"] += tp[:, 1:].sum(dim=0).double()
        stats["fp"] += fp[:, 1:].sum(dim=0).double()
        stats["fn"] += fn[:, 1:].sum(dim=0).double()
    else:
        stats["tp"] += tp.sum(dim=0).double()
        stats["fp"] += fp.sum(dim=0).double()
        stats["fn"] += fn.sum(dim=0).double()

    stats["correct"] += (preds == masks).sum().item()
    stats["pixels"] += masks.numel()


def summarize_stats(stats):
    eps = 1e-7
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    union = tp + fp + fn
    valid = union > 0
    per_class_iou = torch.zeros_like(tp)
    per_class_iou[valid] = tp[valid] / (union[valid] + eps)

    micro_iou = tp.sum() / (union.sum() + eps)
    macro_iou = per_class_iou[valid].mean() if valid.any() else torch.tensor(0.0)
    precision = tp.sum() / (tp.sum() + fp.sum() + eps)
    recall = tp.sum() / (tp.sum() + fn.sum() + eps)
    pixel_accuracy = stats["correct"] / max(stats["pixels"], 1)

    return {
        "micro_iou": float(micro_iou),
        "macro_iou": float(macro_iou),
        "precision": float(precision),
        "recall": float(recall),
        "pixel_accuracy": float(pixel_accuracy),
        "per_class_iou": per_class_iou.numpy(),
    }


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    stats = empty_stats()
    start = time.perf_counter()

    desc = "Training" if is_train else "Validation"
    pbar = tqdm(loader, desc=desc)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks in pbar:
            images = prepare_images(images)
            masks = masks.to(DEVICE, non_blocking=True).long()

            logits = model(images)
            loss = loss_fn(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            update_stats(stats, logits, masks)

            scores = summarize_stats(stats)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                miou=f"{scores['micro_iou']:.4f}",
            )

    elapsed = time.perf_counter() - start
    scores = summarize_stats(stats)
    scores["loss"] = total_loss / max(total_samples, 1)
    scores["fps"] = total_samples / max(elapsed, 1e-7)
    return scores

best_model_path = SAVE_DIR / f"mitunet_resplan.pth"
best_metadata_path = SAVE_DIR / f"mitunet_resplan.json"

best_iou = -1.0
history = []

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")

    train_scores = run_epoch(model, train_loader, optimizer=optimizer)
    valid_scores = run_epoch(model, valid_loader, optimizer=None)
    lr_scheduler.step(valid_scores["micro_iou"])

    row = {
        "epoch": epoch,
        "train_loss": train_scores["loss"],
        "valid_loss": valid_scores["loss"],
        "train_micro_iou": train_scores["micro_iou"],
        "valid_micro_iou": valid_scores["micro_iou"],
        "train_macro_iou": train_scores["macro_iou"],
        "valid_macro_iou": valid_scores["macro_iou"],
        "valid_pixel_accuracy": valid_scores["pixel_accuracy"],
        "valid_fps": valid_scores["fps"],
        "lr": optimizer.param_groups[0]["lr"],
    }
    history.append(row)

    print(
        f"Train loss: {row['train_loss']:.4f} | Valid loss: {row['valid_loss']:.4f} | "
        f"Train mIoU: {row['train_micro_iou']:.4f} | Valid mIoU: {row['valid_micro_iou']:.4f} | "
        f"Valid macro IoU: {row['valid_macro_iou']:.4f} | LR: {row['lr']:.2e}"
    )

    if valid_scores["micro_iou"] > best_iou:
        best_iou = valid_scores["micro_iou"]
        torch.save(model.state_dict(), best_model_path)
        metadata = {
            "classes": CLASS_NAMES,
            "idx_to_class": IDX_TO_CLASS,
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "in_channels": IN_CHANNELS,
            "encoder_name": ENCODER_NAME,
            "epoch": epoch,
            "valid_micro_iou": best_iou,
            "valid_macro_iou": valid_scores["macro_iou"],
        }
        with open(best_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved new best model to {best_model_path}")

print(f"\nTraining complete. Best validation micro IoU: {best_iou:.4f}")
