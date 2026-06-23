import json
import os
import re
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import segmentation_models_pytorch as smp

PROJECT_ROOT = Path("/home/ubuntu/mitunet")
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.model import build_mitunet

# ── Reproducibility ───────────────────────────────────────────────────────────
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

# ── Config ────────────────────────────────────────────────────────────────────
ENCODER_NAME    = "mit_b4"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS     = 3
NUM_CLASSES     = 4
IMAGE_SIZE      = (256, 256)

BATCH_SIZE      = 8
NUM_WORKERS     = 0
VALID_FRACTION  = 0.2
EPOCHS          = 15
LR              = 5e-5
WEIGHT_DECAY    = 1e-4

RESPLAN_CKPT    = PROJECT_ROOT / "checkpoints" / "resplan" / "mitunet_resplan.pth"
SAVE_DIR        = PROJECT_ROOT / "checkpoints" / "synthetic_v1"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

version = "v1"

BEST_MODEL_PATH = SAVE_DIR / f"mitunet_synthetic_{version}.pth"
BEST_META_PATH  = SAVE_DIR / f"mitunet_synthetic_{version}.json"

SYNTH_DATA_ROOT = PROJECT_ROOT / "synthetic_dataset" / "plans"
SYNTH_LOADER_PY = PROJECT_ROOT / "synthetic_dataset" / "data_loader.py"

# Path to a single test image for visual inspection after every epoch.
# Change this to any image you want to track visually.
TEST_IMAGE_PATH = PROJECT_ROOT / "test_images" / "image_0.png"

# ── TensorBoard ───────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "runs" / "synthetic_finetune_v1"
writer  = SummaryWriter(log_dir=str(LOG_DIR))
print(f"TensorBoard logs → {LOG_DIR}")
print(f"  Run: tensorboard --logdir {LOG_DIR}")

# ── Load synthetic dataset ────────────────────────────────────────────────────
code = SYNTH_LOADER_PY.read_text()
code = re.split(r"\nfull_dataset\s*=", code)[0]

namespace = {"PROJECT_ROOT": PROJECT_ROOT}
exec(code, namespace)
SyntheticFloorplanSegDataset = namespace["SyntheticFloorplanSegDataset"]

print("\n--- DATA LOADER LABEL INFO ---")
print("SYNTH_LOADER_PY:", SYNTH_LOADER_PY)
print("SYNTH_DATA_ROOT:", SYNTH_DATA_ROOT)

print("SYNTH_CLASSES:", namespace.get("SYNTH_CLASSES"))
print("CLASS_NAMES:", namespace.get("CLASS_NAMES"))
print("IDX_TO_CLASS:", namespace.get("IDX_TO_CLASS"))
print("TARGET_OLD_TO_NEW:", namespace.get("TARGET_OLD_TO_NEW"))
print("NUM_CLASSES from loader:", namespace.get("NUM_CLASSES"))
print("------------------------------\n")

full_dataset = SyntheticFloorplanSegDataset(
    root_dir=SYNTH_DATA_ROOT,
    image_size=IMAGE_SIZE,
    use_bw=True,
    augment=None,
)

valid_size = max(1, int(len(full_dataset) * VALID_FRACTION))
train_size = len(full_dataset) - valid_size

train_dataset, valid_dataset = random_split(
    full_dataset,
    [train_size, valid_size],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

print(f"Total: {len(full_dataset)} | Train: {len(train_dataset)} | Valid: {len(valid_dataset)}")

images, masks = next(iter(train_loader))
print("\n--- ACTUAL TRAINING BATCH INFO ---")
print("Image batch:", images.shape)
print("Mask batch:", masks.shape)
print("Mask labels in batch:", torch.unique(masks).tolist())

unique, counts = torch.unique(masks, return_counts=True)
for u, c in zip(unique.tolist(), counts.tolist()):
    print(f"label {u}: {c} pixels")
print("----------------------------------\n")

images, masks = next(iter(train_loader))
print(f"Image batch: {images.shape} | Mask batch: {masks.shape} | Labels: {torch.unique(masks).tolist()}")

# ── Model ─────────────────────────────────────────────────────────────────────
def load_matching_weights(model, checkpoint_path):
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    model_state = model.state_dict()
    compatible  = {k: v for k, v in ckpt.items() if k in model_state and model_state[k].shape == v.shape}
    skipped     = sorted(set(ckpt) - set(compatible))

    model_state.update(compatible)
    model.load_state_dict(model_state)
    print(f"Loaded {len(compatible)} tensors | Skipped {len(skipped)} (shape mismatch / missing)")


model = build_mitunet(
    encoder_name=ENCODER_NAME,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=IN_CHANNELS,
    classes=NUM_CLASSES,
    decoder_attention_type="scse",
    checkpoint_path=None,
    device=DEVICE,
)

load_matching_weights(model, RESPLAN_CKPT)

# ── Loss ──────────────────────────────────────────────────────────────────────
ce_loss      = nn.CrossEntropyLoss()
dice_loss    = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
tversky_loss = smp.losses.TverskyLoss(mode="multiclass", from_logits=True, alpha=0.3, beta=0.7)

def loss_fn(logits, masks):
    l_ce      = ce_loss(logits, masks)
    l_dice    = dice_loss(logits, masks)
    l_tversky = tversky_loss(logits, masks)
    return l_ce + l_dice + l_tversky, l_ce, l_dice, l_tversky

# ── Optimizer & scheduler ─────────────────────────────────────────────────────
optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

# ── Metrics helpers ───────────────────────────────────────────────────────────
def prepare_images(images):
    images = images.to(DEVICE, non_blocking=True)
    if images.shape[1] == IN_CHANNELS:
        return images
    if images.shape[1] == 1 and IN_CHANNELS == 3:
        return images.repeat(1, 3, 1, 1)
    raise ValueError(f"Expected {IN_CHANNELS} channels, got {images.shape[1]}")


def empty_stats():
    fg = max(NUM_CLASSES - 1, 1)
    return {
        "tp": torch.zeros(fg, dtype=torch.float64),
        "fp": torch.zeros(fg, dtype=torch.float64),
        "fn": torch.zeros(fg, dtype=torch.float64),
        "correct": 0,
        "pixels": 0,
        "ce": 0.0, "dice": 0.0, "tversky": 0.0,
    }


def update_stats(stats, logits, masks, l_ce, l_dice, l_tversky):
    preds = logits.argmax(dim=1)
    tp, fp, fn, _ = smp.metrics.get_stats(
        preds.detach().cpu(),
        masks.detach().cpu(),
        mode="multiclass",
        num_classes=NUM_CLASSES,
    )
    stats["tp"]      += tp[:, 1:].sum(dim=0).double()
    stats["fp"]      += fp[:, 1:].sum(dim=0).double()
    stats["fn"]      += fn[:, 1:].sum(dim=0).double()
    stats["correct"] += (preds == masks).sum().item()
    stats["pixels"]  += masks.numel()
    stats["ce"]      += l_ce.item()
    stats["dice"]    += l_dice.item()
    stats["tversky"] += l_tversky.item()


def summarize_stats(stats, num_batches):
    eps   = 1e-7
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    union = tp + fp + fn
    valid = union > 0

    per_class_iou        = torch.zeros_like(tp)
    per_class_iou[valid] = tp[valid] / (union[valid] + eps)

    # Per-class precision & recall
    per_class_precision = tp / (tp + fp + eps)
    per_class_recall    = tp / (tp + fn + eps)
    per_class_f1        = 2 * per_class_precision * per_class_recall / (per_class_precision + per_class_recall + eps)

    micro_iou      = tp.sum() / (union.sum() + eps)
    macro_iou      = per_class_iou[valid].mean() if valid.any() else torch.tensor(0.0)
    precision      = tp.sum() / (tp.sum() + fp.sum() + eps)
    recall         = tp.sum() / (tp.sum() + fn.sum() + eps)
    f1             = 2 * precision * recall / (precision + recall + eps)
    pixel_accuracy = stats["correct"] / max(stats["pixels"], 1)

    return {
        "micro_iou":          float(micro_iou),
        "macro_iou":          float(macro_iou),
        "precision":          float(precision),
        "recall":             float(recall),
        "f1":                 float(f1),
        "pixel_accuracy":     float(pixel_accuracy),
        "per_class_iou":      per_class_iou.numpy(),
        "per_class_precision": per_class_precision.numpy(),
        "per_class_recall":   per_class_recall.numpy(),
        "per_class_f1":       per_class_f1.numpy(),
        "ce_loss":            stats["ce"]      / max(num_batches, 1),
        "dice_loss":          stats["dice"]    / max(num_batches, 1),
        "tversky_loss":       stats["tversky"] / max(num_batches, 1),
    }

# ── Train / eval loop ─────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_samples, num_batches = 0.0, 0, 0
    stats = empty_stats()
    pbar  = tqdm(loader, desc="Train" if is_train else "Valid")

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, masks in pbar:
            images = prepare_images(images)
            masks  = masks.to(DEVICE, non_blocking=True).long()

            logits               = model(images)
            loss, l_ce, l_dice, l_tversky = loss_fn(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            bs             = images.size(0)
            total_loss    += loss.item() * bs
            total_samples += bs
            num_batches   += 1
            update_stats(stats, logits, masks, l_ce, l_dice, l_tversky)

            scores = summarize_stats(stats, num_batches)
            pbar.set_postfix(loss=f"{loss.item():.4f}", miou=f"{scores['micro_iou']:.4f}")

    scores         = summarize_stats(stats, num_batches)
    scores["loss"] = total_loss / max(total_samples, 1)
    return scores

# ── Test image prediction helper ──────────────────────────────────────────────
def predict_test_image(model, image_path):
    """
    Load a single image, run inference, return (input_tensor, pred_mask).
    Handles grayscale and RGB images.
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_tensor = TF.to_tensor(img).unsqueeze(0)         # (1, 3, H, W)
    img_tensor = img_tensor.to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        pred   = logits.argmax(dim=1).squeeze(0).cpu()  # (H, W)

    return img_tensor.squeeze(0).cpu(), pred


def make_prediction_figure(img_tensor, pred_mask):
    """Build a matplotlib figure with input + coloured prediction side by side."""
    cmap = plt.get_cmap("tab20", NUM_CLASSES)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_tensor.permute(1, 2, 0).numpy())
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(pred_mask.numpy(), cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    plt.tight_layout()
    return fig


def log_to_tensorboard(writer, tag_prefix, scores, epoch):
    """Write all metrics under train/ or valid/ namespace."""
    writer.add_scalar(f"{tag_prefix}/loss",           scores["loss"],          epoch)
    writer.add_scalar(f"{tag_prefix}/ce_loss",        scores["ce_loss"],       epoch)
    writer.add_scalar(f"{tag_prefix}/dice_loss",      scores["dice_loss"],     epoch)
    writer.add_scalar(f"{tag_prefix}/tversky_loss",   scores["tversky_loss"],  epoch)
    writer.add_scalar(f"{tag_prefix}/micro_iou",      scores["micro_iou"],     epoch)
    writer.add_scalar(f"{tag_prefix}/macro_iou",      scores["macro_iou"],     epoch)
    writer.add_scalar(f"{tag_prefix}/precision",      scores["precision"],     epoch)
    writer.add_scalar(f"{tag_prefix}/recall",         scores["recall"],        epoch)
    writer.add_scalar(f"{tag_prefix}/f1",             scores["f1"],            epoch)
    writer.add_scalar(f"{tag_prefix}/pixel_accuracy", scores["pixel_accuracy"], epoch)

    # Per-class IoU, precision, recall, f1
    for i, (iou, prec, rec, f1) in enumerate(zip(
        scores["per_class_iou"],
        scores["per_class_precision"],
        scores["per_class_recall"],
        scores["per_class_f1"],
    ), start=1):
        writer.add_scalar(f"{tag_prefix}/per_class/iou/class_{i:02d}",       iou,  epoch)
        writer.add_scalar(f"{tag_prefix}/per_class/precision/class_{i:02d}", prec, epoch)
        writer.add_scalar(f"{tag_prefix}/per_class/recall/class_{i:02d}",    rec,  epoch)
        writer.add_scalar(f"{tag_prefix}/per_class/f1/class_{i:02d}",        f1,   epoch)

# ── Training loop ─────────────────────────────────────────────────────────────
best_iou = -1.0
history  = []

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")

    train_scores = run_epoch(model, train_loader, optimizer=optimizer)
    valid_scores = run_epoch(model, valid_loader, optimizer=None)

    current_lr = optimizer.param_groups[0]["lr"]
    lr_scheduler.step(valid_scores["micro_iou"])

    # ── TensorBoard logging ───────────────────────────────────────────────────
    log_to_tensorboard(writer, "train", train_scores, epoch)
    log_to_tensorboard(writer, "valid", valid_scores, epoch)
    writer.add_scalar("lr", current_lr, epoch)

    # Overlay train vs valid mIoU on the same chart for easy comparison
    writer.add_scalars("compare/micro_iou",  {"train": train_scores["micro_iou"],  "valid": valid_scores["micro_iou"]},  epoch)
    writer.add_scalars("compare/macro_iou",  {"train": train_scores["macro_iou"],  "valid": valid_scores["macro_iou"]},  epoch)
    writer.add_scalars("compare/loss",       {"train": train_scores["loss"],        "valid": valid_scores["loss"]},       epoch)
    writer.add_scalars("compare/f1",         {"train": train_scores["f1"],          "valid": valid_scores["f1"]},         epoch)

    # ── Test image prediction every epoch ────────────────────────────────────
    if TEST_IMAGE_PATH.exists():
        img_tensor, pred_mask = predict_test_image(model, TEST_IMAGE_PATH)
        fig = make_prediction_figure(img_tensor, pred_mask)
        writer.add_figure("test_image/prediction", fig, global_step=epoch)
        plt.close(fig)
    else:
        print(f"  [warn] TEST_IMAGE_PATH not found: {TEST_IMAGE_PATH} — skipping visual log")

    # ── Console summary ───────────────────────────────────────────────────────
    print(
        f"  Train → loss: {train_scores['loss']:.4f}  mIoU: {train_scores['micro_iou']:.4f}  F1: {train_scores['f1']:.4f}\n"
        f"  Valid → loss: {valid_scores['loss']:.4f}  mIoU: {valid_scores['micro_iou']:.4f}  F1: {valid_scores['f1']:.4f}  "
        f"PixAcc: {valid_scores['pixel_accuracy']:.4f}  LR: {current_lr:.2e}"
    )

    row = {
        "epoch":           epoch,
        "train_loss":      train_scores["loss"],
        "valid_loss":      valid_scores["loss"],
        "train_micro_iou": train_scores["micro_iou"],
        "valid_micro_iou": valid_scores["micro_iou"],
        "train_macro_iou": train_scores["macro_iou"],
        "valid_macro_iou": valid_scores["macro_iou"],
        "valid_f1":        valid_scores["f1"],
        "valid_pixel_acc": valid_scores["pixel_accuracy"],
        "lr":              current_lr,
    }
    history.append(row)

    # ── Save best model ───────────────────────────────────────────────────────
    if valid_scores["micro_iou"] > best_iou:
        best_iou = valid_scores["micro_iou"]
        torch.save(model.state_dict(), BEST_MODEL_PATH)

        metadata = {
            "num_classes":     NUM_CLASSES,
            "image_size":      IMAGE_SIZE,
            "in_channels":     IN_CHANNELS,
            "encoder_name":    ENCODER_NAME,
            "epoch":           epoch,
            "valid_micro_iou": best_iou,
            "valid_macro_iou": valid_scores["macro_iou"],
            "valid_f1":        valid_scores["f1"],
            "valid_pixel_acc": valid_scores["pixel_accuracy"],
        }
        with open(BEST_META_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Saved best model (mIoU={best_iou:.4f}) → {BEST_MODEL_PATH}")

writer.close()
print(f"\nTraining complete. Best valid mIoU: {best_iou:.4f}")
print(f"TensorBoard: tensorboard --logdir {LOG_DIR}")