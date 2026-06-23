import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for servers / no display
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from synthetic_dataset.data_loader_v1 import (
    SyntheticFloorplanSegDataset,
    DataAugmentation,
)

import segmentation_models_pytorch as smp

from model.model import build_mitunet


# ----------------------------
# Color map (shared by mask saving + tensorboard figures)
# ----------------------------
COLOR_MAP = {
    0: (200, 200, 200),  # background
    1: (40, 40, 40),     # wall
    2: (160, 82, 45),    # door
    3: (135, 206, 250),  # window
}


# ----------------------------
# Utils
# ----------------------------
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mask_to_rgb(mask):
    """Convert an integer class mask (H, W) into an RGB image (H, W, 3)."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        rgb[mask == cls] = color
    return rgb


def save_mask_png(mask, save_path):
    from PIL import Image

    rgb = mask_to_rgb(mask)
    Image.fromarray(rgb).save(save_path)


def load_single_image(image_path, image_size, device):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))

    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    return image_tensor


# ----------------------------
# Dataset
# ----------------------------
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_size=(256, 256), normalize=True):
        from PIL import Image

        self.root_dir = Path(root_dir)
        self.plans_dir = self.root_dir / "plans"
        self.masks_dir = self.root_dir / "masks"

        self.image_size = image_size
        self.normalize = normalize
        self.Image = Image

        self.target_colors_to_new = {
            (200, 200, 200): 0,
            (40, 40, 40): 1,
            (160, 82, 45): 2,
            (255, 50, 50): 2,
            (135, 206, 250): 3,
        }

        self.image_paths = sorted(
            list(self.plans_dir.glob("*.png")) +
            list(self.plans_dir.glob("*.jpg")) +
            list(self.plans_dir.glob("*.jpeg"))
        )

        self.samples = []
        for idx, image_path in enumerate(self.image_paths):
            mask_path = image_path  # self.masks_dir / f"mask_{idx + 1:04d}.png"
            if mask_path.exists():
                self.samples.append((image_path, mask_path))
            else:
                print(f"[warn] Missing mask for {image_path.name}: {mask_path}")

        if len(self.samples) == 0:
            raise RuntimeError(f"No image/mask pairs found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        image = self.Image.open(image_path).convert("RGB")
        mask = self.Image.open(mask_path).convert("RGB")

        image = image.resize((self.image_size[1], self.image_size[0]))
        mask = mask.resize(
            (self.image_size[1], self.image_size[0]),
            resample=self.Image.NEAREST,
        )

        image = np.array(image).astype(np.float32)
        mask_rgb = np.array(mask).astype(np.uint8)

        new_mask = np.zeros(mask_rgb.shape[:2], dtype=np.int64)

        for color, class_id in self.target_colors_to_new.items():
            color = np.array(color, dtype=np.uint8)
            matches = np.all(mask_rgb == color, axis=-1)
            new_mask[matches] = class_id

        if self.normalize:
            image = image / 255.0

        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(new_mask).long()

        return image, mask


# ----------------------------
# Losses
# ----------------------------
def entropy_loss(logits):
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy.mean()


def consistency_loss(logits_1, logits_2):
    probs_1 = torch.softmax(logits_1, dim=1)
    probs_2 = torch.softmax(logits_2, dim=1)
    return F.mse_loss(probs_1, probs_2)


# ----------------------------
# Metrics helpers (Stage 2 — supervised, masks available)
# ----------------------------
def empty_stats(num_classes):
    fg = max(num_classes - 1, 1)
    return {
        "tp": torch.zeros(fg, dtype=torch.float64),
        "fp": torch.zeros(fg, dtype=torch.float64),
        "fn": torch.zeros(fg, dtype=torch.float64),
        "correct": 0,
        "pixels": 0,
        "ce": 0.0,
        "dice": 0.0,
    }


def update_stats(stats, logits, masks, l_ce, l_dice, num_classes):
    preds = logits.argmax(dim=1)
    tp, fp, fn, _ = smp.metrics.get_stats(
        preds.detach().cpu(),
        masks.detach().cpu(),
        mode="multiclass",
        num_classes=num_classes,
    )
    # Foreground classes only (drop background = class 0)
    stats["tp"] += tp[:, 1:].sum(dim=0).double()
    stats["fp"] += fp[:, 1:].sum(dim=0).double()
    stats["fn"] += fn[:, 1:].sum(dim=0).double()
    stats["correct"] += (preds == masks).sum().item()
    stats["pixels"] += masks.numel()
    stats["ce"] += l_ce.item()
    stats["dice"] += l_dice.item()


def summarize_stats(stats, num_batches):
    eps = 1e-7
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    union = tp + fp + fn
    valid = union > 0

    per_class_iou = torch.zeros_like(tp)
    per_class_iou[valid] = tp[valid] / (union[valid] + eps)

    per_class_precision = tp / (tp + fp + eps)
    per_class_recall = tp / (tp + fn + eps)
    per_class_f1 = (
        2 * per_class_precision * per_class_recall
        / (per_class_precision + per_class_recall + eps)
    )

    micro_iou = tp.sum() / (union.sum() + eps)
    macro_iou = per_class_iou[valid].mean() if valid.any() else torch.tensor(0.0)
    precision = tp.sum() / (tp.sum() + fp.sum() + eps)
    recall = tp.sum() / (tp.sum() + fn.sum() + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    pixel_accuracy = stats["correct"] / max(stats["pixels"], 1)

    return {
        "micro_iou": float(micro_iou),
        "macro_iou": float(macro_iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pixel_accuracy": float(pixel_accuracy),
        "per_class_iou": per_class_iou.numpy(),
        "per_class_precision": per_class_precision.numpy(),
        "per_class_recall": per_class_recall.numpy(),
        "per_class_f1": per_class_f1.numpy(),
        "ce_loss": stats["ce"] / max(num_batches, 1),
        "dice_loss": stats["dice"] / max(num_batches, 1),
    }


# ----------------------------
# TensorBoard logging helpers
# ----------------------------
def log_stage1_to_tb(writer, tag_prefix, scores, epoch):
    writer.add_scalar(f"{tag_prefix}/loss", scores["loss"], epoch)
    writer.add_scalar(f"{tag_prefix}/consistency_loss", scores["consistency_loss"], epoch)
    writer.add_scalar(f"{tag_prefix}/entropy_loss", scores["entropy_loss"], epoch)


def log_stage2_to_tb(writer, tag_prefix, scores, epoch):
    writer.add_scalar(f"{tag_prefix}/loss", scores["loss"], epoch)
    writer.add_scalar(f"{tag_prefix}/ce_loss", scores["ce_loss"], epoch)
    writer.add_scalar(f"{tag_prefix}/dice_loss", scores["dice_loss"], epoch)
    writer.add_scalar(f"{tag_prefix}/micro_iou", scores["micro_iou"], epoch)
    writer.add_scalar(f"{tag_prefix}/macro_iou", scores["macro_iou"], epoch)
    writer.add_scalar(f"{tag_prefix}/precision", scores["precision"], epoch)
    writer.add_scalar(f"{tag_prefix}/recall", scores["recall"], epoch)
    writer.add_scalar(f"{tag_prefix}/f1", scores["f1"], epoch)
    writer.add_scalar(f"{tag_prefix}/pixel_accuracy", scores["pixel_accuracy"], epoch)

    for i, (iou, prec, rec, f1) in enumerate(
        zip(
            scores["per_class_iou"],
            scores["per_class_precision"],
            scores["per_class_recall"],
            scores["per_class_f1"],
        ),
        start=1,
    ):
        writer.add_scalar(f"{tag_prefix}/per_class/iou/class_{i:02d}", iou, epoch)
        writer.add_scalar(f"{tag_prefix}/per_class/precision/class_{i:02d}", prec, epoch)
        writer.add_scalar(f"{tag_prefix}/per_class/recall/class_{i:02d}", rec, epoch)
        writer.add_scalar(f"{tag_prefix}/per_class/f1/class_{i:02d}", f1, epoch)


# ----------------------------
# Per-epoch test image prediction + figure
# ----------------------------
def predict_test_image(model, image_path, image_size, device):
    """Load a single image, run inference, return (input_tensor, pred_mask)."""
    image_tensor = load_single_image(image_path, image_size, device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu()  # (H, W)
    model.train(was_training)

    return image_tensor.squeeze(0).cpu(), pred


def make_prediction_figure(img_tensor, pred_mask):
    """Input image + coloured prediction side by side, using the floorplan palette."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_tensor.permute(1, 2, 0).numpy())
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(mask_to_rgb(pred_mask.numpy()))
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    plt.tight_layout()
    return fig


def log_test_image(writer, model, args, epoch, tag="test_image/prediction"):
    """Run the tracking image through the model and log the figure to TensorBoard."""
    track_path = args.track_image_path or args.test_image_path
    if not track_path:
        return
    track_path = Path(track_path)
    if not track_path.exists():
        print(f"  [warn] track image not found: {track_path} — skipping visual log")
        return

    img_tensor, pred_mask = predict_test_image(
        model, track_path, args.image_size, args.device
    )
    fig = make_prediction_figure(img_tensor, pred_mask)
    writer.add_figure(tag, fig, global_step=epoch)
    plt.close(fig)


# ----------------------------
# Loaders
# ----------------------------
def build_loaders(args, dataset_root):
    dataset = SegmentationDataset(
        root_dir=dataset_root,
        image_size=(args.image_size, args.image_size),
    )

    valid_size = max(1, int(len(dataset) * args.valid_fraction))
    train_size = len(dataset) - valid_size

    train_dataset, valid_dataset = random_split(
        dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    print(f"Dataset: {dataset_root}")
    print(f"Total: {len(dataset)} | Train: {len(train_dataset)} | Valid: {len(valid_dataset)}")

    return train_loader, valid_loader


def build_stage2_loaders(args):
    # Augmentation is applied to the TRAIN split ONLY. Validation sees clean data.
    augment = (
        DataAugmentation(
            image_size=(args.image_size, args.image_size),
            min_ops=1,
            max_ops=4,        # cap how many ops stack per sample (None = all 16)
            p=args.aug_prob,
        )
        if args.augment
        else None
    )

    # Two views over the same files: train augmented, valid clean.
    train_base = SyntheticFloorplanSegDataset(
        root_dir=args.dataset_b_root,
        image_size=(args.image_size, args.image_size),
        use_bw=args.use_bw,
        augment=augment,
        normalize=True,
    )
    valid_base = SyntheticFloorplanSegDataset(
        root_dir=args.dataset_b_root,
        image_size=(args.image_size, args.image_size),
        use_bw=args.use_bw,
        augment=None,
        normalize=True,
    )

    n = len(train_base)
    valid_size = max(1, int(n * args.valid_fraction))
    train_size = n - valid_size

    # Deterministic, non-overlapping index split shared by both views (no leakage).
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=generator).tolist()
    train_idx = perm[:train_size]
    valid_idx = perm[train_size:]

    train_dataset = Subset(train_base, train_idx)
    valid_dataset = Subset(valid_base, valid_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_stage2,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size_stage2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    print(f"Stage 2 Dataset: {args.dataset_b_root}")
    print(f"  use_bw={args.use_bw} | augment={'on' if augment is not None else 'off'}")
    print(f"  Total: {n} | Train: {len(train_dataset)} | Valid: {len(valid_dataset)}")

    return train_loader, valid_loader


# ----------------------------
# Model create/load
# ----------------------------
def create_model(args, encoder_weights=None):
    model = build_mitunet(
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
        in_channels=args.in_channels,
        classes=args.num_classes,
        decoder_attention_type=args.decoder_attention_type,
        checkpoint_path=None,
        device=args.device,
    )
    return model


def load_full_model(args, model_path):
    model = create_model(args, encoder_weights=None)

    print(f"Loading model from: {model_path}")
    state = torch.load(model_path, map_location=args.device)
    model.load_state_dict(state, strict=True)

    model.to(args.device)
    model.eval()

    return model


def load_stage1_model(args):
    if args.stage1_model_path:
        model_path = Path(args.stage1_model_path)
    else:
        model_path = Path(args.save_dir) / args.stage1_model_name

    return load_full_model(args, model_path)


def load_stage2_model(args):
    if args.stage2_model_path:
        model_path = Path(args.stage2_model_path)
    else:
        model_path = Path(args.save_dir) / args.stage2_model_name

    return load_full_model(args, model_path)


def load_encoder_a_into_model(args):
    model = create_model(args, encoder_weights=None)

    if args.encoder_a_path:
        encoder_path = Path(args.encoder_a_path)
    else:
        encoder_path = Path(args.save_dir) / args.encoder_a_name

    print(f"Loading Encoder A from: {encoder_path}")
    encoder_state = torch.load(encoder_path, map_location=args.device)
    model.encoder.load_state_dict(encoder_state)

    model.to(args.device)
    return model


# ----------------------------
# Stage 1: Dataset A
# random masks ignored
# ----------------------------
def train_stage_1_dataset_a(args):
    print("\n========== STAGE 1: Dataset A ==========")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    encoder_save_path = save_dir / args.encoder_a_name
    full_model_save_path = save_dir / args.stage1_model_name
    meta_path = save_dir / "stage1_meta.json"

    log_dir = Path(args.log_dir) / "stage1"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs (Stage 1) -> {log_dir}")
    print(f"  Run: tensorboard --logdir {Path(args.log_dir)}")

    train_loader, valid_loader = build_loaders(args, args.dataset_a_root)

    model = create_model(args, encoder_weights=args.encoder_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_stage1,
        weight_decay=args.weight_decay,
    )

    best_valid_loss = float("inf")

    for epoch in range(1, args.epochs_stage1 + 1):
        print(f"\nStage 1 Epoch {epoch}/{args.epochs_stage1}")

        train_scores = run_stage1_epoch(model, train_loader, optimizer, args)
        valid_scores = run_stage1_epoch(model, valid_loader, None, args)

        print(f"Train loss: {train_scores['loss']:.4f}")
        print(f"Valid loss: {valid_scores['loss']:.4f}")

        # ── TensorBoard logging ──────────────────────────────────────────────
        log_stage1_to_tb(writer, "train", train_scores, epoch)
        log_stage1_to_tb(writer, "valid", valid_scores, epoch)
        writer.add_scalars(
            "compare/loss",
            {"train": train_scores["loss"], "valid": valid_scores["loss"]},
            epoch,
        )

        # ── Per-epoch test image prediction ──────────────────────────────────
        log_test_image(writer, model, args, epoch, tag="stage1/test_image")

        if valid_scores["loss"] < best_valid_loss:
            best_valid_loss = valid_scores["loss"]

            torch.save(model.state_dict(), full_model_save_path)
            torch.save(model.encoder.state_dict(), encoder_save_path)

            metadata = {
                "stage": "stage_1",
                "dataset": str(args.dataset_a_root),
                "loss": "consistency + entropy",
                "random_masks_used": False,
                "best_valid_loss": best_valid_loss,
                "epoch": epoch,
            }

            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Saved Stage 1 full model -> {full_model_save_path}")
            print(f"Saved Encoder A -> {encoder_save_path}")

    writer.close()
    return encoder_save_path


def run_stage1_epoch(model, loader, optimizer, args):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_cons = 0.0
    total_ent = 0.0
    total_samples = 0
    num_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for images, _ in tqdm(loader, desc="Train A" if is_train else "Valid A"):
            images = images.to(args.device, non_blocking=True)

            images_aug = images + args.noise_std * torch.randn_like(images)
            images_aug = torch.clamp(images_aug, 0.0, 1.0)

            logits_1 = model(images)
            logits_2 = model(images_aug)

            loss_cons = consistency_loss(logits_1, logits_2)
            loss_ent = entropy_loss(logits_1)

            loss = (
                args.lambda_consistency * loss_cons +
                args.lambda_entropy * loss_ent
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_cons += loss_cons.item() * bs
            total_ent += loss_ent.item() * bs
            total_samples += bs
            num_batches += 1

    denom = max(total_samples, 1)
    return {
        "loss": total_loss / denom,
        "consistency_loss": total_cons / denom,
        "entropy_loss": total_ent / denom,
    }


# ----------------------------
# Stage 2: Dataset B
# proper masks used
# ----------------------------
def train_stage_2_dataset_b(args):
    print("\n========== STAGE 2: Dataset B ==========")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stage2_model_save_path = save_dir / args.stage2_model_name
    stage2_encoder_save_path = save_dir / args.encoder_finetuned_name
    meta_path = save_dir / "stage2_meta.json"

    log_dir = Path(args.log_dir) / "stage2"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs (Stage 2) -> {log_dir}")
    print(f"  Run: tensorboard --logdir {Path(args.log_dir)}")

    train_loader, valid_loader = build_stage2_loaders(args)

    model = load_encoder_a_into_model(args)

    # ── Stage 2 loss: weighted CE + Dice + Tversky ──────────────────────────
    class_weights = torch.tensor(args.class_weights, dtype=torch.float32, device=args.device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    criterion_tversky = smp.losses.TverskyLoss(
        mode="multiclass",
        from_logits=True,
        alpha=args.tversky_alpha,
        beta=args.tversky_beta,
    )
    print(f"Class weights: {class_weights.tolist()}")
    print(f"Tversky alpha/beta: {args.tversky_alpha}/{args.tversky_beta}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_stage2,
        weight_decay=args.weight_decay,
    )

    best_iou = -1.0

    for epoch in range(1, args.epochs_stage2 + 1):
        print(f"\nStage 2 Epoch {epoch}/{args.epochs_stage2}")

        train_scores = run_stage2_epoch(
            model, train_loader, optimizer,
            criterion_ce, criterion_dice, criterion_tversky, args
        )
        valid_scores = run_stage2_epoch(
            model, valid_loader, None,
            criterion_ce, criterion_dice, criterion_tversky, args
        )

        print(
            f"  Train -> loss: {train_scores['loss']:.4f}  "
            f"mIoU: {train_scores['micro_iou']:.4f}  F1: {train_scores['f1']:.4f}"
        )
        print(
            f"  Valid -> loss: {valid_scores['loss']:.4f}  "
            f"mIoU: {valid_scores['micro_iou']:.4f}  F1: {valid_scores['f1']:.4f}  "
            f"PixAcc: {valid_scores['pixel_accuracy']:.4f}"
        )

        # Per-class IoU (foreground only: wall, door, window)
        classes = ["wall", "door", "window"]
        print("  Per-class IoU -> " + "  ".join(
            f"{name}: {iou:.4f}"
            for name, iou in zip(classes, valid_scores["per_class_iou"])
        ))

        # ── TensorBoard logging ──────────────────────────────────────────────
        log_stage2_to_tb(writer, "train", train_scores, epoch)
        log_stage2_to_tb(writer, "valid", valid_scores, epoch)
        writer.add_scalars(
            "compare/micro_iou",
            {"train": train_scores["micro_iou"], "valid": valid_scores["micro_iou"]},
            epoch,
        )
        writer.add_scalars(
            "compare/macro_iou",
            {"train": train_scores["macro_iou"], "valid": valid_scores["macro_iou"]},
            epoch,
        )
        writer.add_scalars(
            "compare/loss",
            {"train": train_scores["loss"], "valid": valid_scores["loss"]},
            epoch,
        )
        writer.add_scalars(
            "compare/f1",
            {"train": train_scores["f1"], "valid": valid_scores["f1"]},
            epoch,
        )

        # ── Per-epoch test image prediction ──────────────────────────────────
        log_test_image(writer, model, args, epoch, tag="stage2/test_image")

        # ── Save best model (by validation micro IoU) ────────────────────────
        if valid_scores["micro_iou"] > best_iou:
            best_iou = valid_scores["micro_iou"]

            torch.save(model.state_dict(), stage2_model_save_path)
            torch.save(model.encoder.state_dict(), stage2_encoder_save_path)

            metadata = {
                "stage": "stage_2",
                "dataset": str(args.dataset_b_root),
                "loss": "weighted_cross_entropy + dice + tversky",
                "class_weights": args.class_weights,
                "tversky_alpha": args.tversky_alpha,
                "tversky_beta": args.tversky_beta,
                "best_valid_micro_iou": best_iou,
                "valid_macro_iou": valid_scores["macro_iou"],
                "valid_f1": valid_scores["f1"],
                "valid_pixel_accuracy": valid_scores["pixel_accuracy"],
                "epoch": epoch,
            }

            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"  ✓ Saved Stage 2 model (mIoU={best_iou:.4f}) -> {stage2_model_save_path}")
            print(f"  ✓ Saved fine-tuned encoder -> {stage2_encoder_save_path}")

    writer.close()
    print(f"\nStage 2 complete. Best valid mIoU: {best_iou:.4f}")


def run_stage2_epoch(model, loader, optimizer, criterion_ce, criterion_dice, criterion_tversky, args):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    num_batches = 0
    stats = empty_stats(args.num_classes)

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        pbar = tqdm(loader, desc="Train B" if is_train else "Valid B")
        for images, masks in pbar:
            images = images.to(args.device, non_blocking=True)
            masks = masks.to(args.device, non_blocking=True).long()

            logits = model(images)

            ce = criterion_ce(logits, masks)
            dice = criterion_dice(logits, masks)
            tversky = criterion_tversky(logits, masks)
            loss = ce + dice + tversky

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            num_batches += 1

            update_stats(stats, logits, masks, ce, dice, args.num_classes)
            scores = summarize_stats(stats, num_batches)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", miou=f"{scores['micro_iou']:.4f}"
            )

    scores = summarize_stats(stats, num_batches)
    scores["loss"] = total_loss / max(total_samples, 1)
    return scores


# ----------------------------
# Single Image Prediction
# ----------------------------
def predict_single_image(model, image_path, args, save_path):
    image = load_single_image(
        image_path=image_path,
        image_size=args.image_size,
        device=args.device,
    )

    with torch.no_grad():
        logits = model(image)
        pred = torch.argmax(logits, dim=1)

    pred_mask = pred[0].detach().cpu().numpy().astype(np.uint8)
    save_mask_png(pred_mask, save_path)

    print(f"Saved mask -> {save_path}")


def testing_stage1_model_output(args):
    print("\n========== TEST STAGE 1 MODEL OUTPUT ==========")

    output_dir = Path(args.test_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_stage1_model(args)

    image_name = Path(args.test_image_path).stem
    save_path = output_dir / f"{image_name}_stage1_mask.png"

    predict_single_image(model, args.test_image_path, args, save_path)


def testing_stage2_model_output(args):
    print("\n========== TEST STAGE 2 MODEL OUTPUT ==========")

    output_dir = Path(args.test_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_stage2_model(args)

    image_name = Path(args.test_image_path).stem
    save_path = output_dir / f"{image_name}_stage2_mask.png"

    predict_single_image(model, args.test_image_path, args, save_path)


def testing_both_outputs(args):
    testing_stage1_model_output(args)
    testing_stage2_model_output(args)

def add_version_suffix(filename, version):
    """Insert a version tag before the extension: model.pth -> model_v1.pth."""
    if not version:
        return filename
    version = str(version)
    if version.isdigit():          # allow '1' -> 'v1'
        version = f"v{version}"
    p = Path(filename)
    return f"{p.stem}_{version}{p.suffix}"

# ----------------------------
# Argparse
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_mode",
        type=str,
        default="both",
        choices=[
            "stage_1",
            "stage_2",
            "both",
            "test_stage1",
            "test_stage2",
            "test_both",
        ],
    )

    parser.add_argument("--project_root", type=str, default="/home/ubuntu/mitunet")

    parser.add_argument(
        "--dataset_a_root",
        type=str,
        default="/home/ubuntu/mitunet/dataset_A_random_masks",
    )

    parser.add_argument(
        "--dataset_b_root",
        type=str,
        default="/home/ubuntu/mitunet/synthetic_dataset/plans",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/ubuntu/mitunet/checkpoints/encoder_a_to_b",
    )

    parser.add_argument(
        "--test_output_dir",
        type=str,
        default="/home/ubuntu/mitunet/test_outputs",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/ubuntu/mitunet/runs/encoder_a_to_b",
    )

    parser.add_argument("--track_image_path", type=str, default="")
    parser.add_argument("--test_image_path", type=str, default="")

    parser.add_argument("--encoder_a_path", type=str, default="")
    parser.add_argument("--stage1_model_path", type=str, default="")
    parser.add_argument("--stage2_model_path", type=str, default="")

    parser.add_argument("--encoder_a_name", type=str, default="encoder_a_mit_b4.pth")
    parser.add_argument("--encoder_finetuned_name", type=str, default="encoder_a_finetuned_on_b.pth")
    parser.add_argument("--stage1_model_name", type=str, default="model_stage1_dataset_a.pth")
    parser.add_argument("--stage2_model_name", type=str, default="model_stage2_dataset_b.pth")

    parser.add_argument("--encoder_name", type=str, default="mit_b4")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")
    parser.add_argument("--decoder_attention_type", type=str, default="scse")

    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512) # changed from 256 -> 512

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs_stage1", type=int, default=10)
    parser.add_argument("--epochs_stage2", type=int, default=20)

    parser.add_argument("--use_bw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aug_prob", type=float, default=0.9)
    parser.add_argument("--batch_size_stage2", type=int, default=4)

    parser.add_argument("--lr_stage1", type=float, default=5e-5)
    parser.add_argument("--lr_stage2", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument(
        "--model_version",
        type=str,
        default="",
        help="Optional version tag appended to saved model/encoder filenames "
            "(e.g. '1' or 'v1' -> ..._v1.pth). Empty = no suffix.",
    )

    # ── Stage 2 loss: weighted CE + Dice + Tversky ──────────────────────────
    parser.add_argument(
        "--class_weights",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 2.0],   # background, wall, door, window
    )
    parser.add_argument("--tversky_alpha", type=float, default=0.3)  # false-positive penalty
    parser.add_argument("--tversky_beta",  type=float, default=0.7)  # false-negative penalty

    parser.add_argument("--valid_fraction", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lambda_consistency", type=float, default=1.0)
    parser.add_argument("--lambda_entropy", type=float, default=0.01)
    parser.add_argument("--noise_std", type=float, default=0.05)

    args = parser.parse_args()

    args.project_root = Path(args.project_root)
    args.dataset_a_root = Path(args.dataset_a_root)
    args.dataset_b_root = Path(args.dataset_b_root)
    args.save_dir = Path(args.save_dir)
    args.test_output_dir = Path(args.test_output_dir)
    args.log_dir = Path(args.log_dir)

    # Apply version tag to all saved/loaded artifact filenames
    # args.encoder_a_name        = add_version_suffix(args.encoder_a_name, args.model_version)
    # args.encoder_finetuned_name = add_version_suffix(args.encoder_finetuned_name, args.model_version)
    # args.stage1_model_name     = add_version_suffix(args.stage1_model_name, args.model_version)
    args.stage2_model_name     = add_version_suffix(args.stage2_model_name, args.model_version)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if len(args.class_weights) != args.num_classes:
        raise ValueError(
            f"--class_weights has {len(args.class_weights)} values "
            f"but num_classes is {args.num_classes}"
        )

    if args.run_mode in ["test_stage1", "test_stage2", "test_both"]:
        if args.test_image_path == "":
            raise ValueError("Please provide --test_image_path for testing.")

    return args


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    seed_everything(args.seed)

    print(f"Device: {args.device}")
    print(f"Run mode: {args.run_mode}")

    if args.run_mode == "stage_1":
        train_stage_1_dataset_a(args)

    elif args.run_mode == "stage_2":
        train_stage_2_dataset_b(args)

    elif args.run_mode == "both":
        encoder_path = train_stage_1_dataset_a(args)
        args.encoder_a_path = str(encoder_path)
        train_stage_2_dataset_b(args)

    elif args.run_mode == "test_stage1":
        testing_stage1_model_output(args)

    elif args.run_mode == "test_stage2":
        testing_stage2_model_output(args)

    elif args.run_mode == "test_both":
        testing_both_outputs(args)


if __name__ == "__main__":
    main()


###
"""
python src/train_enc_dec.py --run_mode both \
  --track_image_path /home/ubuntu/mitunet/test_images/image_0.png

python src/train_enc_dec.py --run_mode stage_1 \
  --track_image_path /home/ubuntu/mitunet/test_images/image_0.png

python src/train_enc_dec.py --run_mode stage_2 \
  --track_image_path /home/ubuntu/mitunet/test_images/image_0.png

python src/train_enc_dec.py --run_mode test_stage1 \
  --test_image_path /home/ubuntu/mitunet/test_images/image_0.png

python src/train_enc_dec.py --run_mode test_stage2 \
  --test_image_path /home/ubuntu/mitunet/test_images/image_0.png

tensorboard --logdir /home/ubuntu/mitunet/runs/encoder_a_to_b

python src/train_enc_dec.py --run_mode both \
  --dataset_a_root /home/ubuntu/mitunet/dataset_A_random_masks \
  --dataset_b_root /home/ubuntu/mitunet/synthetic_dataset/plans \
  --save_dir /home/ubuntu/mitunet/checkpoints/encoder_a_to_b \
  --log_dir /home/ubuntu/mitunet/runs/encoder_a_to_b \
  --track_image_path /home/ubuntu/mitunet/test_images/image_0.png \
  --epochs_stage1 10 --epochs_stage2 20 \
  --batch_size 2 --batch_size_stage2 8 \
  --lr_stage1 5e-5 --lr_stage2 1e-4 \
  --image_size 256 --num_classes 4 \
  --use_bw

python src/train_enc_dec.py --run_mode stage_2 --model_version 1 --test_image_path /home/ubuntu/mitunet/test_images/image_0.png --log_dir /home/ubuntu/mitunet/runs/encoder_a_to_b_1

"""