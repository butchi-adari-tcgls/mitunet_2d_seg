# test_floorplan_segmentation.py

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from PIL import Image
from model.model import build_mitunet


# ----------------------------
# Color Map
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
def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in COLOR_MAP.items():
        rgb[mask == cls] = color

    return rgb


def save_mask_png(mask, save_path):
    rgb = mask_to_rgb(mask)
    Image.fromarray(rgb).save(save_path)

# def save_mask_png(mask, save_path):
#     # Saves raw class IDs as pixel values: 0, 1, 2, 3
#     Image.fromarray(mask.astype(np.uint8), mode="L").save(save_path)

def load_single_image(image_path, image_size, device):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))

    image_np = np.array(image).astype(np.float32) / 255.0

    image_tensor = torch.tensor(image_np)
    image_tensor = image_tensor.permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    return image_tensor


# ----------------------------
# Model
# ----------------------------
def create_model(args):
    model = build_mitunet(
        encoder_name=args.encoder_name,
        encoder_weights=None,
        in_channels=args.in_channels,
        classes=args.num_classes,
        decoder_attention_type=args.decoder_attention_type,
        checkpoint_path=None,
        device=args.device,
    )
    return model


def load_model(args):
    model = create_model(args)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")

    state = torch.load(model_path, map_location=args.device)
    model.load_state_dict(state, strict=True)

    model.to(args.device)
    model.eval()

    return model


# ----------------------------
# Prediction
# ----------------------------
# def predict_single_image(model, image_path, args):
#     image_path = Path(image_path)

#     image = load_single_image(
#         image_path=image_path,
#         image_size=args.image_size,
#         device=args.device,
#     )

#     with torch.no_grad():
#         logits = model(image)
#         pred = torch.argmax(logits, dim=1)

#     pred_mask = pred[0].detach().cpu().numpy().astype(np.uint8)

#     output_dir = Path(args.test_output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     save_path = output_dir / f"{image_path.stem}_mask.png"
#     save_mask_png(pred_mask, save_path)

#     print(f"Saved mask -> {save_path}")

def predict_single_image(model, image_path, args):
    image_path = Path(image_path)

    image = load_single_image(
        image_path=image_path,
        image_size=args.image_size,
        device=args.device,
    )

    with torch.no_grad():
        logits = model(image)
        pred = torch.argmax(logits, dim=1)

    # 2D label mask: values are class IDs only: 0, 1, 2, 3
    pred_mask = pred[0].detach().cpu().numpy().astype(np.uint8)

    output_dir = Path(args.test_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as .npy label mask
    npy_save_path = output_dir / f"{image_path.stem}_mask.npy"
    np.save(npy_save_path, pred_mask)

    # Save as .png label mask
    # This is NOT RGB. Pixel values are still 0, 1, 2, 3.
    png_save_path = output_dir / f"{image_path.stem}_mask.png"
    # Image.fromarray(pred_mask, mode="L").save(png_save_path)
    save_mask_png(pred_mask, png_save_path)

    print(f"Saved label mask npy -> {npy_save_path}")
    print(f"Saved label mask png -> {png_save_path}")

def get_images_from_folder(folder_path):
    folder_path = Path(folder_path)

    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]

    image_paths = []
    for ext in image_extensions:
        image_paths.extend(folder_path.glob(ext))

    return sorted(image_paths)


def run_testing(args):
    model = load_model(args)

    if args.test_image_path:
        predict_single_image(model, args.test_image_path, args)

    elif args.test_folder_path:
        image_paths = get_images_from_folder(args.test_folder_path)

        if len(image_paths) == 0:
            raise RuntimeError(f"No images found in folder: {args.test_folder_path}")

        print(f"Found {len(image_paths)} images")

        for image_path in image_paths:
            predict_single_image(model, image_path, args)

    else:
        raise ValueError("Provide either --test_image_path or --test_folder_path")


# ----------------------------
# Args
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model .pth file",
    )

    parser.add_argument(
        "--test_image_path",
        type=str,
        default="",
        help="Path to a single test image",
    )

    parser.add_argument(
        "--test_folder_path",
        type=str,
        default="",
        help="Path to folder containing test images",
    )

    parser.add_argument(
        "--test_output_dir",
        type=str,
        default="/home/ubuntu/mitunet/test_outputs",
        help="Folder to save predicted masks",
    )

    parser.add_argument("--encoder_name", type=str, default="mit_b4")
    parser.add_argument("--decoder_attention_type", type=str, default="scse")

    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512)

    args = parser.parse_args()

    if args.test_image_path and args.test_folder_path:
        raise ValueError("Use only one: --test_image_path or --test_folder_path")

    if not args.test_image_path and not args.test_folder_path:
        raise ValueError("Provide --test_image_path or --test_folder_path")

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    print(f"Device: {args.device}")
    print(f"Output dir: {args.test_output_dir}")

    run_testing(args)


if __name__ == "__main__":
    main()


"""
python src/test_enc_dec.py \
  --model_path /home/ubuntu/mitunet/checkpoints/encoder_a_to_b/model_stage2_dataset_b_v1.pth \
  --test_image_path /home/ubuntu/mitunet/test_images/image_0.png \

python src/test_enc_dec.py \
  --model_path /home/ubuntu/mitunet/checkpoints/encoder_a_to_b/model_stage2_dataset_b_v1.pth \
  --test_folder_path /home/ubuntu/mitunet/test_images/CPMS_ClusterPlans \
  --test_output_dir /home/ubuntu/mitunet/test_outputs/CPMS_ClusterPlans_masks/
  
python src/test_enc_dec.py   --model_path /home/ubuntu/mitunet/checkpoints/encoder_a_to_b/model_stage2_dataset_b_v1.pth   --test_folder_path /home/ubuntu/mitunet/test_images/FloorPlans/   --test_output_dir /home/ubuntu/mitunet/test_outputs/FloorPlans_masks/

"""