import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, "/home/ubuntu/mitunet")

from src.model.model import build_mitunet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model with weights
model = build_mitunet(
    encoder_name="mit_b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    decoder_attention_type="scse",
    checkpoint_path="/home/ubuntu/mitunet/mitunet.pth",
    device=DEVICE,
)

print(f"Model loaded on device: {DEVICE}\n")

# Print torchsummary
print("=" * 80)
print("MitUNet Model Summary")
print("=" * 80)

# ============================================================================
# PREDICTION ON TEST IMAGE
# ============================================================================
print("\n" + "=" * 80)
print("Running Prediction on Test Image")
print("=" * 80)

# Define inference transforms
inference_transforms = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Load test image
test_image_path = "/home/ubuntu/mitunet/test_images/image.png"
print(f"Loading test image from: {test_image_path}")

def predict(test_image_path):
    image_bgr = cv2.imread(test_image_path)
    if image_bgr is None:
        print(f"ERROR: Could not load image from {test_image_path}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = image_rgb.shape[:2]
    print(f"Original image size: {original_w}x{original_h}")

    # Apply transforms
    augmented = inference_transforms(image=image_rgb)
    image_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

    # Run inference
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).float()

    pred_mask_np = pred_mask.cpu().squeeze().numpy()
    print(f"Prediction shape: {pred_mask_np.shape}")
    print(f"Prediction range: [{pred_mask_np.min():.4f}, {pred_mask_np.max():.4f}]")

    # Save outputs
    output_dir = "/home/ubuntu/mitunet/outputs"
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save prediction mask
    pred_mask_uint8 = (pred_mask_np * 255).astype(np.uint8)
    output_mask_path = os.path.join(output_dir, "prediction_mask.png")
    cv2.imwrite(output_mask_path, pred_mask_uint8)
    print(f"✓ Prediction mask saved to: {output_mask_path}")

    # Save visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(pred_mask_np, cmap='gray')
    axes[1].set_title("MitUNet Prediction Mask")
    axes[1].axis('off')

    plt.tight_layout()
    output_viz_path = os.path.join(output_dir, "prediction_visualization.png")
    plt.savefig(output_viz_path, dpi=100, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_viz_path}")
    plt.close()

    print("\n" + "=" * 80)
    print("Prediction Complete! ✓")
    print("=" * 80)

predict(test_image_path)