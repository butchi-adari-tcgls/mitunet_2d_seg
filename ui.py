import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import gradio as gr
from matplotlib import pyplot as plt
import io

# --- make your project modules importable (adjust if your layout differs) ---
PROJECT_ROOT = Path("/home/ubuntu/mitunet")
sys.path.append(str(PROJECT_ROOT))
from src.model.model import build_mitunet
# with this
from src.build_graph import (
    load_color_mask_exact,
    mask_to_graph,
    clean_graph,
    mask_to_rgb as graph_mask_to_rgb,
    plot_graph,
)

# ---------------------------------------------------------------------------
# Inference config — mirror your Stage 2 testing args
# ---------------------------------------------------------------------------
DEVICE                 = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE             = 512                 # matches --image_size default
NUM_CLASSES            = 4
IN_CHANNELS            = 3
ENCODER_NAME           = "mit_b4"
DECODER_ATTENTION_TYPE = "scse"
CHECKPOINT_PATH        = PROJECT_ROOT / "checkpoints/encoder_a_to_b/model_stage2_dataset_b_v1.pth"

# same palette as training (COLOR_MAP)
COLOR_MAP = {
    0: (200, 200, 200),  # background
    1: (40, 40, 40),     # wall
    2: (160, 82, 45),    # door
    3: (135, 206, 250),  # window
}


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        rgb[mask == cls] = color
    return rgb


def _load_stage2_model():
    model = build_mitunet(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,           # weights come from the checkpoint
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        decoder_attention_type=DECODER_ATTENTION_TYPE,
        checkpoint_path=None,
        device=DEVICE,
    )
    print(f"[seg] loading Stage 2 model -> {CHECKPOINT_PATH}")
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


# load ONCE at startup, not per-upload
MODEL = _load_stage2_model()


@torch.no_grad()
def run_segmentation(image: Image.Image) -> Image.Image:
    # preprocess exactly like load_single_image()
    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # (1,3,H,W)

    # predict like predict_single_image()
    logits = MODEL(x)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # (H,W)

    # colorize for display
    return Image.fromarray(mask_to_rgb(pred))


def clean_mask(image: Image.Image, seg_mask: Image.Image):
    rgb = np.array(seg_mask.convert("RGB"))
    mask_np = np.zeros(rgb.shape[:2], np.int32)

    for cls, color in COLOR_MAP.items():
        mask_np[np.all(rgb == np.array(color, np.uint8), axis=-1)] = cls

    graph = mask_to_graph(
        mask_np,
        room_min_area=120,
        match_tolerance=8,
        contour_mode="poly",
        seal_ksize="auto",
        verbose=False,
    )

    cleaned_np = graph["cleaned_mask"]
    return Image.fromarray(graph_mask_to_rgb(cleaned_np))
    
# replace your build_graph()
def build_graph(image: Image.Image, cleaned_mask: Image.Image):
    rgb = np.array(cleaned_mask.convert("RGB"))
    mask_np = np.zeros(rgb.shape[:2], np.int32)

    for cls, color in COLOR_MAP.items():
        mask_np[np.all(rgb == np.array(color, np.uint8), axis=-1)] = cls

    graph = mask_to_graph(
        mask_np,
        room_min_area=120,
        match_tolerance=8,
        contour_mode="poly",
        seal_ksize="auto",
        verbose=False,
    )
    graph = clean_graph(graph, keep_outside=True)

    buf = io.BytesIO()
    fig = plot_graph(
        graph,
        image_size=mask_np.shape,
        show_mask=True,
        save_path=None,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    graph_image = Image.open(buf).convert("RGB")
    return graph_image, graph


def run_pipeline(image: Image.Image):
    if image is None:
        yield None, None, None, None
        return

    # 0) show original immediately
    yield image, None, None, None

    # 1) segmentation
    seg_mask = run_segmentation(image)
    yield image, seg_mask, None, None

    # 2) cleaned mask
    cleaned = clean_mask(image, seg_mask)
    yield image, seg_mask, cleaned, None

    # 3) graph
    graph_image, _graph = build_graph(image, cleaned)
    yield image, seg_mask, cleaned, graph_image


# ===========================================================================
# UI
# ===========================================================================
CSS = """
.panel-img img { object-fit: contain !important; }
#title { text-align: center; margin-bottom: 4px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Mask & Graph Pipeline") as demo:
    gr.Markdown("# 🧩 Segmentation → Cleaning → Graph Pipeline", elem_id="title")
    gr.Markdown(
        "Upload an image — the pipeline runs automatically and fills the four "
        "panels below as each stage completes."
    )

    with gr.Row():
        input_image = gr.Image(
            label="Upload image",
            type="pil",
            sources=["upload", "clipboard"],
            height=300,
        )

    with gr.Row():
        original_out = gr.Image(
            label="Original", type="pil", height=280, elem_classes="panel-img"
        )
        seg_out = gr.Image(
            label="Segmentation Mask", type="pil", height=280, elem_classes="panel-img"
        )
    with gr.Row():
        cleaned_out = gr.Image(
            label="Cleaned Mask", type="pil", height=280, elem_classes="panel-img"
        )
        graph_out = gr.Image(
            label="Graph", type="pil", height=280, elem_classes="panel-img"
        )

    outputs = [original_out, seg_out, cleaned_out, graph_out]

    # Auto-trigger on upload (and on paste/change).
    input_image.upload(run_pipeline, inputs=input_image, outputs=outputs)
    input_image.change(run_pipeline, inputs=input_image, outputs=outputs)


if __name__ == "__main__":
    demo.launch()