"""
graph_maker.py

Floor-plan -> room adjacency graph, built step by step so every
intermediate stage can be inspected and tuned.

Pipeline (each step has its own `show` plot):
    a) predict_mask           — run the segmentation model
    b) add_border             — seal the image with a constant border
    c) morph_close            — close gaps (doors/windows)
    d) invert                 — flip so rooms become white blobs
    e) connected_components   — label each blob
    f) filter_rooms           — drop noise + outside region by area
    g) detect_nodes           — polygonise the kept rooms
    h) detect_edges           — find pairs of adjacent rooms via
                                       parallel + close + overlapping polygon segments
    i) segment_nodes_edges    — clean final viz: rooms + edges, no bboxes

Run from the CLI:
    python graph_maker.py --img-path /path/to/floorplan.png --show
    python graph_maker.py --img-path ... --kernel-size 21 --min-room-area 800

Compare configurations in a notebook (model is loaded only once):
    gm = GraphMaker(weights_path=...)
    g1 = gm.build_graph(image, kernel_size=15, min_room_area=500, show=True)
    g2 = gm.build_graph(image, kernel_size=21, min_room_area=800, show=True)
"""

import argparse
import json
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


DEFAULT_WEIGHTS = "mitunet.pth"

def segment_direction(p1: np.ndarray, p2: np.ndarray) -> np.ndarray | None:
    """Unit vector from p1 -> p2, or None if the segment has zero length."""
    v = p2 - p1
    n = np.linalg.norm(v)
    if n == 0:
        return None
    return v / n


def projection_overlap(
    a1: np.ndarray, a2: np.ndarray,
    b1: np.ndarray, b2: np.ndarray,
    axis: np.ndarray,
) -> float:
    """Length of the overlap between segments [a1, a2] and [b1, b2] when
    projected onto `axis`."""
    a_proj = sorted([np.dot(a1, axis), np.dot(a2, axis)])
    b_proj = sorted([np.dot(b1, axis), np.dot(b2, axis)])
    overlap = min(a_proj[1], b_proj[1]) - max(a_proj[0], b_proj[0])
    return max(0.0, float(overlap))


def point_to_line_distance(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> float:
    """Perpendicular distance from point p to the infinite line through a, b."""
    ab = b - a
    n = np.linalg.norm(ab)
    if n == 0:
        return float(np.linalg.norm(p - a))
    return float(abs(np.cross(ab, p - a)) / n)


def segments_match(
    a1: np.ndarray, a2: np.ndarray,
    b1: np.ndarray, b2: np.ndarray,
    gap_thresh: float = 20.0,
    min_overlap: float = 10.0,
    angle_thresh: float = 0.95,
) -> bool:
    """
    Two segments are considered "shared wall" if:
      - they are almost parallel             (|cos angle| >= angle_thresh)
      - the perpendicular gap between them is small  (<= gap_thresh)
      - their projection along A's direction overlaps enough (>= min_overlap)
    """
    dir_a = segment_direction(a1, a2)
    dir_b = segment_direction(b1, b2)
    if dir_a is None or dir_b is None:
        return False

    if abs(np.dot(dir_a, dir_b)) < angle_thresh:
        return False

    # d1 = point_to_line_distance(b1, a1, a2)
    # d2 = point_to_line_distance(b2, a1, a2)
    # dist = (d1 + d2) / 2

    d1 = point_to_line_distance(b1, a1, a2)
    d2 = point_to_line_distance(b2, a1, a2)
    d3 = point_to_line_distance(a1, b1, b2)
    d4 = point_to_line_distance(a2, b1, b2)
    dist = min((d1 + d2) / 2, (d3 + d4) / 2)
    if dist > gap_thresh:
        return False

    overlap = projection_overlap(a1, a2, b1, b2, dir_a)
    return overlap >= min_overlap


# ====================================================================== #
# GraphMaker
# ====================================================================== #
class GraphMaker:
    """
    Build a room adjacency graph from a floor-plan image.

    The constructor only loads the segmentation model. All tuning knobs
    are passed to `build_graph` (or to the individual step methods),
    so you can compare configurations on the same instance:

        gm = GraphMaker(weights_path=...)
        g1 = gm.build_graph(image, kernel_size=15, min_room_area=500)
        g2 = gm.build_graph(image, kernel_size=21, min_room_area=800)

    Hyperparameters (all passed per-call)
    -------------------------------------
    border               : pixels of constant border added in step 2.
    kernel_size          : structuring element size for closing (step 3).
    iterations           : closing iterations (step 3).
    min_room_area        : minimum kept component area in pixels (step 6).
    max_room_area        : max kept component area; None -> 50% of padded image.
    poly_epsilon_ratio   : approxPolyDP epsilon as fraction of perimeter (step 7).
    gap_thresh           : max perpendicular distance between two polygon
                            segments to call them a shared wall (step 8, px).
    min_overlap          : min projected overlap between two segments to call
                            them a shared wall (step 8, px).
    angle_thresh         : |cos(angle)| threshold for two segments to count
                            as parallel (step 8). 0.95 ≈ within ~18°.
    """

    def __init__(
        self,
        weights_path: str,
        device: str | None = None,
        input_size: int = 512,
    ):
        self.weights_path = weights_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        self.model = self._load_model()
        self.transform = A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def _load_model(self) -> torch.nn.Module:
        aux_segformer = smp.Segformer(
            encoder_name="mit_b4", encoder_weights=None
        )
        model = smp.Unet(
            encoder_name="mit_b4",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            decoder_attention_type="scse",
        )
        # Transplant the SegFormer encoder onto the Unet
        model.encoder = aux_segformer.encoder

        state = torch.load(self.weights_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _new_fig(title: str, figsize=(6, 6)):
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=12)
        return fig

    @staticmethod
    def _random_colors(n: int, seed: int = 42) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, 3), dtype=np.uint8)
        rng = np.random.default_rng(seed)
        return rng.integers(60, 255, size=(n, 3), dtype=np.uint8)

    @torch.no_grad()
    def predict_mask(
        self, image_bgr: np.ndarray, show: bool = False
    ) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image passed to step1_predict_mask")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image_rgb)
        input_tensor = augmented["image"].unsqueeze(0).to(self.device)

        logits = self.model(input_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float().squeeze().cpu().numpy()

        if show:
            display_rgb = cv2.resize(image_rgb, (self.input_size, self.input_size))
            self._new_fig("Step 1 — Predict wall mask", figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(display_rgb); plt.title("Input image"); plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap="gray"); plt.title("Predicted mask"); plt.axis("off")
            plt.tight_layout()

        return mask

    def add_border(
        self,
        wall_mask: np.ndarray,
        border: int = 3,
        show: bool = False,
    ) -> np.ndarray:
        wall_img = (wall_mask > 0).astype(np.uint8) * 255
        bordered = cv2.copyMakeBorder(
            wall_img, border, border, border, border,
            cv2.BORDER_CONSTANT, value=255,
        )

        if show:
            self._new_fig(
                f"Step 2 — Add border (border={border})", figsize=(10, 5)
            )
            plt.subplot(1, 2, 1)
            plt.imshow(wall_img, cmap="gray"); plt.title("Wall mask"); plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(bordered, cmap="gray"); plt.title("Bordered"); plt.axis("off")
            plt.tight_layout()

        return bordered

    def morph_close(
        self,
        bordered: np.ndarray,
        kernel_size: int = 15,
        iterations: int = 2,
        show: bool = False,
    ) -> np.ndarray:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        closed = cv2.morphologyEx(
            bordered, cv2.MORPH_CLOSE, kernel, iterations=iterations
        )

        if show:
            self._new_fig(
                f"Step 3 — Morph close "
                f"(kernel_size={kernel_size}, iterations={iterations})",
                figsize=(10, 5),
            )
            plt.subplot(1, 2, 1)
            plt.imshow(bordered, cmap="gray"); plt.title("Before"); plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(closed, cmap="gray"); plt.title("After closing"); plt.axis("off")
            plt.tight_layout()

        return closed

    def invert(
        self, closed: np.ndarray, show: bool = False
    ) -> np.ndarray:
        inverted = cv2.bitwise_not(closed)

        if show:
            self._new_fig(
                "Step 4 — Invert (rooms become white blobs)", figsize=(10, 5)
            )
            plt.subplot(1, 2, 1)
            plt.imshow(closed, cmap="gray"); plt.title("Closed"); plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(inverted, cmap="gray"); plt.title("Inverted"); plt.axis("off")
            plt.tight_layout()

        return inverted

    def connected_components(
        self, inverted: np.ndarray, show: bool = False
    ):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            inverted, connectivity=4
        )

        if show:
            colors = np.zeros((num_labels, 3), dtype=np.uint8)
            colors[1:] = self._random_colors(num_labels - 1)
            cc_vis = colors[labels]  # fancy-indexing -> HxWx3
            self._new_fig(
                f"Step 5 — Connected components ({num_labels - 1} blobs)",
                figsize=(6, 6),
            )
            plt.imshow(cc_vis); plt.axis("off")
            plt.tight_layout()

        return num_labels, labels, stats, centroids

    def filter_rooms(
        self,
        num_labels: int,
        labels: np.ndarray,
        stats: np.ndarray,
        centroids: np.ndarray,
        border: int = 3,
        min_room_area: int = 500,
        max_room_area: int | None = None,
        show: bool = False,
    ) -> list[dict]:
        H, W = labels.shape
        max_area = (
            max_room_area if max_room_area is not None else int(H * W * 0.5)
        )

        # Any component touching the outer padded band is "outside"
        edge_band = border + 2
        edge_labels = set()
        edge_labels.update(labels[:edge_band, :].flatten())
        edge_labels.update(labels[-edge_band:, :].flatten())
        edge_labels.update(labels[:, :edge_band].flatten())
        edge_labels.update(labels[:, -edge_band:].flatten())
        edge_labels.discard(0)  # background

        rooms = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]

            if not (min_room_area <= area <= max_area):
                continue
            if i in edge_labels:
                continue

            rooms.append({
                "id": len(rooms) + 1,
                "label": int(i),
                "bbox": (int(x), int(y), int(w), int(h)),
                "centroid": (int(cx), int(cy)),
                "area": int(area),
            })

        if show:
            vis = np.zeros((H, W, 3), dtype=np.uint8)
            colors = self._random_colors(len(rooms))
            for r, c in zip(rooms, colors):
                vis[labels == r["label"]] = c
            self._new_fig(
                f"Step 6 — Filter rooms "
                f"(min_area={min_room_area}, max_area={max_area}; "
                f"{len(rooms)} kept)",
                figsize=(6, 6),
            )
            plt.imshow(vis)
            for r in rooms:
                cx, cy = r["centroid"]
                plt.text(
                    cx, cy, f"R{r['id']}",
                    color="white", ha="center", va="center", fontsize=9,
                    bbox=dict(facecolor="black", alpha=0.5, pad=2),
                )
            plt.axis("off")
            plt.tight_layout()

        return rooms

    def detect_nodes(
        self,
        rooms: list[dict],
        labels: np.ndarray,
        poly_epsilon_ratio: float = 0.003,
        show: bool = False,
    ) -> list[dict]:
        for r in rooms:
            comp_mask = (labels == r["label"]).astype(np.uint8) * 255
            # CHAIN_APPROX_NONE keeps every boundary pixel; gives approxPolyDP
            # a denser input to work with -> cleaner polygons.
            contours, _ = cv2.findContours(
                comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            dense_contour = contours[0] if contours else None

            polygon = None
            if dense_contour is not None:
                eps = poly_epsilon_ratio * cv2.arcLength(dense_contour, True)
                approx = cv2.approxPolyDP(dense_contour, eps, True)
                polygon = approx.reshape(-1, 2).tolist()

            r["polygon"] = polygon
            r["_contour"] = dense_contour  # internal; not part of final output

        if show:
            H, W = labels.shape
            vis = np.zeros((H, W, 3), dtype=np.uint8)
            for r in rooms:
                if r["polygon"]:
                    pts = np.array(r["polygon"], dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(
                        vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2
                    )
                cv2.circle(vis, r["centroid"], 6, (0, 0, 255), -1)
            self._new_fig(
                f"Step 7 — Detect nodes "
                f"(poly_epsilon_ratio={poly_epsilon_ratio}; "
                f"{len(rooms)} nodes)",
                figsize=(6, 6),
            )
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            for r in rooms:
                cx, cy = r["centroid"]
                plt.text(
                    cx, cy + 12, f"R{r['id']}",
                    color="yellow", ha="center", va="top", fontsize=9,
                )
            plt.axis("off")
            plt.tight_layout()

        return rooms

    def detect_edges(
        self,
        rooms: list[dict],
        labels: np.ndarray,
        gap_thresh: float = 20.0,
        min_overlap: float = 10.0,
        angle_thresh: float = 0.95,
        show: bool = False,
    ) -> list[list[int]]:
        edges: list[list[int]] = []

        for i in range(len(rooms)):
            poly_a = rooms[i].get("polygon")
            if not poly_a or len(poly_a) < 2:
                continue
            pts_a = np.asarray(poly_a, dtype=float)

            for j in range(i + 1, len(rooms)):
                poly_b = rooms[j].get("polygon")
                if not poly_b or len(poly_b) < 2:
                    continue
                pts_b = np.asarray(poly_b, dtype=float)

                matched = False
                for k in range(len(pts_a)):
                    a1 = pts_a[k]
                    a2 = pts_a[(k + 1) % len(pts_a)]
                    for l in range(len(pts_b)):
                        b1 = pts_b[l]
                        b2 = pts_b[(l + 1) % len(pts_b)]
                        if segments_match(
                            a1, a2, b1, b2,
                            gap_thresh=gap_thresh,
                            min_overlap=min_overlap,
                            angle_thresh=angle_thresh,
                        ):
                            edges.append([rooms[i]["id"], rooms[j]["id"]])
                            matched = True
                            break
                    if matched:
                        break

        if show:
            H, W = labels.shape
            vis = np.zeros((H, W, 3), dtype=np.uint8)

            # faint room fills for context
            colors = self._random_colors(len(rooms))
            for r, c in zip(rooms, colors):
                vis[labels == r["label"]] = (c * 0.4).astype(np.uint8)

            # polygon outlines (this is what edge matching actually operates on)
            for r in rooms:
                if r["polygon"]:
                    pts = np.array(r["polygon"], dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(
                        vis, [pts], isClosed=True, color=(0, 255, 0), thickness=1
                    )

            # adjacency lines between centroids
            id_to_centroid = {r["id"]: r["centroid"] for r in rooms}
            for a, b in edges:
                cv2.line(
                    vis, id_to_centroid[a], id_to_centroid[b],
                    (0, 255, 255), 2,
                )
            for r in rooms:
                cv2.circle(vis, r["centroid"], 6, (0, 0, 255), -1)

            self._new_fig(
                f"Step 8 — Detect edges "
                f"(gap_thresh={gap_thresh}, min_overlap={min_overlap}, "
                f"angle_thresh={angle_thresh}; {len(edges)} edges)",
                figsize=(6, 6),
            )
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            for r in rooms:
                cx, cy = r["centroid"]
                plt.text(
                    cx, cy + 12, f"R{r['id']}",
                    color="yellow", ha="center", va="top", fontsize=9,
                )
            plt.axis("off")
            plt.tight_layout()

        return edges

    def segment_nodes_edges(
        self,
        rooms: list[dict],
        edges: list[list[int]],
        labels: np.ndarray,
        show: bool = False,
    ) -> np.ndarray:
        H, W = labels.shape
        vis = np.zeros((H, W, 3), dtype=np.uint8)

        # Fill rooms using the labels array (clean, exact boundaries)
        colors = self._random_colors(len(rooms))
        for r, c in zip(rooms, colors):
            vis[labels == r["label"]] = c

        # Edges as lines between centroids (double-stroke for visibility)
        id_to_centroid = {r["id"]: r["centroid"] for r in rooms}
        for a, b in edges:
            cv2.line(vis, id_to_centroid[a], id_to_centroid[b], (255, 255, 255), 4)
            cv2.line(vis, id_to_centroid[a], id_to_centroid[b], (0, 0, 0), 2)

        # Node markers
        for r in rooms:
            cv2.circle(vis, r["centroid"], 8, (255, 255, 255), -1)
            cv2.circle(vis, r["centroid"], 8, (0, 0, 0), 2)

        if show:
            self._new_fig(
                f"Step 9 — Segmented nodes & edges "
                f"({len(rooms)} nodes, {len(edges)} edges)",
                figsize=(7, 7),
            )
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            for r in rooms:
                cx, cy = r["centroid"]
                plt.text(
                    cx, cy, f"R{r['id']}",
                    color="black", ha="center", va="center",
                    fontsize=8, fontweight="bold",
                )
            plt.axis("off")
            plt.tight_layout()

        return vis

    def build_graph(
        self,
        image_bgr: np.ndarray,
        border: int = 3,
        kernel_size: int = 15,
        iterations: int = 2,
        min_room_area: int = 500,
        max_room_area: int | None = None,
        poly_epsilon_ratio: float = 0.003,
        gap_thresh: float = 20.0,
        min_overlap: float = 10.0,
        angle_thresh: float = 0.95,
        show: bool = False,
    ) -> dict:
        mask     = self.predict_mask(image_bgr, show=show)
        bordered = self.add_border(mask, border=border, show=show)
        closed   = self.morph_close(
            bordered, kernel_size=kernel_size, iterations=iterations, show=show
        )
        inverted = self.invert(closed, show=show)
        n, labels, stats, cents = self.connected_components(inverted, show=show)
        rooms    = self.filter_rooms(
            n, labels, stats, cents,
            border=border,
            min_room_area=min_room_area,
            max_room_area=max_room_area,
            show=show,
        )
        rooms    = self.detect_nodes(
            rooms, labels, poly_epsilon_ratio=poly_epsilon_ratio, show=show
        )
        edges    = self.detect_edges(
            rooms, labels,
            gap_thresh=gap_thresh,
            min_overlap=min_overlap,
            angle_thresh=angle_thresh,
            show=show,
        )
        _        = self.segment_nodes_edges(rooms, edges, labels, show=show)

        # Build a clean, JSON-serialisable graph (strip internal fields)
        nodes = [
            {
                "id":       r["id"],
                "polygon":  r["polygon"],
                "centroid": list(r["centroid"]),
                "area":     r["area"],
            }
            for r in rooms
        ]
        return {"nodes": nodes, "edges": edges}


# ====================================================================== #
# CLI
# ====================================================================== #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a room adjacency graph from a floor-plan image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required / general
    p.add_argument("--img-path", required=True, help="Path to a floor-plan image.")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to .pth weights.")
    p.add_argument("--device", default=None, help="cuda | cpu (auto-detected).")
    p.add_argument("--input-size", type=int, default=512)

    # hyperparameters
    p.add_argument("--border", type=int, default=3)
    p.add_argument("--kernel-size", type=int, default=15)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--min-room-area", type=int, default=500)
    p.add_argument("--max-room-area", type=int, default=None)
    p.add_argument("--poly-epsilon-ratio", type=float, default=0.003)
    p.add_argument("--gap-thresh", type=float, default=20.0,
                   help="Max perpendicular distance between two polygon segments to call them a shared wall (px).")
    p.add_argument("--min-overlap", type=float, default=10.0,
                   help="Min projected overlap between two segments to call them a shared wall (px).")
    p.add_argument("--angle-thresh", type=float, default=0.95,
                   help="|cos(angle)| threshold for parallelism; 0.95 ≈ within ~18 deg.")

    # display / output
    p.add_argument("--show", action="store_true",
                   help="Render the plot of every step.")
    p.add_argument("--output", default=None,
                   help="Optional path to write the graph as JSON.")

    return p.parse_args()


def main():
    args = parse_args()

    gm = GraphMaker(
        weights_path=args.weights,
        device=args.device,
        input_size=args.input_size,
    )

    image = cv2.imread(args.img_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.img_path}")

    graph = gm.build_graph(
        image,
        border=args.border,
        kernel_size=args.kernel_size,
        iterations=args.iterations,
        min_room_area=args.min_room_area,
        max_room_area=args.max_room_area,
        poly_epsilon_ratio=args.poly_epsilon_ratio,
        gap_thresh=args.gap_thresh,
        min_overlap=args.min_overlap,
        angle_thresh=args.angle_thresh,
        show=args.show,
    )

    print(json.dumps(graph, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(graph, f, indent=2)
        print(f"\nSaved graph to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()