import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


class GraphMaker:
    """
    Wraps a MIT-UNet wall segmentation model and turns floor-plan images
    into a room adjacency graph.

    Pipeline:
        image (np.ndarray, BGR) -> wall mask -> rooms (CC + filtering) -> graph
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

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
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

        state_dict = torch.load(self.weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------ #
    # Step 1: wall segmentation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_wall_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            image_bgr: HxWx3 BGR image (as returned by cv2.imread / cv2.imdecode)

        Returns:
            Binary wall mask of shape (input_size, input_size), dtype float32 in {0., 1.}
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image passed to predict_wall_mask")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image_rgb)
        input_tensor = augmented["image"].unsqueeze(0).to(self.device)

        logits = self.model(input_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float()
        return mask.squeeze().cpu().numpy()

    # ------------------------------------------------------------------ #
    # Step 2: room extraction from the wall mask
    # ------------------------------------------------------------------ #
    def find_rooms(
        self,
        wall_mask: np.ndarray,
        border: int = 1,
        kernel_size: int = 15,
        iterations: int = 2,
        min_room_area: int = 500,
        max_room_area: int | None = 50_000,
        verbose: bool = False,
    ) -> tuple[list[dict], dict]:
        """
        Returns:
            rooms: list of dicts (id, label, bbox, polygon, centroid, area, contour)
            debug: dict with intermediate masks (wall_img, closed, labels)
        """
        wall_img = (wall_mask > 0).astype(np.uint8) * 255

        # Seal the image border so "outside" can't escape through gaps
        wall_img = cv2.copyMakeBorder(
            wall_img, border, border, border, border,
            cv2.BORDER_CONSTANT, value=255,
        )

        # Close gaps (doors/windows) so rooms become enclosed
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        closed = cv2.morphologyEx(
            wall_img, cv2.MORPH_CLOSE, kernel, iterations=iterations
        )

        # Invert -> rooms become white blobs
        inverted = cv2.bitwise_not(closed)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            inverted, connectivity=4
        )

        H_pad, W_pad = inverted.shape
        if max_room_area is None:
            max_room_area = int(H_pad * W_pad * 0.5)

        # Any component touching the padded edge band is "outside"
        edge_band = border + 2
        edge_labels = set()
        edge_labels.update(labels[:edge_band, :].flatten())
        edge_labels.update(labels[-edge_band:, :].flatten())
        edge_labels.update(labels[:, :edge_band].flatten())
        edge_labels.update(labels[:, -edge_band:].flatten())
        edge_labels.discard(0)  # background

        if verbose:
            print(f"Total components (excl. background): {num_labels - 1}")

        rooms: list[dict] = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]

            if not (min_room_area <= area <= max_room_area):
                if verbose:
                    print(f"  [skip-area]    comp {i}: area={area}")
                continue
            if i in edge_labels:
                if verbose:
                    print(f"  [skip-outside] comp {i}: area={area}")
                continue

            comp_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            dense_contour = contours[0] if contours else None

            polygon_list = None
            if dense_contour is not None:
                epsilon = 0.01 * cv2.arcLength(dense_contour, True)
                approx = cv2.approxPolyDP(dense_contour, epsilon, True)
                # Always return as [[x, y], ...]
                polygon_list = approx.reshape(-1, 2).tolist()

            rooms.append({
                "id": len(rooms) + 1,
                "label": int(i),
                "bbox": (int(x), int(y), int(w), int(h)),
                "polygon": polygon_list,
                "centroid": (int(cx), int(cy)),
                "area": int(area),
                "contour": dense_contour,  # kept for edge detection; stripped before JSON
            })

            if verbose:
                print(f"  [KEEP]         comp {i}: area={area}, bbox=({x},{y},{w},{h})")

        debug = {"wall_img": wall_img, "closed": closed, "labels": labels}
        return rooms, debug

    # ------------------------------------------------------------------ #
    # Step 3: graph construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_graph_from_rooms(
        rooms: list[dict],
        gap_thresh: float = 5.0,
        min_close_points: int = 20,
    ) -> dict:
        """
        Two rooms are connected if their dense contours share at least
        `min_close_points` points within `gap_thresh` pixels of each other.
        """
        nodes = [
            {
                "id": r["id"],
                "polygon": r["polygon"],
                "bbox": list(r["bbox"]),
                "centroid": list(r["centroid"]),
                "area": r["area"],
            }
            for r in rooms
        ]

        edges: list[list[int]] = []
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                ca = rooms[i]["contour"]
                cb = rooms[j]["contour"]
                if ca is None or cb is None:
                    continue

                pts_a = ca.reshape(-1, 2)
                pts_b = cb.reshape(-1, 2)

                close_points = 0
                for pa in pts_a:
                    dmin = np.sqrt(((pts_b - pa) ** 2).sum(axis=1)).min()
                    if dmin <= gap_thresh:
                        close_points += 1
                    if close_points >= min_close_points:
                        edges.append([rooms[i]["id"], rooms[j]["id"]])
                        break

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------ #
    # End-to-end
    # ------------------------------------------------------------------ #
    def build_graph(
        self,
        image_bgr: np.ndarray,
        border: int = 3,
        kernel_size: int = 15,
        iterations: int = 2,
        min_room_area: int = 500,
        max_room_area: int | None = None,
        gap_thresh: float = 5.0,
        min_close_points: int = 20,
        verbose: bool = False,
    ) -> dict:
        wall_mask = self.predict_wall_mask(image_bgr)
        rooms, _ = self.find_rooms(
            wall_mask,
            border=border,
            kernel_size=kernel_size,
            iterations=iterations,
            min_room_area=min_room_area,
            max_room_area=max_room_area,
            verbose=verbose,
        )
        return self.build_graph_from_rooms(
            rooms,
            gap_thresh=gap_thresh,
            min_close_points=min_close_points,
        )

    # ------------------------------------------------------------------ #
    # Convenience: file path wrapper (for local testing / scripts)
    # ------------------------------------------------------------------ #
    def build_graph_from_path(self, img_path: str, **kwargs) -> dict:
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        return self.build_graph(image, **kwargs)


if __name__ == "__main__":
    import os

    weights = os.environ.get(
        "MITUNET_WEIGHTS",
        "/opt/dlami/nvme/mitunet_weights/mitunet.pth",
    )
    dataset_path = "/opt/dlami/nvme/CPMS_ClusterPlans/plans"

    gm = GraphMaker(weights_path=weights)

    img_path = f"{dataset_path}/page10_img25.png"

    graph = gm.build_graph_from_path(img_path, verbose=True)

    print("\n<<< Graph >>>")
    print(f"Nodes: {len(graph['nodes'])}")
    print(f"Edges: {len(graph['edges'])}")
    for e in graph["edges"]:
        print(f"  {e[0]} <-> {e[1]}")