import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = "/opt/dlami/nvme/mitunet_weights/mitunet.pth"

aux_segformer = smp.Segformer(encoder_name="mit_b4", encoder_weights=None)
model = smp.Unet(
    encoder_name="mit_b4",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    decoder_attention_type="scse"
)
# Transplant the encoder
model.encoder = aux_segformer.encoder

# 3. Load trained weights
state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def predict(img_path, mask_path=None, show=False):

    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    augmented = transform(image=image_rgb)
    input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float()

    result_mask = mask.squeeze().cpu().numpy()

    # ---------------------------------
    # Load GT mask if provided
    # ---------------------------------
    gt_mask = None

    if mask_path is not None:
        gt_mask = cv2.imread(mask_path, 0)

        # wall only
        gt_mask = (gt_mask == 1).astype(np.uint8)

    # ---------------------------------
    # Plot
    # ---------------------------------
    if show:

        cols = 3 if gt_mask is not None else 2

        plt.figure(figsize=(18, 6))

        # Original image
        plt.subplot(1, cols, 1)
        plt.title("Original Image")
        plt.imshow(image_rgb)
        plt.axis('off')

        # Prediction
        plt.subplot(1, cols, 2)
        plt.title("Predicted Mask")
        plt.imshow(result_mask, cmap='gray')
        plt.axis('off')

        # GT mask
        if gt_mask is not None:
            plt.subplot(1, cols, 3)
            plt.title("GT Wall Mask")
            plt.imshow(gt_mask, cmap='gray')
            plt.axis('off')

        plt.show()

    return result_mask

def find_room_boundaries(img_path, pred_wall, border=3, kernel_size=15, iterations=2, min_room_area = 500, max_room_area = 50000):
    # Make a clean binary wall image from your straight_img
    wall_img = (pred_wall > 0).astype(np.uint8) * 255

    # Seal the image border so "outside" can't escape through missing perimeter walls
    wall_img = cv2.copyMakeBorder(wall_img, border, border, border, border,
                                cv2.BORDER_CONSTANT, value=255)

    # Close gaps (doors / windows) so rooms become enclosed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(wall_img, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # Invert -> rooms become white blobs, walls become black barriers
    inverted = cv2.bitwise_not(closed)

    # H_pad, W_pad = inverted.shape
    # seeds = [(border, border),
    #         (W_pad - 1 - border, border),
    #         (border, H_pad - 1 - border),
    #         (W_pad - 1 - border, H_pad - 1 - border)]

    # for sx, sy in seeds:
    #     if inverted[sy, sx] == 255:   # this pixel is part of the outside region
    #         cv2.floodFill(inverted, None, (sx, sy), 0)

    # Find connected components (each blob = one room candidate)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inverted, connectivity=4
    )

    # Filter out the "outside" region and noise
    H, W = inverted.shape
    total_area = H * W
    if max_room_area is None:
        max_room_area = total_area * 0.5    

    print(f"Total components found (excl. background): {num_labels - 1}")
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        keep = "KEEP" if min_room_area <= area <= max_room_area else "skip"
        print(f"  [{keep}] comp {i}: area={area:>7}, bbox=({x},{y},{w},{h})")

    rooms = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        if area < min_room_area or area > max_room_area:
            continue

        # Get the room's outline as a polygon
        mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0] if contours else None

        epsilon = 0.02 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)

        rooms.append({
            'id':       len(rooms) + 1,
            'label':    i,
            'bbox':     (int(x), int(y), int(w), int(h)),   # x, y, width, height
            'polygon': contour.squeeze().tolist() if contour is not None else None,
            'centroid': (int(cx), int(cy)),
            'area':     int(area),
            'contour':  contour,
        })

    print(f"Detected {len(rooms)} rooms")
    for r in rooms:
        print(f"  Room {r['id']:>2}: bbox={r['bbox']}, centroid={r['centroid']}, area={r['area']}")

    # Colored room fills
    np.random.seed(42)
    room_vis = cv2.cvtColor(wall_img, cv2.COLOR_GRAY2BGR)
    for r in rooms:
        color = np.random.randint(60, 255, size=3).tolist()
        room_vis[labels == r['label']] = color

    # Bounding boxes + IDs on top of the wall image
    bbox_vis = cv2.cvtColor(wall_img, cv2.COLOR_GRAY2BGR)
    for r in rooms:
        x, y, w, h = r['bbox']
        cv2.rectangle(bbox_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(bbox_vis, f"R{r['id']}", r['centroid'],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Plot all stages
    plt.figure(figsize=(20, 5))
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 5, 1)
    plt.title("Input Image")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.title("Wall lines (input)")
    plt.imshow(wall_img, cmap='gray'); plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.title(f"After closing (k={kernel_size})")
    plt.imshow(closed, cmap='gray'); plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.title(f"Rooms ({len(rooms)} found)")
    plt.imshow(cv2.cvtColor(room_vis, cv2.COLOR_BGR2RGB)); plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.title("Bounding boxes")
    plt.imshow(cv2.cvtColor(bbox_vis, cv2.COLOR_BGR2RGB)); plt.axis('off')

    plt.tight_layout()
    # plt.show()

    plt.savefig("room_detection.png", dpi=300)
    plt.close()

    return rooms

# make the co-ords as nodes and create a graph with edges as the walls. Then we can apply graph algorithms to find rooms, connectivity, etc.
def build_graph_from_mask(rooms):
    # create the graph
    graph = {
        "nodes": {},
        "edges": []
    }
    # add nodes
    for room in rooms:
        nid = room['id']
        graph["nodes"][nid] = room['polygon']

    # edges
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            room_a = rooms[i]
            room_b = rooms[j]

            # Check if they share a wall (i.e., their contours are close)
            contour_a = room_a['contour']
            contour_b = room_b['contour']

            # TODO: does both contours points are close to each other with threshold atleast 2-3 points? not with centroid
            if contour_a is not None and contour_b is not None:

                pts_a = contour_a.reshape(-1, 2)
                pts_b = contour_b.reshape(-1, 2)

                close_pts = 0
                dist_thresh = 20
                min_shared_points = 1

                for pt in pts_a:
                    # signed distance to contour_b
                    dist = abs(cv2.pointPolygonTest(
                        contour_b,
                        (float(pt[0]), float(pt[1])),
                        True
                    ))

                    if dist <= dist_thresh:
                        close_pts += 1

                    if close_pts >= min_shared_points:
                        graph["edges"].append((room_a['id'], room_b['id']))
                        break

    # print the graph
    print("Graph:")
    print("Nodes:")
    for nid, coord in graph["nodes"].items():
        print(f"  {nid}: {coord}")
    print("Edges:")
    for edge in graph["edges"]:
        print(f"  {edge[0]} <-> {edge[1]}")

    return graph

def build_graph(img_path, show=False, border=3, kernel_size=15, iterations=2, min_room_area = 500, max_room_area = None):
    pred_wall = predict(img_path, show=False)
    rooms = find_room_boundaries(img_path, pred_wall, border=border, kernel_size=kernel_size, iterations=iterations, min_room_area=min_room_area, max_room_area=max_room_area)
    return build_graph_from_mask(rooms)

if __name__ == "__main__":
    dataset_path = "/opt/dlami/nvme/mitunet_dataset_1/images"

    idx = np.random.randint(0, 100)  # Randomly select an index for testing
    img_path = f"{dataset_path}/{idx:05d}.png"
    build_graph(img_path)