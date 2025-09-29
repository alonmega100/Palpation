import cv2
import numpy as np
import torch
from pupil_apriltags import Detector
from segment_anything import sam_model_registry, SamPredictor

# --- Load SAM once globally (faster if you call function multiple times) ---
SAM_CKPT = "../DATA/sam_vit_h_4b8939.pth"
SAM_TYPE = "vit_h"   # vit_b / vit_l / vit_h
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(device)
predictor = SamPredictor(sam)

# --- Function ---
def segment_tag_with_sam(img_path, pad_pixels=0):
    """
    Detects the largest AprilTag in the image, uses SAM to segment it.
    Returns:
        overlay_rgb: original image with green overlay on mask
        masked_rgb: black background with only the segmented region
    """
    # Load image
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Detect AprilTags (2D only)
    detector = Detector(families="tag36h11", nthreads=2,
                        quad_decimate=1.0, refine_edges=True)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dets = detector.detect(gray)
    if not dets:
        raise RuntimeError("No AprilTag detected in image.")

    # Pick largest tag
    def quad_area(corners): return cv2.contourArea(corners.astype(np.float32))
    tag = max(dets, key=lambda d: quad_area(d.corners))
    corners = tag.corners

    # Build padded box
    x0, y0 = np.min(corners[:, 0]), np.min(corners[:, 1])
    x1, y1 = np.max(corners[:, 0]), np.max(corners[:, 1])
    H, W = gray.shape
    x0, y0 = max(0, x0 - pad_pixels), max(0, y0 - pad_pixels)
    x1, y1 = min(W - 1, x1 + pad_pixels), min(H - 1, y1 + pad_pixels)
    box = np.array([[x0, y0, x1, y1]], dtype=np.float32)

    # Run SAM
    predictor.set_image(rgb)
    with torch.no_grad():
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    mask = masks[np.argmax(scores)]

    # Build results
    overlay_rgb = rgb.copy()
    overlay_rgb[mask] = (0.3 * overlay_rgb[mask] + 0.7 * np.array([0, 255, 0])).astype(np.uint8)

    masked_rgb = np.zeros_like(rgb)
    masked_rgb[mask] = rgb[mask]

    return overlay_rgb, masked_rgb


# === Example usage ===
if __name__ == "__main__":
    img_path = "../../DATA/grid_capture_20250925_172501/grid_0000_dx-0.200_dy-0.200_dz+0.150_yaw-8.0_pit+0.3_rol+3.7_raw.png"
    overlay, masked = segment_tag_with_sam(img_path)

    cv2.imwrite("../overlay_out.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite("../masked_out.png", cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
    print("Saved overlay_out.png and masked_out.png")
