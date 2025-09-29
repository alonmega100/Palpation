import os
import json
import cv2
import numpy as np
from pathlib import Path
from sam import segment_tag_with_sam  # your function

PATH = "../../DATA/grid_capture_20250925_172501"
PAD_RATIO = 0.20          # 20% padding around the tight bbox
MIN_BOX = 16              # ignore tiny specks; require at least this many pixels in width/height
RESIZE_TO = 256           # set to None to keep native crop size (else, int like 224/256/320)

def crop_from_mask(orig_bgr: np.ndarray, masked_rgb: np.ndarray,
                   pad_ratio: float = PAD_RATIO, min_box: int = MIN_BOX):
    """
    Derive a binary mask from 'masked' (non-black pixels), compute padded bbox,
    and crop from the ORIGINAL image.
    Returns crop_bgr, (x0,y0,x1,y1) in original coords.
    """
    H, W = orig_bgr.shape[:2]

    # Build a binary mask: any non-near-black pixel in masked => foreground
    # (robust to compression noise via threshold)
    gray = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > 5).astype(np.uint8)  # 0/1

    # Optional: remove tiny specks / fill small holes
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if mask.sum() == 0:
        raise ValueError("Empty mask (no foreground pixels).")

    # Find tight bbox via contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(np.vstack(cnts))

    if w < min_box or h < min_box:
        raise ValueError(f"Tiny bbox ({w}x{h}); likely bad segmentation.")

    # Pad bbox
    pad = int(round(max(w, h) * pad_ratio))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)

    crop_bgr = orig_bgr[y0:y1, x0:x1].copy()
    return crop_bgr, (int(x0), int(y0), int(x1), int(y1))

def maybe_resize(img_bgr: np.ndarray, size: int | None):
    if size is None:
        return img_bgr
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)

def process_dir(path: str):
    path = Path(path)
    for f in os.listdir(path):
        if not f.endswith("raw.png"):
            continue

        img_path = path / f
        print(f"[process] {img_path}")

        try:
            # Load original and run segmentation
            orig_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if orig_bgr is None:
                raise RuntimeError("Failed to read image.")

            overlay_rgb, masked_rgb = segment_tag_with_sam(str(img_path))  # assumed RGB

            # Save overlay/masked (optional; keep your existing behavior)
            base = f.replace("_raw.png", "")
            overlay_path = path / f"{base}_overlay.png"
            masked_path  = path / f"{base}_masked.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(masked_path),  cv2.cvtColor(masked_rgb,  cv2.COLOR_RGB2BGR))

            # --- Crop around the mask ---
            crop_bgr, (x0, y0, x1, y1) = crop_from_mask(orig_bgr, masked_rgb)

            # Optional resize to square (keeps model input consistent)
            crop_bgr_out = maybe_resize(crop_bgr, RESIZE_TO)

            crop_path = path / f"{base}_crop.png"
            cv2.imwrite(str(crop_path), crop_bgr_out)

            # Save crop metadata so you can keep absolute position later
            meta = {
                "orig_size": {"w": int(orig_bgr.shape[1]), "h": int(orig_bgr.shape[0])},
                "crop_bbox_xyxy": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "pad_ratio": PAD_RATIO,
                "resized_to": RESIZE_TO if RESIZE_TO is not None else "native",
            }
            meta_path = path / f"{base}_crop_meta.json"
            with open(meta_path, "w") as jf:
                json.dump(meta, jf, indent=2)

            print(f" -> saved {overlay_path.name}, {masked_path.name}, {crop_path.name}, {meta_path.name}")

        except Exception as e:
            print(f" !! failed on {f}: {e}")

if __name__ == "__main__":
    process_dir(PATH)
