import os
import cv2
from sam import segment_tag_with_sam  # your function

PATH = "../DATA/grid_capture_20250925_172501"

for f in os.listdir(PATH):
    if not f.endswith("raw.png"):
        continue

    img_path = os.path.join(PATH, f)
    print(f"[process] {img_path}")

    try:
        overlay, masked = segment_tag_with_sam(img_path)

        # build output filenames
        base = f.replace("_raw.png", "")
        overlay_path = os.path.join(PATH, base + "_overlay.png")
        masked_path  = os.path.join(PATH, base + "_masked.png")

        # save results
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(masked_path,  cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))

        print(f" -> saved {overlay_path}, {masked_path}")

    except Exception as e:
        print(f" !! failed on {f}: {e}")
