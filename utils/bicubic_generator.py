# utils/bicubic_generator.py

import cv2
import os

def simulate_low_quality(input_path, output_path, scale=2):
    img = cv2.imread(input_path)
    if img is None:
        print(f"[ERROR] Image not found: {input_path}")
        return

    h, w = img.shape[:2]

    # Downscale to simulate low-res
    low_res = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)

    # Upscale back to original size
    upscaled = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)

    # Save the upscaled image
    cv2.imwrite(output_path, upscaled)
    print(f"[INFO] Simulated low-quality image saved at: {output_path}")
