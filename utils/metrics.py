# utils/metrics.py

from skimage.metrics import structural_similarity as ssim
import cv2

def calculate_ssim(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("[ERROR] One or both images not found.")
        return None

    score, _ = ssim(img1, img2, full=True)
    return score


 
