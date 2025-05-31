
import cv2
import numpy as np
import os

print("🔁 Script started...")

# Paths
image_path = "C:\\Users\\joelv\\image-sharpening-project\\data\\sample.jpg.jpg"
print("Looking for image at:", image_path)

# Load image
image = cv2.imread(image_path)

if image is None:
    print("❌ Image not loaded.")
    print("🔍 Current working directory:", os.getcwd())
    exit()
else:
    print("✅ Image loaded successfully!")

# Resize
image = cv2.resize(image, (640, 360))

# Blur and Sharpen
blurred = cv2.GaussianBlur(image, (9, 9), 2)
kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
sharpened = cv2.filter2D(blurred, -1, kernel)

# Save outputs
os.makedirs("results", exist_ok=True)
cv2.imwrite("results/original.jpg", image)
cv2.imwrite("results/blurred.jpg", blurred)
cv2.imwrite("results/sharpened.jpg", sharpened)

print("✅ Images saved in 'results' folder.")
