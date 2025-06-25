import cv2
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from models.student_model import StudentUNet  # Match filename and class name

# ----------- CONFIGURATION ------------
video_input_path = 'data/input_video.mp4'
video_output_path = 'results/sharpened_output.mp4'
model_path = 'checkpoints/student_epoch_20.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize_dim = (1920, 1080)  # Ensure matches training input size
show_preview = False  # Set True to show live preview
# --------------------------------------

# Load model
model = StudentUNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transform
to_tensor = transforms.ToTensor()

# Open input video
cap = cv2.VideoCapture(video_input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

resize_needed = (width, height) != resize_dim

# Setup output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_path, fourcc, fps, resize_dim)

print(f"🔧 Processing {frame_count} frames at {fps} FPS...")

with torch.no_grad():
    for _ in tqdm(range(frame_count), desc="Sharpening Video"):
        ret, frame = cap.read()
        if not ret:
            break

        if resize_needed:
            frame = cv2.resize(frame, resize_dim)

        # BGR -> RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = to_tensor(img_rgb).unsqueeze(0).to(device)

        # Inference
        sharpened = model(tensor)
        sharpened = torch.clamp(sharpened, 0.0, 1.0)

        # Convert to image
        output_img = sharpened.squeeze().cpu().permute(1, 2, 0).numpy()
        output_img = (output_img * 255.0).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        out.write(output_bgr)

        # Optional display
        if show_preview:
            cv2.imshow("Sharpened", output_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Cleanup
cap.release()
out.release()
if show_preview:
    cv2.destroyAllWindows()

print(f"✅ Sharpened video saved at: {video_output_path}")
