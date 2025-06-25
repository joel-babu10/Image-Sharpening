import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.restormer_mini import RestormerMini
from dataset import ImagePairDataset
from utils.metrics import calculate_psnr, calculate_ssim
from transforms.augmentations import DeblurAugmentation


def tensor_to_image(tensor):
    """Convert [1, 3, H, W] tensor -> [H, W, 3] float32 RGB image in [0, 1]"""
    tensor = tensor.squeeze().detach().cpu().numpy()  # [3, H, W]
    tensor = np.transpose(tensor, (1, 2, 0))          # [H, W, 3]
    return np.clip(tensor, 0.0, 1.0).astype(np.float32)


def add_label(image, label):
    """Draw label on top-left corner of an image (image in [0,1] float32)"""
    img_uint8 = (image * 255).astype(np.uint8)
    labeled = cv2.putText(
        img_uint8.copy(), label, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
    )
    return labeled.astype(np.float32) / 255.0  # Convert back to [0,1] float


def test_teacher():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_blurry_dir = "data/test_degraded"
    test_sharp_dir = "data/test_high_res"
    checkpoint_path = "checkpoints/best_teacher_restormer.pth"
    save_dir = "results/test_comparisons"
    os.makedirs(save_dir, exist_ok=True)

    transform = DeblurAugmentation(image_size=256)
    test_dataset = ImagePairDataset(test_blurry_dir, test_sharp_dir, transform=transform, image_size=256)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = RestormerMini().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    total_psnr, total_ssim = 0.0, 0.0

    with torch.no_grad():
        for i, (blurry, sharp) in enumerate(tqdm(test_loader, desc="Testing")):
            blurry, sharp = blurry.to(device), sharp.to(device)

            output = model(blurry)
            output = torch.clamp(output, 0.0, 1.0)

            # Metrics
            psnr = calculate_psnr(output, sharp)
            ssim = calculate_ssim(output, sharp)
            total_psnr += psnr
            total_ssim += ssim

            # Convert to images in [0,1]
            input_img = tensor_to_image(blurry)
            output_img = tensor_to_image(output)
            gt_img = tensor_to_image(sharp)

            # Add readable labels
            input_img = add_label(input_img, "Input")
            output_img = add_label(output_img, "Output")
            gt_img = add_label(gt_img, "GT")

            # Concatenate and save
            comparison = np.hstack([input_img, output_img, gt_img])
            comparison = (comparison * 255).astype(np.uint8)
            save_path = os.path.join(save_dir, f"compare_{i+1:03d}.png")
            cv2.imwrite(save_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    print(f"\n✅ Test PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    test_teacher()
