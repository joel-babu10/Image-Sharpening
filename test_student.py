import os
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.student_model import StudentUNet
from dataset import ImagePairDataset
from transforms.augmentations import DeblurAugmentation
from utils.metrics import calculate_psnr, calculate_ssim

def tensor_to_image(tensor):
    tensor = tensor.squeeze().detach().cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))  # CHW -> HWC
    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
    return tensor

def add_label(image, label):
    return cv2.putText(image.copy(), label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0), 2, cv2.LINE_AA)

def test_student():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_blurry_dir = "data/test_degraded"
    test_sharp_dir = "data/test_high_res"
    checkpoint_path = "checkpoints/best_student_unet.pth"
    save_dir = "results/test_student_comparisons"
    os.makedirs(save_dir, exist_ok=True)

    transform = DeblurAugmentation(image_size=256)
    test_dataset = ImagePairDataset(test_blurry_dir, test_sharp_dir, transform=transform, image_size=256)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = StudentUNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    total_psnr, total_ssim = 0.0, 0.0
    results = []

    with torch.no_grad():
        for i, (blurry, sharp) in enumerate(tqdm(test_loader, desc="Testing Student")):
            blurry, sharp = blurry.to(device), sharp.to(device)

            output = model(blurry)
            output = torch.clamp(output, 0.0, 1.0)

            psnr = calculate_psnr(output, sharp)
            ssim = calculate_ssim(output, sharp)
            total_psnr += psnr
            total_ssim += ssim

            # Save metrics for CSV
            results.append({
                "Image": f"compare_{i+1:03d}.png",
                "PSNR": psnr,
                "SSIM": ssim
            })

            # Save comparison image
            input_img = tensor_to_image(blurry)
            output_img = tensor_to_image(output)
            gt_img = tensor_to_image(sharp)

            input_img = add_label(input_img, "Input")
            output_img = add_label(output_img, "Output")
            gt_img = add_label(gt_img, "GT")

            comparison = np.hstack([input_img, output_img, gt_img])
            cv2.imwrite(os.path.join(save_dir, f"compare_{i+1:03d}.png"),
                        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)

    print(f"\n✅ Student Test PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.loc[len(df.index)] = ["Average", avg_psnr, avg_ssim]
    df.to_csv(os.path.join(save_dir, "student_test_metrics.csv"), index=False)
    print(f"📄 Metrics saved to {os.path.join(save_dir, 'student_test_metrics.csv')}")

if __name__ == "__main__":
    test_student()
