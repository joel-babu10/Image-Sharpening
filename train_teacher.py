import os
import csv
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from models.restormer_mini import RestormerMini
from dataset import ImagePairDataset
from utils.losses import ReconstructionLoss
from utils.metrics import calculate_psnr, calculate_ssim
from transforms.augmentations import DeblurAugmentation


def train_teacher():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blurry_dir = "data/degraded"
    sharp_dir = "data/high_res"
    ckpt_dir = "checkpoints"
    result_dir = "results/teacher_outputs"
    log_dir = "training_logs"
    log_path = os.path.join(log_dir, "teacher_metrics.csv")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    transform = DeblurAugmentation(image_size=256)
    dataset = ImagePairDataset(blurry_dir, sharp_dir, transform=transform, image_size=256)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = RestormerMini().to(device)
    criterion = ReconstructionLoss('l1')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 20
    save_interval = 5
    best_psnr = 0.0

    # Initialize CSV log
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss", "psnr", "ssim"])

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for blurry, sharp in pbar:
            blurry, sharp = blurry.to(device), sharp.to(device)

            output = model(blurry)
            loss = criterion(output, sharp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        # Evaluation on one batch
        model.eval()
        with torch.no_grad():
            sample_blurry, sample_sharp = next(iter(dataloader))
            sample_blurry = sample_blurry.to(device)
            sample_sharp = sample_sharp.to(device)

            sample_output = model(sample_blurry)
            sample_output = torch.clamp(sample_output.float(), 0.0, 1.0)
            sample_sharp = torch.clamp(sample_sharp.float(), 0.0, 1.0)

            psnr = calculate_psnr(sample_output, sample_sharp)
            ssim = calculate_ssim(sample_output, sample_sharp)

        # Log metrics
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss, psnr, ssim])

        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            save_image(sample_output, os.path.join(result_dir, f"teacher_epoch_{epoch+1}_output.png"))
            print(f"[Epoch {epoch+1}] PSNR: {psnr:.2f}, SSIM: {ssim:.3f}")

            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_teacher_restormer.pth"))
                print("✅ Saved new best model")


if __name__ == "__main__":
    train_teacher()
