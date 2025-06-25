import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from models.student_model import StudentUNet
from models.restormer_mini import RestormerMini
from dataset import ImagePairDataset
from transforms.augmentations import DeblurAugmentation
from utils.distillation_loss import KnowledgeDistillationLoss
from utils.metrics import calculate_psnr, calculate_ssim


def train_student():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blurry_dir = "data/degraded"
    sharp_dir = "data/high_res"
    ckpt_dir = "checkpoints"
    result_dir = "results/student_outputs"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    transform = DeblurAugmentation(image_size=256)
    dataset = ImagePairDataset(blurry_dir, sharp_dir, transform=transform, image_size=256)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    student = StudentUNet().to(device)
    teacher = RestormerMini().to(device)
    teacher_ckpt = "checkpoints/best_teacher_restormer.pth"
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    criterion = KnowledgeDistillationLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

    epochs = 20
    save_interval = 5
    best_psnr = 0.0

    for epoch in range(epochs):
        student.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for blurry, sharp in pbar:
            blurry, sharp = blurry.to(device), sharp.to(device)

            with torch.no_grad():
                teacher_out = teacher(blurry)

            student_out = student(blurry)

            loss = criterion(student_out, sharp, teacher_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            student.eval()
            with torch.no_grad():
                sample_blurry, sample_sharp = next(iter(dataloader))
                sample_blurry = sample_blurry.to(device)
                sample_sharp = sample_sharp.to(device)

                sample_output = student(sample_blurry)
                sample_output = torch.clamp(sample_output.float(), 0.0, 1.0)
                sample_sharp = torch.clamp(sample_sharp.float(), 0.0, 1.0)

                save_image(sample_output, os.path.join(result_dir, f"student_epoch_{epoch+1}_output.png"))

                psnr = calculate_psnr(sample_output, sample_sharp)
                ssim = calculate_ssim(sample_output, sample_sharp)
                print(f"[Epoch {epoch+1}] PSNR: {psnr:.2f}, SSIM: {ssim:.3f}")

                if psnr > best_psnr:
                    best_psnr = psnr
                    torch.save(student.state_dict(), os.path.join(ckpt_dir, "best_student_unet.pth"))
                    print("✅ Saved new best student model")


if __name__ == "__main__":
    train_student()
