import torch
import torch.nn.functional as F

# Try importing piq for SSIM, else fallback to skimage
try:
    import piq
    _use_piq = True
except ImportError:
    _use_piq = False
    from skimage.metrics import structural_similarity as skimage_ssim
    import numpy as np


def calculate_psnr(output: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).
    Args:
        output (torch.Tensor): Predicted image tensor (NCHW or CHW).
        target (torch.Tensor): Ground truth image tensor.
        max_val (float): Max possible pixel value (usually 1.0 for normalized).
    Returns:
        float: PSNR value in dB.
    """
    if output.shape != target.shape:
        raise ValueError("Output and target must have the same shape.")
    
    mse = F.mse_loss(output, target, reduction='mean')
    if mse.item() == 0:
        return float('inf')

    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr.item()


def calculate_ssim(output: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Calculate SSIM (Structural Similarity Index).
    Args:
        output (torch.Tensor): Predicted image tensor (NCHW or CHW).
        target (torch.Tensor): Ground truth image tensor.
        data_range (float): Value range (1.0 if normalized).
    Returns:
        float: SSIM score.
    """
    if output.shape != target.shape:
        raise ValueError("Output and target must have the same shape.")

    # Ensure batch shape (N, C, H, W)
    if output.dim() == 3:
        output = output.unsqueeze(0)
        target = target.unsqueeze(0)

    if _use_piq:
        return piq.ssim(output, target, data_range=data_range).item()
    else:
        output_np = output[0].permute(1, 2, 0).cpu().numpy()
        target_np = target[0].permute(1, 2, 0).cpu().numpy()
        return skimage_ssim(output_np, target_np, channel_axis=2, data_range=data_range)
