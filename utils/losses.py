import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class ReconstructionLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss = nn.L1Loss() if loss_type == 'l1' else nn.MSELoss()

    def forward(self, student_out, target_hr):
        return self.loss(student_out, target_hr)


class PerceptualLoss(nn.Module):
    def __init__(self, layer=8):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:layer].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, student_out, teacher_out):
        s = self.vgg(self._normalize(student_out))
        t = self.vgg(self._normalize(teacher_out))
        return F.mse_loss(s, t)

    def _normalize(self, x):
        # Normalization expected by VGG16
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std


class FeatureDistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_features, teacher_features):
        loss = 0
        for sf, tf in zip(student_features, teacher_features):
            loss += self.mse(sf, tf)
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        edge_kernel = torch.tensor(
            [[[-1., -1., -1.],
              [-1.,  8., -1.],
              [-1., -1., -1.]]], dtype=torch.float32
        ).expand(3, 1, 3, 3)  # Expand for RGB

        self.edge_kernel = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.edge_kernel.weight.copy_(edge_kernel)
        self.edge_kernel.weight.requires_grad = False

        self.l1 = nn.L1Loss()

    def forward(self, student_out, target_hr):
        edge_student = self.edge_kernel(student_out)
        edge_target = self.edge_kernel(target_hr)
        return self.l1(edge_student, edge_target)
