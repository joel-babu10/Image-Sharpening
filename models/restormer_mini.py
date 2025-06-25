# models/restormer_mini.py
import torch
import torch.nn as nn
from einops import rearrange

class EfficientAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        window_size = 8
        assert h % window_size == 0 and w % window_size == 0

        x = rearrange(x, 'b c (h ws1) (w ws2) -> (b h w) c ws1 ws2',
                      ws1=window_size, ws2=window_size)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), qkv)
        q = q * self.scale
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k).softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=window_size, y=window_size)
        out = rearrange(out, '(b h w) c ws1 ws2 -> b c (h ws1) (w ws2)',
                        h=h//window_size, w=w//window_size, ws1=window_size, ws2=window_size)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.attn = EfficientAttention(dim, heads)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = x + self.ffn(self.norm(x))
        return x

class RestormerMini(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dim=32, depth=4):
        super().__init__()
        self.proj_in = nn.Conv2d(in_ch, dim, 3, padding=1)
        self.transformer = nn.Sequential(*[TransformerBlock(dim) for _ in range(depth)])
        self.proj_out = nn.Conv2d(dim, out_ch, 3, padding=1)

    def forward(self, x):
        # x must be float tensor with shape [B, 3, H, W]
        x = self.proj_in(x)
        x = self.transformer(x)
        return torch.sigmoid(self.proj_out(x))
