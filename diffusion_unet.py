import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from PIL import Image
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb  # shape: [batch, dim]


# Hyperparameters
T = 100
IMG_SIZE = 28
BATCH_SIZE = 256
EPOCHS = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Diffusion coefficients
betas = torch.linspace(1e-4, 0.02, T).to(DEVICE)
alphas = 1. - betas
alpha_bars = torch.cumprod(alphas, dim=0).to(DEVICE)


def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = torch.sqrt(alpha_bars[t])[:, None, None, None]
    sqrt_1mab = torch.sqrt(1 - alpha_bars[t])[:, None, None, None]
    return sqrt_ab * x0 + sqrt_1mab * noise

class BetterUNetWithTime(nn.Module):
    def __init__(self, time_emb_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU()
            )
        self.down1 = conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.middle = conv_block(64, 128)
        self.time_to_middle = nn.Linear(time_emb_dim, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = conv_block(64 + 64, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up1 = conv_block(32 + 32, 32)
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)  # [B, time_emb_dim]
        # Down
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        m = self.middle(p2)
        # Add time embedding to middle
        t_middle = self.time_to_middle(t_emb).view(-1, 128, 1, 1)
        m = m + t_middle
        # Up
        u2 = self.up2(m)
        u2 = self.conv_up2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2)
        u1 = self.conv_up1(torch.cat([u1, d1], dim=1))
        return self.final(u1)



# Load data
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model setup
# model = SimpleUNet().to(DEVICE)
model = BetterUNetWithTime().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Train loop
for epoch in range(EPOCHS):
    for x, _ in loader:
        x = x.to(DEVICE)
        t = torch.randint(0, T, (x.size(0),), device=DEVICE)
        noise = torch.randn_like(x)
        x_noisy = q_sample(x, t, noise)
        pred = model(x_noisy, t)
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/diffusion_mnist_unet.pth")
print("Model saved at checkpoints/diffusion_mnist_unet.pth")

# Sampling from noise
@torch.no_grad()
def p_sample(x, t, model):
    t_tensor = torch.tensor([t], device=DEVICE).long()
    pred_noise = model(x, t_tensor)
    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bars[t]

    sqrt_one_over_alpha = 1. / torch.sqrt(alpha_t)
    sqrt_one_minus_ab = torch.sqrt(1 - alpha_bar_t)

    x_prev = sqrt_one_over_alpha * (x - (beta_t / sqrt_one_minus_ab) * pred_noise)
    if t > 0:
        noise = torch.randn_like(x)
        x_prev += torch.sqrt(beta_t) * noise
    return x_prev

# Generate GIF
x = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
frames = []

for t in reversed(range(T)):
    x = p_sample(x, t, model)
    if t % 2 == 0:  # skip every other frame
        # frame = x.squeeze().cpu().clamp(0, 1).numpy()
        # frames.append((frame * 255).astype(np.uint8))
        img = x.squeeze().cpu().clamp(0, 1).numpy()
        img_uint8 = (img * 255).astype(np.uint8)
        # Upscale to 256x256 using nearest-neighbor to keep pixelated style
        img_big = Image.fromarray(img_uint8).resize((512, 512), resample=Image.NEAREST)
        frames.append(np.array(img_big))

os.makedirs("outputs", exist_ok=True)
imageio.mimsave("outputs/diffusion_result.gif", frames, duration=0.016)
print("GIF saved at outputs/diffusion_result.gif")
