# Loss Landscape Script: "Extreme Chaos Version"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation

# --- 1. Create extremely chaotic synthetic data ---
np.random.seed(42)
N = 1000
x = np.linspace(-5, 5, N)
y = np.sin(3 * x) + np.sin(7 * x) + 0.8 * np.random.randn(N)
y += 2.0 * (np.random.rand(N) > 0.7).astype(float)  # Random bigger jumps

x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# --- 2. Define a much deeper and wider chaotic model ---
class ExtremeChaoticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return self.fc5(x)

model = ExtremeChaoticNet()

# --- 3. Quick partial training ---
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    preds = model(x_tensor)
    loss = loss_fn(preds, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Finished partial training. Final loss: {loss.item():.4f}")

# --- 4. Save original parameters and model state ---
saved_state = model.state_dict()
params_vector = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

# --- 5. Create two heavily amplified random directions ---
torch.manual_seed(0)
direction1 = torch.randn_like(params_vector)
direction2 = torch.randn_like(params_vector)

# Normalize and heavily amplify
direction1 /= direction1.norm()
direction2 /= direction2.norm()

direction1 *= 50
direction2 *= 50

# --- 6. Create a loss surface grid with very wide range ---
alpha = np.linspace(-10.0, 10.0, 80)
beta = np.linspace(-10.0, 10.0, 80)

loss_surface = np.zeros((len(alpha), len(beta)))

for i, a in tqdm(enumerate(alpha), total=len(alpha)):
    for j, b in enumerate(beta):
        model.load_state_dict(saved_state)  # Reset model before each perturbation
        perturb = params_vector + a * direction1 + b * direction2
        torch.nn.utils.vector_to_parameters(perturb.clone(), model.parameters())

        preds = model(x_tensor)
        loss = loss_fn(preds, y_tensor)
        loss_surface[i, j] = loss.item()

# Restore original weights
model.load_state_dict(saved_state)

# --- 7. Plot static 3D surface and contour plots ---
A, B = np.meshgrid(alpha, beta)

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, B, loss_surface.T, cmap='inferno')
ax.set_xlabel('Direction 1')
ax.set_ylabel('Direction 2')
ax.set_zlabel('Loss')
ax.set_title('Extreme Chaotic Loss Landscape')
plt.savefig("loss_surface_static.png")
plt.close()

fig = plt.figure(figsize=(10,8))
plt.contourf(A, B, loss_surface.T, levels=100, cmap='inferno')
plt.colorbar(label='Loss')
plt.xlabel('Direction 1')
plt.ylabel('Direction 2')
plt.title('Extreme Chaotic Loss Landscape Contours')
plt.savefig("loss_surface_contour.png")
plt.close()

# --- 8. Create and save GIF sweeping around the landscape ---
fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    elev = 30
    azim = frame
    ax.plot_surface(A, B, loss_surface.T, cmap='inferno')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title('Sweeping Extreme Chaotic Loss Landscape')

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 3), interval=100)
ani.save('loss_landscape.gif', writer='pillow')

print("\nSaved plots:")
print("- loss_surface_static.png")
print("- loss_surface_contour.png")
print("- loss_landscape.gif")

# --- 9. Closing Thoughts ---
print("\nNow you can truly see how chaotic and broken the loss landscape can get!")
print("This is what deep learning optimizers have to navigate: cliffs, ridges, sudden traps, chaotic valleys!")
