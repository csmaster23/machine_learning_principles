import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import imageio.v2 as imageio
import os

# Set PyTorch to CPU only
device = torch.device("cpu")

# Step 1: Generate dataset
X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Train the teacher model
teacher = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
teacher.fit(X_train, y_train)
teacher_probs = teacher.predict_proba(X_train)

# Convert training data to torch tensors
X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
teacher_soft = torch.tensor(teacher_probs, dtype=torch.float32).to(device)

# Step 3: Define student model
class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )

    def forward(self, x):
        return self.net(x)

student = StudentNet().to(device)
optimizer = optim.Adam(student.parameters(), lr=0.001)
criterion = nn.KLDivLoss(reduction="batchmean")

# Step 4: Create grid for visualization
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100)
)
grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)

# Get teacher predictions on the grid
teacher_grid = teacher.predict_proba(grid)[:, 1].reshape(xx.shape)

# Step 5: Training + Visualization
os.makedirs("frames", exist_ok=True)
images = []

for epoch in range(1, 201):
    student.train()
    optimizer.zero_grad()
    student_logits = student(X_tensor)
    student_log_soft = F.log_softmax(student_logits, dim=1)
    loss = criterion(student_log_soft, teacher_soft)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    if epoch % 5 == 0:
        student.eval()
        with torch.no_grad():
            pred_probs = F.softmax(student(grid_tensor), dim=1)[:, 1].reshape(xx.shape).cpu().numpy()
            diff_map = np.abs(pred_probs - teacher_grid)

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            axs[0].contourf(xx, yy, teacher_grid, levels=20, cmap='coolwarm')
            axs[0].scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=10, alpha=0.5)
            axs[0].set_title("Teacher")

            axs[1].contourf(xx, yy, pred_probs, levels=20, cmap='coolwarm')
            axs[1].scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=10, alpha=0.5)
            axs[1].set_title(f"Student (Epoch {epoch})")

            axs[2].contourf(xx, yy, diff_map, levels=20, cmap='Reds')
            axs[2].set_title("Error vs Teacher")

            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])

            filename = f"frames/frame_{epoch:03d}.png"
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            images.append(imageio.imread(filename))

# Step 6: Save GIF
imageio.mimsave("distillation_demo.gif", images, duration=0.2)
print("✅ GIF saved as distillation_demo.gif")
