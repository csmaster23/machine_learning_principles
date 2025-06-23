import os
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.collections import LineCollection
import umap.umap_ as umap
import numpy as np
import imageio.v2 as imageio
from matplotlib.lines import Line2D

# --- CONFIG ---
N_SAMPLES = 500
QUERY_INDEX = 12  # index of the query image (e.g., a car or dog)
SAVE_DIR = "gif_frames_radar_lines"
GIF_PATH = "similarity_radar_lines.gif"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Load and preprocess CIFAR-10 (cars and dogs only) ---
transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])
dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
target_classes = [1, 5]  # 1=automobile, 5=dog
indices = [i for i, (_, label) in enumerate(dataset) if label in target_classes][:N_SAMPLES]
subset = Subset(dataset, indices)
loader = DataLoader(subset, batch_size=64, shuffle=False)

# --- Load pretrained ResNet18 and extract embeddings ---
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model.eval()

embeddings, labels = [], []
with torch.no_grad():
    for imgs, lbls in loader:
        feats = model(imgs)
        embeddings.append(feats)
        labels.append(lbls)

embeddings = torch.cat(embeddings).numpy()
labels = torch.cat(labels).numpy()

# --- UMAP dimensionality reduction ---
reducer = umap.UMAP(random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

# --- Compute cosine similarity from query point ---
query_vec = embeddings[QUERY_INDEX].reshape(1, -1)
sims = cosine_similarity(query_vec, embeddings).flatten()
sorted_indices = np.argsort(-sims)  # high similarity first

# --- Normalize similarities for color mapping ---
norm = colors.Normalize(vmin=sims.min(), vmax=sims.max())
cmap = cm.get_cmap("RdYlGn")

# --- Animation config ---
num_frames = 30
steps = np.linspace(1, len(sorted_indices), num_frames).astype(int)

# --- Generate frames ---
for frame_num, k in enumerate(steps, 1):
    fig, ax = plt.subplots(figsize=(7, 6))

    # --- Color points by ground truth labels ---
    label_colors = ['blue' if lbl == 1 else 'orange' for lbl in labels]
    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], color=label_colors, s=20, zorder=1)

    # --- Query point (black w/ yellow outline) ---
    qx, qy = embedding_2d[QUERY_INDEX]
    ax.scatter(qx, qy, color='black', edgecolor='yellow', s=100, linewidth=2, zorder=3)

    # --- Draw similarity lines from query to top-K neighbors ---
    segments, line_colors = [], []
    for i in sorted_indices[:k]:
        if i == QUERY_INDEX:
            continue
        px, py = embedding_2d[i]
        segments.append([[qx, qy], [px, py]])
        line_colors.append(cmap(norm(sims[i])))

    lc = LineCollection(segments, colors=line_colors, linewidths=2, zorder=2)
    ax.add_collection(lc)

    # --- Colorbar showing cosine similarity scale ---
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Cosine Similarity to Query")

    # --- Legend for classes and query ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Car (Ground Truth)', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Dog (Ground Truth)', markerfacecolor='orange', markersize=8),
        Line2D([0], [0], marker='o', color='black', markeredgecolor='yellow', label='Query Image', markersize=10, markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True)

    ax.set_title(f"Similarity Radar — Top {k} Neighbors")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/frame_{frame_num:02d}.png")
    plt.close()

# --- Compile GIF ---
frames = [imageio.imread(f"{SAVE_DIR}/frame_{i:02d}.png") for i in range(1, num_frames + 1)]
imageio.mimsave(GIF_PATH, frames, duration=0.8)
