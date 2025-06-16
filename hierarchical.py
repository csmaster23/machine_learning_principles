import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from scipy.cluster.hierarchy import linkage, dendrogram
import imageio
import os
import shutil
import matplotlib.animation as animation

# ---------------------
# CONFIGURATION
# ---------------------
n_samples = 30
fade_frames = 5
gif_name = "hierarchical_moons_synced.gif"
mp4_name = "hierarchical_moons_synced.mp4"
final_pause_frames = 10
legend_on = True

# ---------------------
# DATA
# ---------------------
np.random.seed(42)
X, _ = make_moons(n_samples=n_samples, noise=0.07)
n = len(X)
Z = linkage(X, method="ward")

os.makedirs("frames", exist_ok=True)

cluster_assignments = {i: i for i in range(n)}
cluster_to_color = {i: plt.cm.tab20(i % 20) for i in range(n)}
recently_merged_points = set()

# ---------------------
# COLOR HELPERS
# ---------------------
def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )
def make_link_color_func(cluster_to_color, step):
    def link_color_func(cluster_id):
        # Allow color only if this cluster merge has occurred
        if cluster_id >= n + step:
            return "#cccccc"
        rgb = cluster_to_color.get(cluster_id)
        if rgb:
            hex_color = rgb_to_hex(rgb[:3])
            return hex_color
        else:
            return "#aaaaaa"
    return link_color_func






# ---------------------
# MAIN LOOP
# ---------------------
frame_idx = 0
for step in range(1, len(Z) + 1):
    a, b = int(Z[step - 1, 0]), int(Z[step - 1, 1])
    new_cluster_id = n + step - 1
    points_to_update = [idx for idx, cid in cluster_assignments.items() if cid in {a, b}]
    recently_merged_points = set(points_to_update)

    inherited_color = cluster_to_color.get(a, cluster_to_color.get(b, (0.5, 0.5, 0.5)))
    cluster_to_color[new_cluster_id] = inherited_color

    for idx in points_to_update:
        cluster_assignments[idx] = new_cluster_id

    cluster_to_color.pop(a, None)
    cluster_to_color.pop(b, None)

    base_colors = [cluster_to_color[cluster_assignments[i]] for i in range(n)]

    for fade_frame in range(fade_frames):
        alpha = 1 - (fade_frame / fade_frames)
        mixed_colors = []

        for i in range(n):
            if i in recently_merged_points:
                r, g, b = base_colors[i][:3]
                mixed = tuple(alpha * np.array([r, g, b]) + (1 - alpha) * np.array([1, 1, 1]))
                mixed_colors.append(mixed)
            else:
                mixed_colors.append(base_colors[i][:3])

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Left: Clustering
        axs[0].scatter(X[:, 0], X[:, 1], c=mixed_colors, s=80, edgecolor='k', linewidth=0.7)
        axs[0].set_title(f"Agglomerative Clustering\nStep {step}/{len(Z)}", fontsize=14)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_aspect('equal')

        if legend_on and step == len(Z):
            # Legend showing final cluster colors
            unique_clusters = sorted(set(cluster_assignments.values()))
            legend_colors = [cluster_to_color[cid] for cid in unique_clusters]
            for idx, cid in enumerate(unique_clusters):
                axs[0].scatter([], [], c=[legend_colors[idx]], label=f"Cluster {idx}", s=60)
            axs[0].legend(loc="upper left", fontsize=8, frameon=True)

        # Right: dendrogram with custom color sync + merge line
        axs[1].set_title("Dendrogram (Building)", fontsize=14)
        link_color_func = make_link_color_func(cluster_to_color, step)

        dendrogram(
            Z,
            ax=axs[1],
            no_labels=True,
            link_color_func=link_color_func
        )

        axs[1].axhline(y=Z[step - 1, 2], color='gray', linestyle='--', linewidth=1)
        axs[1].tick_params(left=False, labelleft=False)
        axs[1].set_xlabel("Points")
        axs[1].set_ylabel("Distance")


        plt.tight_layout()
        fname = f"frames/frame_{frame_idx:03d}.png"
        plt.savefig(fname)
        plt.close()
        frame_idx += 1

# Final pause frames to let it linger
for _ in range(final_pause_frames):
    shutil.copyfile(f"frames/frame_{frame_idx - 1:03d}.png",
                    f"frames/frame_{frame_idx:03d}.png")
    frame_idx += 1

# ---------------------
# EXPORT GIF
# ---------------------
with imageio.get_writer(gif_name, mode="I", duration=0.15) as writer:
    for i in range(frame_idx):
        image = imageio.v2.imread(f"frames/frame_{i:03d}.png")
        writer.append_data(image)


# ---------------------
# CLEANUP
# ---------------------
shutil.rmtree("frames")
print(f"GIF saved as '{gif_name}'")
