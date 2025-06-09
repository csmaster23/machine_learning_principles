import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load real dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Meshgrid for decision boundary plotting
h = 0.05
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Output directory for frames
frame_dir = "svm_frames"
os.makedirs(frame_dir, exist_ok=True)

# Generate frames with varying C values
frames = []
C_values = np.logspace(-2, 2, 20)
for C in C_values:
    clf = SVC(kernel='rbf', C=C, gamma='scale')
    clf.fit(X_train_pca, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(f"SVM Decision Boundary (PCA)\nC = {C:.2f}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()

    # Save and load frame
    fname = os.path.join(frame_dir, f"frame_C_{C:.2f}.png")
    plt.savefig(fname)
    frames.append(imageio.imread(fname))
    plt.close()

# Create GIF
gif_path = "svm_pca_decision_boundary.gif"
imageio.mimsave(gif_path, frames, fps=3)

print(f"GIF saved as: {gif_path}")

