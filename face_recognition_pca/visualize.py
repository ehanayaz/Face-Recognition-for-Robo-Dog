"""Plot mean face and top-K eigenfaces with Matplotlib."""

from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from face_recognition_pca.learning_module import PCAFaceModel


def _to_image(vec: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    w, h = shape
    return vec.reshape(h, w)


def plot_mean_and_eigenfaces(
    model: PCAFaceModel,
    save_path: str | None = None,
    max_display: int = 16,
    show: bool = True,
) -> None:
    """Show mean image and up to max_display eigenfaces in a grid."""
    shape = model.image_shape
    k_show = min(max_display, model.k)

    cols = 4
    rows = 1 + (k_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.atleast_2d(axes)

    mean_img = _to_image(model.mean, shape)
    axes[0, 0].imshow(mean_img, cmap="gray")
    axes[0, 0].set_title("Mean face")
    axes[0, 0].axis("off")
    for j in range(1, cols):
        axes[0, j].axis("off")

    idx = 0
    for r in range(1, rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx < k_show:
                ef = model.components[:, idx]
                # Normalize for display
                disp = ef - ef.min()
                if disp.max() > 0:
                    disp = disp / disp.max()
                ax.imshow(_to_image(disp, shape), cmap="gray")
                ax.set_title(f"Eigenface {idx + 1}")
            ax.axis("off")
            idx += 1

    plt.tight_layout()
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_class_centroids_2d(
    model: PCAFaceModel,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """If K>=2, scatter first two PCA coordinates of class centroids."""
    if model.k < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    c = model.centroids
    ax.scatter(c[:, 0], c[:, 1], s=80)
    for i, name in enumerate(model.label_names):
        ax.annotate(name, (c[i, 0], c[i, 1]), fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Class centroids (first 2 PCs)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
