"""
PCA (Eigenfaces): mean-centering, SVD-based principal components, projection model.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from face_recognition_pca.config import IMAGE_SIZE
from face_recognition_pca.input_module import folder_role


@dataclass
class PCAFaceModel:
    """
    Stores mean face, top-K eigenfaces (columns of `components`), and class centroids in PCA space.
    """

    mean: np.ndarray  # (d,)
    components: np.ndarray  # (d, K) — columns are eigenfaces
    k: int
    label_names: List[str]
    centroids: np.ndarray  # (C, K) — one row per class
    image_shape: Tuple[int, int] = IMAGE_SIZE
    roles: Dict[str, str] = field(default_factory=dict)
    suggested_threshold: Optional[float] = None

    def project(self, face_row: np.ndarray) -> np.ndarray:
        """Project one or more faces (N, d) or (d,) to K-dimensional PCA coordinates."""
        single = face_row.ndim == 1
        x = face_row.reshape(1, -1) if single else face_row
        centered = x - self.mean
        coeffs = centered @ self.components
        return coeffs.ravel() if single else coeffs

    def predict_label_and_distance(
        self, face_row: np.ndarray
    ) -> Tuple[str, float, float]:
        """
        Nearest centroid in PCA space (Euclidean).

        Returns:
            predicted_label, min_distance, second_min_distance (for margin diagnostics).
        """
        z = self.project(face_row)
        dmat = np.linalg.norm(self.centroids - z, axis=1)
        order = np.argsort(dmat)
        best = order[0]
        min_d = float(dmat[best])
        second = float(dmat[order[1]]) if len(order) > 1 else float("inf")
        return self.label_names[best], min_d, second

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        np.savez(
            os.path.join(directory, "pca_model.npz"),
            mean=self.mean,
            components=self.components,
            k=self.k,
            centroids=self.centroids,
            image_height=self.image_shape[1],
            image_width=self.image_shape[0],
        )
        meta = {
            "label_names": self.label_names,
            "roles": self.roles,
            "suggested_threshold": self.suggested_threshold,
        }
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "PCAFaceModel":
        path = os.path.join(directory, "pca_model.npz")
        data = np.load(path)
        mean = data["mean"]
        components = data["components"]
        k = int(data["k"])
        centroids = data["centroids"]
        w = int(data["image_width"])
        h = int(data["image_height"])
        meta_path = os.path.join(directory, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            label_names = meta["label_names"]
            roles = meta.get("roles", {})
            suggested_threshold = meta.get("suggested_threshold")
        else:
            label_names = [str(i) for i in range(len(centroids))]
            roles = {}
            suggested_threshold = None
        return cls(
            mean=mean,
            components=components,
            k=k,
            label_names=label_names,
            centroids=centroids,
            image_shape=(w, h),
            roles=roles,
            suggested_threshold=suggested_threshold,
        )


def train_pca_model(
    X: np.ndarray,
    y: np.ndarray,
    label_names: List[str],
    k: int,
    image_shape: Tuple[int, int] = IMAGE_SIZE,
) -> PCAFaceModel:
    """
    Train Eigenfaces model.

    X: (N, d) training faces (rows).
    y: (N,) integer labels.
    k: number of principal components (must be <= min(N, d)).
    """
    n, d = X.shape
    k = min(k, n, d)
    mean = X.mean(axis=0)
    centered = X - mean

    # SVD on (N x d): centered = U S Vt; rows of Vt are principal axes in sklearn convention
    # Here we want (d, K) matrix whose columns span PCA subspace: use right singular vectors.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    # vh shape (min(N,d), d) — rows of vh are eigenvectors of covariance in feature space
    components = vh[:k].T  # (d, K)

    # Project training set
    proj = centered @ components  # (N, K)

    centroids = np.zeros((len(label_names), k))
    intra_max: List[float] = []
    for c in range(len(label_names)):
        mask = y == c
        if not np.any(mask):
            continue
        z_c = proj[mask]
        centroids[c] = z_c.mean(axis=0)
        dists = np.linalg.norm(z_c - centroids[c], axis=1)
        intra_max.append(float(np.max(dists)) if len(dists) else 0.0)

    roles = {lbl: folder_role(lbl) for lbl in label_names}
    suggested = float(max(intra_max) * 1.5) if intra_max else None

    return PCAFaceModel(
        mean=mean,
        components=components,
        k=k,
        label_names=label_names,
        centroids=centroids,
        image_shape=image_shape,
        roles=roles,
        suggested_threshold=suggested,
    )
