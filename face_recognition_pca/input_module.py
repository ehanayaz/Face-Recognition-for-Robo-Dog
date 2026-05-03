"""
Load face images from a directory tree: root/<identity_folder>/*.jpg|png|...
Each subfolder name is the class label (e.g. owner_Alice, member_Bob).
"""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np

from face_recognition_pca.config import IMAGE_SIZE


def _list_image_paths(root: str) -> List[Tuple[str, str]]:
    """Return list of (absolute_path, class_label) for all images under root."""
    pairs: List[Tuple[str, str]] = []
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root directory not found: {root}")

    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub):
            continue
        label = name
        for fn in os.listdir(sub):
            lower = fn.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                pairs.append((os.path.join(sub, fn), label))
    if not pairs:
        raise ValueError(
            f"No images found under {root}. Expected subfolders with images inside."
        )
    return pairs


def load_image_as_vector(path: str, size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Load one image: BGR read -> grayscale -> resize -> flatten to float64 column vector.
    Shape (d,) where d = width * height.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    vec = resized.astype(np.float64).reshape(-1)
    return vec


def build_dataset(
    root: str,
    size: Tuple[int, int] = IMAGE_SIZE,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build data matrix X and label vector y.

    X: shape (N, d) — each row is one flattened face.
    y: shape (N,) — integer labels 0..C-1.
    label_names: ordered list of class names (folder names).

    Returns:
        X, y, label_names
    """
    pairs = _list_image_paths(root)
    labels_sorted = sorted({lbl for _, lbl in pairs})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels_sorted)}

    vectors: List[np.ndarray] = []
    y_list: List[int] = []
    for path, lbl in pairs:
        try:
            v = load_image_as_vector(path, size)
        except ValueError:
            continue
        vectors.append(v)
        y_list.append(label_to_idx[lbl])

    if not vectors:
        raise ValueError("No valid images could be loaded.")

    X = np.stack(vectors, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, labels_sorted


def folder_role(folder_name: str) -> str:
    """
    Map folder name to display role: Owner, Family, or Other (custom label).
    Convention: name starts with 'owner' (case-insensitive) -> Owner;
    contains 'member' or starts with 'member_' -> Family; else use folder name as role hint.
    """
    low = folder_name.lower()
    if low.startswith("owner"):
        return "Owner"
    if "member" in low or low.startswith("family"):
        return "Family"
    return "Family"
