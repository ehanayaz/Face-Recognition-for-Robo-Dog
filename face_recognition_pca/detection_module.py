"""
Live capture, face detection, PCA projection, nearest-centroid matching with unknown rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from face_recognition_pca.config import (
    DEFAULT_DISTANCE_THRESHOLD,
    FACE_DETECTOR_MIN_NEIGHBORS,
    FACE_DETECTOR_SCALE_FACTOR,
    FACE_MIN_SIZE,
    IMAGE_SIZE,
)
from face_recognition_pca.learning_module import PCAFaceModel


@dataclass
class RecognitionResult:
    identity_folder: str
    role: str
    distance: float
    is_unknown: bool
    face_bbox: Tuple[int, int, int, int]


def _load_face_cascade() -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {path}")
    return cascade


def detect_faces_bgr(
    frame_bgr: np.ndarray,
    cascade: cv2.CascadeClassifier,
) -> list:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_DETECTOR_SCALE_FACTOR,
        minNeighbors=FACE_DETECTOR_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE,
    )
    return faces


def face_patch_to_vector(
    frame_bgr: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Crop face region, grayscale, resize, flatten."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]
    x2 = min(x + w, w_img)
    y2 = min(y + h, h_img)
    x = max(0, x)
    y = max(0, y)
    crop = gray[y:y2, x:x2]
    if crop.size == 0:
        return np.zeros(size[0] * size[1], dtype=np.float64)
    resized = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float64).reshape(-1)


def recognize_face(
    model: PCAFaceModel,
    face_vector: np.ndarray,
    distance_threshold: float,
) -> Tuple[str, str, float, bool]:
    """
    Returns identity_folder_name, display_role, distance, is_unknown.
    """
    label, dist, _ = model.predict_label_and_distance(face_vector)
    role = model.roles.get(label, "Family")
    unknown = dist > distance_threshold
    if unknown:
        return "Unknown", "Unknown", dist, True
    return label, role, dist, False


def run_frame(
    model: PCAFaceModel,
    frame_bgr: np.ndarray,
    cascade: cv2.CascadeClassifier,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> list:
    """Process one BGR frame; return list of RecognitionResult."""
    results: list = []
    for (x, y, w, h) in detect_faces_bgr(frame_bgr, cascade):
        vec = face_patch_to_vector(frame_bgr, x, y, w, h, model.image_shape)
        ident, role, dist, unk = recognize_face(model, vec, distance_threshold)
        results.append(
            RecognitionResult(
                identity_folder=ident,
                role=role,
                distance=dist,
                is_unknown=unk,
                face_bbox=(x, y, w, h),
            )
        )
    return results


def open_camera(camera_index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")
    return cap
