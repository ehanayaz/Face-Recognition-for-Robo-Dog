"""Expand face bounding boxes and crop BGR regions for enrollment."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def expand_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    margin: float,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """Expand bbox by margin fraction of max(w,h), clamp to image."""
    m = int(max(w, h) * margin)
    x0 = max(0, x - m)
    y0 = max(0, y - m)
    x1 = min(img_w, x + w + m)
    y1 = min(img_h, y + h + m)
    return x0, y0, x1 - x0, y1 - y0


def crop_square_bgr(
    frame_bgr: np.ndarray, x: int, y: int, w: int, h: int
) -> np.ndarray:
    """Crop region and pad to square (letterbox with border replicate)."""
    H, W = frame_bgr.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    patch = frame_bgr[y:y2, x:x2].copy()
    ph, pw = patch.shape[:2]
    side = max(ph, pw)
    out = np.zeros((side, side, 3), dtype=patch.dtype)
    oy = (side - ph) // 2
    ox = (side - pw) // 2
    out[oy : oy + ph, ox : ox + pw] = patch
    return out


def crop_face_from_detection(
    frame_bgr: np.ndarray,
    bbox_xywh: Tuple[int, int, int, int],
    margin: float = 0.25,
    square: bool = False,
) -> np.ndarray:
    x, y, w, h = bbox_xywh
    img_h, img_w = frame_bgr.shape[:2]
    x, y, w, h = expand_bbox(x, y, w, h, margin, img_w, img_h)
    if square:
        return crop_square_bgr(frame_bgr, x, y, w, h)
    x2 = min(x + w, img_w)
    y2 = min(y + h, img_h)
    return frame_bgr[y:y2, x:x2].copy()
