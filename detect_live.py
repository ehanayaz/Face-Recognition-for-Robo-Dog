#!/usr/bin/env python3
"""
Live webcam recognition: detect faces, project to PCA space, Euclidean distance to centroids.
Distance > threshold => Unknown.
"""

from __future__ import annotations

import argparse

import cv2

from face_recognition_pca.config import DEFAULT_DISTANCE_THRESHOLD
from face_recognition_pca.detection_module import _load_face_cascade, open_camera, run_frame
from face_recognition_pca.learning_module import PCAFaceModel


def draw_results(frame, results: list) -> None:
    for r in results:
        x, y, w, h = r.face_bbox
        color = (0, 0, 255) if r.is_unknown else (0, 180, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{r.role} | {r.identity_folder} | d={r.distance:.1f}"
        cv2.putText(
            frame,
            text,
            (x, max(y - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Live PCA face recognition")
    ap.add_argument("--model", default="model_out", help="Directory with trained model")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Max Euclidean distance in PCA space to accept match (else Unknown)",
    )
    ap.add_argument("--camera", type=int, default=0, help="Camera index")
    args = ap.parse_args()

    model = PCAFaceModel.load(args.model)
    thr = args.threshold
    if thr is None and model.suggested_threshold is not None:
        thr = model.suggested_threshold
    if thr is None:
        thr = DEFAULT_DISTANCE_THRESHOLD

    print(f"Using K={model.k}, distance threshold={thr:.2f}")

    cascade = _load_face_cascade()
    cap = open_camera(args.camera)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = run_frame(model, frame, cascade, distance_threshold=thr)
            draw_results(frame, results)
            cv2.imshow("PCA Face Recognition — q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
