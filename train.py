#!/usr/bin/env python3
"""
Train PCA Eigenfaces model from a dataset root:
  data_root/owner_Name/*.jpg
  data_root/member_1_Name/*.jpg
  ...
"""

from __future__ import annotations

import argparse
import os

def main() -> None:
    p = argparse.ArgumentParser(description="Train PCA face model (Eigenfaces)")
    p.add_argument("--data", required=True, help="Root folder containing class subfolders")
    p.add_argument("--out", default="model_out", help="Directory to save model and plots")
    p.add_argument("--k", type=int, default=None, help="Number of eigenfaces (components)")
    p.add_argument("--no-show", action="store_true", help="Save plots only, do not open windows")
    args = p.parse_args()

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    from face_recognition_pca.config import DEFAULT_K
    from face_recognition_pca.input_module import build_dataset
    from face_recognition_pca.learning_module import train_pca_model
    from face_recognition_pca.visualize import plot_class_centroids_2d, plot_mean_and_eigenfaces

    k = args.k if args.k is not None else DEFAULT_K

    X, y, label_names = build_dataset(args.data)
    print(f"Loaded {X.shape[0]} images, dim={X.shape[1]}, classes={label_names}")

    model = train_pca_model(X, y, label_names, k=k)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    model.save(out_dir)

    print(f"Saved model to {out_dir}")
    if model.suggested_threshold is not None:
        print(
            f"Suggested distance threshold (coarse): {model.suggested_threshold:.2f} "
            "(tune on validation / live tests)"
        )

    show = not args.no_show
    plot_mean_and_eigenfaces(
        model,
        save_path=os.path.join(out_dir, "eigenfaces.png"),
        show=show,
    )
    if model.k >= 2:
        plot_class_centroids_2d(
            model,
            save_path=os.path.join(out_dir, "centroids_pc1_pc2.png"),
            show=show,
        )


if __name__ == "__main__":
    main()
