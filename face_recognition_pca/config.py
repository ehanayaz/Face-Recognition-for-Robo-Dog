"""Default hyperparameters and paths."""

IMAGE_SIZE = (100, 100)  # width, height
DEFAULT_K = 50
# Distance above this (in PCA space) => Unknown (tune on validation data)
DEFAULT_DISTANCE_THRESHOLD = 3500.0
FACE_DETECTOR_SCALE_FACTOR = 1.1
FACE_DETECTOR_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (60, 60)
