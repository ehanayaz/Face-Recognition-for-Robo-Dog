# Project-II: PCA-Based Face Recognition

PCA Eigenfaces pipeline with a small **FastAPI web UI** for enrolling family members, managing profiles, and live recognition—suited for a pet robot or home prototype.

## Features

- **Web GUI** (`run_gui.py`): add owner/members from photos or zip, review detected faces, add more photos to existing people, tune PCA settings, live preview.
- **Live**: choose **this device’s camera** (browser) or **the PC’s camera** (MJPEG stream). Frames are analyzed on the machine running the server.
- **CLI**: `train.py` and `detect_live.py` for training from disk and OpenCV-only live demo.

## Requirements

- Python 3.10+ recommended  
- Webcam optional (for Live); enrollment uses uploaded images only.

## Setup

```bash
cd Auto-P2
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the web app

```bash
python run_gui.py
```

Open **http://127.0.0.1:8000** (default bind is localhost only).

### Phone or tablet on the same Wi‑Fi

Listen on all interfaces and use your PC’s LAN IP in the browser:

```bash
PCA_FACE_HOST=0.0.0.0 python run_gui.py
```

Then open `http://<your-pc-ip>:8000` on the phone. Allow the port in your firewall if needed.

**Camera note:** many mobile browsers only allow **HTTPS** (or localhost) for camera access; plain `http://192.168.x.x` may block **This device** mode. Use **Server PC camera** mode or HTTPS if that happens.

## Data layout (local, not in git)

Paths are under `data/` (see `.gitignore`):

| Path | Contents |
|------|-----------|
| `data/dataset/<class>/` | Face crops per person (`owner_*`, `member_*`) |
| `data/model/` | Saved PCA model (`pca_model.npz`, `meta.json`) |
| `data/app_settings.json` | `k`, threshold, camera index, crop options |

Re-enrolling or saving new crops retrains the model automatically from `data/dataset/`.

## CLI usage

Train from a folder tree `root/<class_name>/*.jpg`:

```bash
python train.py --data /path/to/root --out model_out --k 50 --no-show
```

Live recognition with OpenCV window (uses saved model path):

```bash
python detect_live.py --model model_out --threshold 2500
```

For the web app, point training output at `data/model` if you train offline, or rely on the UI to train after enrollment.

## Stack

NumPy, OpenCV (Haar frontal faces), FastAPI, Uvicorn, Jinja2, Matplotlib (training plots).

## Security

Binding `0.0.0.0` exposes the UI on your LAN with **no built-in login**. Use only on networks you trust, or add authentication / HTTPS before wider exposure.
