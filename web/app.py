"""
FastAPI web GUI: enrollment, members, settings, MJPEG live stream.
"""

from __future__ import annotations

import io
import re
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from face_recognition_pca.detection_module import (
    _load_face_cascade,
    detect_faces_bgr,
    run_frame,
)
from face_recognition_pca.input_module import folder_role
from face_recognition_pca.learning_module import PCAFaceModel

from web.crop_utils import crop_face_from_detection
from web.paths import (
    DATASET_DIR,
    MODEL_DIR,
    PROJECT_ROOT,
    TEMP_ENROLL_DIR,
    effective_threshold,
    ensure_data_dirs,
    load_settings,
    save_settings,
)

TEMPLATES_DIR = PROJECT_ROOT / "web" / "templates"
STATIC_DIR = PROJECT_ROOT / "web" / "static"

app = FastAPI(title="PCA Face Recognition")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_pending_lock = threading.Lock()
_pending_enroll: dict[str, dict[str, Any]] = {}

_model_lock = threading.Lock()
_model_cache: PCAFaceModel | None = None
_model_mtime: float = 0.0

_cascade_instance: Any = None
_cascade_lock = threading.Lock()


def _get_cascade():
    global _cascade_instance
    with _cascade_lock:
        if _cascade_instance is None:
            _cascade_instance = _load_face_cascade()
        return _cascade_instance


def invalidate_model_cache() -> None:
    global _model_cache, _model_mtime
    with _model_lock:
        _model_cache = None
        _model_mtime = 0.0


def get_model() -> PCAFaceModel | None:
    global _model_cache, _model_mtime
    npz = MODEL_DIR / "pca_model.npz"
    if not npz.is_file():
        return None
    mtime = npz.stat().st_mtime
    with _model_lock:
        if _model_cache is None or mtime != _model_mtime:
            _model_cache = PCAFaceModel.load(str(MODEL_DIR))
            _model_mtime = mtime
        return _model_cache


def make_class_folder(role: str, display_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", display_name.strip()).strip("_")
    if not slug:
        slug = "person"
    prefix = "owner" if role == "owner" else "member"
    return f"{prefix}_{slug}"


def _safe_member_folder(folder: str) -> Path | None:
    """Return resolved dataset subfolder if it exists and stays under DATASET_DIR."""
    if not folder or ".." in folder or "/" in folder or "\\" in folder:
        return None
    path = (DATASET_DIR / folder).resolve()
    try:
        path.relative_to(DATASET_DIR.resolve())
    except ValueError:
        return None
    if not path.is_dir():
        return None
    return path


def _extract_faces_from_image(
    img: np.ndarray,
    enroll_dir: Path,
    start_idx: int,
    cascade,
    margin: float,
    square: bool,
    crops_out: list[str],
) -> int:
    faces = detect_faces_bgr(img, cascade)
    idx = start_idx
    for (x, y, w, h) in faces:
        crop = crop_face_from_detection(img, (x, y, w, h), margin, square)
        if crop.size == 0:
            continue
        path = enroll_dir / f"crop_{idx:04d}.png"
        cv2.imwrite(str(path), crop)
        crops_out.append(path.name)
        idx += 1
    return idx


async def process_uploads_to_crops(
    enroll_id: str,
    file_list: list[Any],
    margin: float,
    square: bool,
) -> list[str]:
    cascade = _get_cascade()
    enroll_dir = TEMP_ENROLL_DIR / enroll_id
    enroll_dir.mkdir(parents=True, exist_ok=True)
    crops: list[str] = []
    idx = 0
    image_ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    for uf in file_list:
        if not getattr(uf, "filename", None):
            continue
        raw = await uf.read()
        name = (getattr(uf, "filename", "") or "").lower()
        if name.endswith(".zip"):
            try:
                zf = zipfile.ZipFile(io.BytesIO(raw))
            except zipfile.BadZipFile:
                continue
            with zf:
                for zname in zf.namelist():
                    if zname.endswith("/") or zname.startswith("__"):
                        continue
                    zl = zname.lower()
                    if not zl.endswith(image_ext):
                        continue
                    try:
                        data = zf.read(zname)
                    except KeyError:
                        continue
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    idx = _extract_faces_from_image(
                        img, enroll_dir, idx, cascade, margin, square, crops
                    )
        else:
            if not name.endswith(image_ext):
                continue
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            idx = _extract_faces_from_image(
                img, enroll_dir, idx, cascade, margin, square, crops
            )
    return crops


def retrain_model() -> tuple[bool, str | None]:
    from face_recognition_pca.input_module import build_dataset
    from face_recognition_pca.learning_module import train_pca_model

    ensure_data_dirs()

    def _clear_model_dir() -> None:
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        X, y, label_names = build_dataset(str(DATASET_DIR))
    except (ValueError, FileNotFoundError):
        _clear_model_dir()
        invalidate_model_cache()
        return True, None

    settings = load_settings()
    k = int(settings["k"])
    try:
        model = train_pca_model(X, y, label_names, k=k)
    except Exception as e:
        return False, str(e)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR))
    invalidate_model_cache()
    return True, None


def draw_results(frame: np.ndarray, results: list) -> None:
    for r in results:
        x, y, w, h = r.face_bbox
        color = (60, 60, 200) if r.is_unknown else (100, 140, 90)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{r.role} | {r.identity_folder} | d={r.distance:.0f}"
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


def _jpeg_bytes(frame_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return buf.tobytes()


def _placeholder_frame(message: str) -> bytes:
    img = np.full((480, 640, 3), (248, 245, 240), dtype=np.uint8)
    cv2.putText(
        img,
        message[:80],
        (40, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (42, 42, 42),
        1,
        cv2.LINE_AA,
    )
    return _jpeg_bytes(img)


def mjpeg_generator():
    settings = load_settings()
    cam_idx = int(settings["camera_index"])
    cap = cv2.VideoCapture(cam_idx)
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    try:
        if not cap.isOpened():
            while True:
                yield boundary + _placeholder_frame("Camera unavailable") + b"\r\n"
                time.sleep(0.5)
            return
        cascade = _get_cascade()
        while True:
            ok, frame = cap.read()
            if not ok:
                yield boundary + _placeholder_frame("Frame capture failed") + b"\r\n"
                time.sleep(0.1)
                continue
            model = get_model()
            thr_effective = effective_threshold(
                settings,
                model.suggested_threshold if model else None,
            )
            if model:
                results = run_frame(model, frame, cascade, thr_effective)
                draw_results(frame, results)
            else:
                cv2.putText(
                    frame,
                    "No model — add members and train",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (80, 80, 120),
                    1,
                    cv2.LINE_AA,
                )
            jpeg = _jpeg_bytes(frame)
            yield boundary + jpeg + b"\r\n"
    finally:
        cap.release()


@app.on_event("startup")
def _startup() -> None:
    ensure_data_dirs()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    model_ok = (MODEL_DIR / "pca_model.npz").is_file()
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"request": request, "title": "Home", "model_ok": model_ok},
    )


@app.get("/enroll", response_class=HTMLResponse)
async def enroll_get(request: Request, role: str = "owner"):
    if role not in ("owner", "member"):
        role = "owner"
    return templates.TemplateResponse(
        request,
        "enroll.html",
        {"request": request, "title": "Add person", "role": role, "error": None},
    )


@app.post("/enroll", response_class=HTMLResponse)
async def enroll_post(request: Request):
    form = await request.form()
    role = form.get("role") or "owner"
    display_name = (form.get("display_name") or "").strip()
    if role not in ("owner", "member"):
        role = "owner"
    if not display_name:
        return templates.TemplateResponse(
            request,
            "enroll.html",
            {
                "request": request,
                "title": "Add person",
                "role": role,
                "error": "Please enter a name.",
            },
            status_code=400,
        )

    raw_files = form.getlist("files")
    files = [f for f in raw_files if getattr(f, "filename", None)]
    if not files:
        return templates.TemplateResponse(
            request,
            "enroll.html",
            {
                "request": request,
                "title": "Add person",
                "role": role,
                "error": "Please choose at least one image or a .zip file.",
            },
            status_code=400,
        )

    settings = load_settings()
    margin = float(settings.get("crop_margin", 0.25))
    square = bool(settings.get("square_crop", False))

    enroll_id = uuid.uuid4().hex
    class_folder = make_class_folder(str(role), display_name)

    crops = await process_uploads_to_crops(enroll_id, files, margin, square)
    if not crops:
        shutil.rmtree(TEMP_ENROLL_DIR / enroll_id, ignore_errors=True)
        return templates.TemplateResponse(
            request,
            "enroll.html",
            {
                "request": request,
                "title": "Add person",
                "role": role,
                "error": "No faces detected in the uploaded files. Try clearer photos.",
            },
            status_code=400,
        )

    with _pending_lock:
        _pending_enroll[enroll_id] = {
            "class_folder": class_folder,
            "crop_names": crops,
            "created": time.time(),
            "extend": False,
        }

    return RedirectResponse(f"/enroll/review/{enroll_id}", status_code=303)


@app.get("/members/{folder}/photos", response_class=HTMLResponse)
async def add_photos_get(request: Request, folder: str):
    if _safe_member_folder(folder) is None:
        raise HTTPException(404, "Member not found.")
    return templates.TemplateResponse(
        request,
        "add_photos.html",
        {
            "request": request,
            "title": "Add photos",
            "folder": folder,
            "error": None,
        },
    )


@app.post("/members/{folder}/photos", response_class=HTMLResponse)
async def add_photos_post(request: Request, folder: str):
    member_path = _safe_member_folder(folder)
    if member_path is None:
        raise HTTPException(404, "Member not found.")

    form = await request.form()
    raw_files = form.getlist("files")
    files = [f for f in raw_files if getattr(f, "filename", None)]
    if not files:
        return templates.TemplateResponse(
            request,
            "add_photos.html",
            {
                "request": request,
                "title": "Add photos",
                "folder": folder,
                "error": "Please choose at least one image or a .zip file.",
            },
            status_code=400,
        )

    settings = load_settings()
    margin = float(settings.get("crop_margin", 0.25))
    square = bool(settings.get("square_crop", False))

    enroll_id = uuid.uuid4().hex
    crops = await process_uploads_to_crops(enroll_id, files, margin, square)
    if not crops:
        shutil.rmtree(TEMP_ENROLL_DIR / enroll_id, ignore_errors=True)
        return templates.TemplateResponse(
            request,
            "add_photos.html",
            {
                "request": request,
                "title": "Add photos",
                "folder": folder,
                "error": "No faces detected. Try clearer front-facing photos.",
            },
            status_code=400,
        )

    with _pending_lock:
        _pending_enroll[enroll_id] = {
            "class_folder": folder,
            "crop_names": crops,
            "created": time.time(),
            "extend": True,
        }

    return RedirectResponse(f"/enroll/review/{enroll_id}", status_code=303)


@app.get("/enroll/review/{enroll_id}", response_class=HTMLResponse)
async def enroll_review(request: Request, enroll_id: str):
    with _pending_lock:
        data = _pending_enroll.get(enroll_id)
    if not data:
        raise HTTPException(404, "Review session expired or invalid.")
    crops = data["crop_names"]
    extend = bool(data.get("extend", False))
    cancel_href = "/members" if extend else "/enroll?role=owner"
    return templates.TemplateResponse(
        request,
        "enroll_review.html",
        {
            "request": request,
            "title": "Review faces",
            "enroll_id": enroll_id,
            "class_folder": data["class_folder"],
            "crops": crops,
            "extend": extend,
            "cancel_href": cancel_href,
        },
    )


@app.get("/enroll/preview/{enroll_id}/{crop_name}")
async def enroll_preview(enroll_id: str, crop_name: str):
    if not re.match(r"^crop_\d{4}\.png$", crop_name):
        raise HTTPException(404)
    path = TEMP_ENROLL_DIR / enroll_id / crop_name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


@app.post("/enroll/confirm")
async def enroll_confirm(request: Request):
    form = await request.form()
    enroll_id = form.get("enroll_id")
    if not enroll_id or not isinstance(enroll_id, str):
        raise HTTPException(400)

    with _pending_lock:
        data = _pending_enroll.pop(enroll_id, None)
    if not data:
        raise HTTPException(404, "Session expired.")

    keep_vals = form.getlist("keep")
    keep_set = {int(x) for x in keep_vals if str(x).isdigit()}

    class_folder = data["class_folder"]
    crop_names = data["crop_names"]
    dest_dir = DATASET_DIR / class_folder
    dest_dir.mkdir(parents=True, exist_ok=True)

    counter = int(time.time()) % 100000
    for i, name in enumerate(crop_names):
        if i not in keep_set:
            continue
        src = TEMP_ENROLL_DIR / enroll_id / name
        if not src.is_file():
            continue
        counter += 1
        dest = dest_dir / f"face_{counter:06d}.png"
        shutil.copy2(src, dest)

    shutil.rmtree(TEMP_ENROLL_DIR / enroll_id, ignore_errors=True)

    ok, err = retrain_model()
    if not ok:
        return templates.TemplateResponse(
            request,
            "message.html",
            {
                "request": request,
                "title": "Training issue",
                "message": err or "Training failed.",
                "level": "warn",
            },
            status_code=500,
        )

    return RedirectResponse("/members?saved=1", status_code=303)


@app.get("/members", response_class=HTMLResponse)
async def members_page(request: Request):
    ensure_data_dirs()
    rows = []
    for p in sorted(DATASET_DIR.iterdir()):
        if not p.is_dir():
            continue
        n = sum(
            1
            for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        )
        rows.append(
            {
                "folder": p.name,
                "role": folder_role(p.name),
                "count": n,
            }
        )
    saved = request.query_params.get("saved")
    return templates.TemplateResponse(
        request,
        "members.html",
        {"request": request, "title": "Members", "members": rows, "saved": saved},
    )


@app.post("/members/delete/{folder}")
async def member_delete(folder: str):
    if ".." in folder or "/" in folder or "\\" in folder:
        raise HTTPException(400)
    path = (DATASET_DIR / folder).resolve()
    try:
        path.relative_to(DATASET_DIR.resolve())
    except ValueError:
        raise HTTPException(400)
    if path.is_dir():
        shutil.rmtree(path)
    invalidate_model_cache()
    retrain_model()
    return RedirectResponse("/members", status_code=303)


@app.get("/settings", response_class=HTMLResponse)
async def settings_get(request: Request):
    s = load_settings()
    model = get_model()
    suggested = model.suggested_threshold if model else None
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "request": request,
            "title": "Settings",
            "settings": s,
            "suggested_threshold": suggested,
        },
    )


@app.post("/settings")
async def settings_post(request: Request):
    form = await request.form()
    s = load_settings()
    try:
        s["k"] = max(1, int(form.get("k") or s["k"]))
    except (TypeError, ValueError):
        pass
    try:
        s["camera_index"] = max(0, int(form.get("camera_index") or 0))
    except (TypeError, ValueError):
        pass
    thr_raw = form.get("distance_threshold")
    if thr_raw is None or str(thr_raw).strip() == "":
        s["distance_threshold"] = None
    else:
        try:
            s["distance_threshold"] = float(thr_raw)
        except ValueError:
            pass
    try:
        s["crop_margin"] = float(form.get("crop_margin") or s["crop_margin"])
    except (TypeError, ValueError):
        pass
    s["square_crop"] = form.get("square_crop") == "on"
    save_settings(s)
    invalidate_model_cache()
    return RedirectResponse("/settings?ok=1", status_code=303)


@app.get("/live", response_class=HTMLResponse)
async def live_page(request: Request):
    model_ok = (MODEL_DIR / "pca_model.npz").is_file()
    return templates.TemplateResponse(
        request,
        "live.html",
        {"request": request, "title": "Live", "model_ok": model_ok},
    )


@app.post("/api/analyze_frame")
async def analyze_frame(file: UploadFile = File(...)):
    """
    Run face detection + PCA on one JPEG/PNG frame from the browser camera.
    Coordinates match the uploaded image pixel dimensions.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty body")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    h0, w0 = img.shape[:2]

    model = get_model()
    settings = load_settings()
    thr = effective_threshold(
        settings,
        model.suggested_threshold if model else None,
    )

    if model is None:
        return JSONResponse(
            {
                "faces": [],
                "width": w0,
                "height": h0,
                "no_model": True,
            }
        )

    cascade = _get_cascade()
    results = run_frame(model, img, cascade, thr)
    faces = []
    for r in results:
        x, y, w, h = r.face_bbox
        faces.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "identity": r.identity_folder,
                "role": r.role,
                "distance": float(r.distance),
                "unknown": r.is_unknown,
            }
        )
    return JSONResponse(
        {"faces": faces, "width": w0, "height": h0, "no_model": False}
    )


@app.get("/stream")
async def stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
