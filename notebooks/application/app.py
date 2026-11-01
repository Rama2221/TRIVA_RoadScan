"""Streamlit dashboard for road damage detection and priority reports."""
from __future__ import annotations

import json
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from matplotlib.patches import Rectangle

st.set_page_config(page_title="RoadScan | Road Damage Detection", page_icon="🛣️", layout="wide")

CLASS_NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole",
}
CLASS_SLUGS = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]


def find_project_root() -> Path:
    """Locate the shared project configuration relative to this application."""
    for candidate in [*Path(__file__).resolve().parents, Path.cwd(), *Path.cwd().parents]:
        if (candidate / "configs/project.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError("RoadScan project configuration was not found.")


def find_weights(root: Path) -> Path:
    """Use the detector included in the deployment package."""
    return root / "models/deployment/rf_small.pth"


@st.cache_resource(show_spinner="Loading RF-DETR Small on CPU…")
def load_detector(project_root: str, weights_path: str, resolution: int):
    root = Path(project_root).expanduser().resolve()
    weights = Path(weights_path).expanduser().resolve()
    if not (root / "src/roadscan/predict.py").is_file():
        raise FileNotFoundError(f"Could not find src/roadscan/predict.py in project folder: {root}")
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights were not found: {weights}")
    selection = json.loads((root / "configs/detector_selection.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != selection["weights_sha256"]:
        raise ValueError("The detector does not match the recorded model selection.")
    source_dir = str(root / "src")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    import importlib
    importlib.invalidate_caches()
    predict_module = importlib.import_module("roadscan.predict")
    priority_module = importlib.import_module("roadscan.priority")
    detector = predict_module.RoadScanRF(
        weights=weights,
        classes=CLASS_SLUGS,
        device="cpu",
        resolution=int(resolution),
        conf_floor=0.001,
        max_det=300,
    )
    return detector, priority_module.priority_score, priority_module.PRIORITY_CONFIG


def normalize_and_save(uploaded_file: Any, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    uploaded_file.seek(0)
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        suffix = ".jpg"
    dest = target_dir / f"{uuid.uuid4().hex}{suffix}"
    with Image.open(uploaded_file) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if suffix in {".jpg", ".jpeg"}:
            image.save(dest, quality=95)
        else:
            image.save(dest)
    return dest


def filter_predictions(predictions: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the operating confidence threshold without changing model weights."""
    predictions = np.asarray(predictions, dtype=float).reshape(-1, 6)
    if not len(predictions):
        return predictions
    return predictions[predictions[:, 4] >= float(threshold)]


def draw_detections(image: Image.Image, predictions: np.ndarray):
    """Display each accepted damage region with its class and confidence."""
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(image)
    for x1, y1, x2, y2, confidence, class_id in predictions:
        class_name = CLASS_NAMES.get(int(class_id), f"Class {int(class_id)}")
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2))
        ax.text(
            x1, max(0, y1 - 5), f"{class_name} {confidence:.2f}", fontsize=9,
            bbox={"alpha": 0.75, "pad": 2},
        )
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def make_report(filename: str, predictions: np.ndarray, image: Image.Image,
                priority_fn, priority_config, elapsed: float) -> dict:
    """Build a report from the detector outputs and shared priority policy."""
    priority = priority_fn(predictions, image.width, image.height, priority_config)
    detections = []
    for x1, y1, x2, y2, confidence, class_id in predictions:
        detections.append({
            "class_id": int(class_id),
            "class_name": CLASS_NAMES.get(int(class_id), f"Class {int(class_id)}"),
            "confidence": round(float(confidence), 6),
            "bbox_xyxy": [round(float(v), 2) for v in [x1, y1, x2, y2]],
        })
    return {
        "report_id": uuid.uuid4().hex[:12],
        "file_name": filename,
        "processed_at": datetime.now(ZoneInfo("Asia/Riyadh")).isoformat(timespec="seconds"),
        "detections_count": int(len(predictions)),
        "priority_score": round(float(priority["score"]), 2),
        "priority_band": priority["band"],
        "needs_review": bool(priority["needs_review"]),
        "mean_confidence": round(float(np.mean(predictions[:, 4])), 4) if len(predictions) else None,
        "processing_seconds": round(float(elapsed), 4),
        "detections": detections,
        "priority_details": priority,
    }


st.title("RoadScan")
st.caption("Road damage detection and maintenance priority assessment")

# Keep the dashboard and recorded evaluation on the same deployed checkpoint.
project_root = find_project_root()
selection = json.loads((project_root / "configs/detector_selection.json").read_text(encoding="utf-8"))
project_root_text = str(project_root)
weights_text = str(find_weights(project_root))
resolution = int(selection["rf_resolution"])
state_dir_text = str(project_root / "runtime/dashboard_reports")

with st.sidebar:
    st.header("Detection settings")
    st.caption("RF-DETR Small · CPU")
    confidence_threshold = st.slider(
        "Confidence threshold", min_value=0.05, max_value=0.90,
        value=float(selection["thresholds"][0]), step=0.01,
    )

st.markdown("### Upload road images")
uploads = st.file_uploader(
    "Choose one or more road images",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)
run_button = st.button("Detect road damage", type="primary", disabled=not uploads, width="content")

if run_button and uploads:
    try:
        detector, priority_fn, priority_config = load_detector(project_root_text, weights_text, int(resolution))
    except Exception as exc:
        st.error(f"Could not load RF-DETR: {exc}")
        st.stop()

    state_dir = Path(state_dir_text).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = Path(tempfile.gettempdir()) / "roadscan_streamlit_uploads"
    progress = st.progress(0)
    status = st.empty()
    report_rows = []
    report_json = []

    for index, uploaded in enumerate(uploads, start=1):
        status.write(f"Processing {index}/{len(uploads)}: **{uploaded.name}**")
        image_path = None
        try:
            image_path = normalize_and_save(uploaded, upload_dir)
            with Image.open(image_path) as opened:
                preview = opened.copy().convert("RGB")
            started = time.perf_counter()
            raw_predictions = detector.raw(image_path)
            predictions = filter_predictions(raw_predictions, confidence_threshold)
            elapsed = time.perf_counter() - started
            report = make_report(uploaded.name, predictions, preview, priority_fn, priority_config, elapsed)
            report["confidence_threshold_used"] = confidence_threshold
            report["checkpoint"] = str(Path(weights_text).expanduser().resolve())
            report_json.append(report)
            report_rows.append({
                "report_id": report["report_id"],
                "file_name": report["file_name"],
                "detections_count": report["detections_count"],
                "priority_score": report["priority_score"],
                "priority_band": report["priority_band"],
                "needs_review": report["needs_review"],
                "mean_confidence": report["mean_confidence"],
                "processing_seconds": report["processing_seconds"],
            })
            with st.expander(f"{uploaded.name} — {elapsed:.2f} seconds", expanded=(len(uploads) == 1)):
                left, right = st.columns([1, 1])
                with left:
                    st.image(preview, caption=uploaded.name, width="stretch")
                with right:
                    a, b = st.columns(2)
                    a.metric("Detected damages", len(predictions))
                    b.metric("Priority score", f"{report['priority_score']}/100")
                    st.metric("Priority band", report["priority_band"])
                    st.metric("CPU processing time", f"{elapsed:.2f} s")
                    if report["needs_review"]:
                        st.warning("This result should be reviewed by a person.")
                    if len(predictions) == 0:
                        st.info("No detections passed the selected confidence threshold. This does not prove the road is damage-free.")
                if len(predictions):
                    draw_detections(preview, predictions)
                st.json(report, expanded=False)
        except Exception as exc:
            st.error(f"{uploaded.name}: {exc}")
        finally:
            if image_path is not None:
                try:
                    image_path.unlink(missing_ok=True)
                except Exception:
                    pass
        progress.progress(index / len(uploads))

    if report_json:
        # Keep a dated JSON record and an append-only CSV report queue in the chosen folder.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = state_dir / f"roadscan_batch_{stamp}.json"
        json_path.write_text(json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8")
        csv_path = state_dir / "roadscan_reports.csv"
        new_df = pd.DataFrame(report_rows)
        if csv_path.exists():
            try:
                old_df = pd.read_csv(csv_path)
                new_df = pd.concat([old_df, new_df], ignore_index=True)
            except Exception:
                pass
        new_df.to_csv(csv_path, index=False)
        status.success("Finished. Reports saved.")
        st.session_state["roadscan_batch_rows"] = report_rows
        st.session_state["roadscan_batch_json"] = report_json

if st.session_state.get("roadscan_batch_rows"):
    st.subheader("Batch summary")
    batch_rows = st.session_state["roadscan_batch_rows"]
    batch_reports = st.session_state["roadscan_batch_json"]
    st.dataframe(pd.DataFrame(batch_rows), width="stretch", hide_index=True)
    st.download_button(
        "Download results CSV", data=pd.DataFrame(batch_rows).to_csv(index=False).encode("utf-8"),
        file_name="roadscan_results.csv", mime="text/csv",
    )
    st.download_button(
        "Download detailed reports JSON",
        data=json.dumps(batch_reports, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="roadscan_reports.json", mime="application/json",
    )
