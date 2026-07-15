import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = Path(r"D:\Editors\Vs code\Code\03_FiverrProjects\test\t")
if not IMAGE_DIR.exists() or not IMAGE_DIR.is_dir():
    IMAGE_DIR = BASE_DIR

STATE_FILE = IMAGE_DIR / ".editor_state.json"
OUTPUT_DIR = IMAGE_DIR / "colored_pic"
MASK_DIR = OUTPUT_DIR / "_masks"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

app = Flask(__name__)


def list_images():
    images = []
    for p in sorted(IMAGE_DIR.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(p)
    return images


def read_state():
    if not STATE_FILE.exists():
        return {"last_index": 0}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"last_index": 0}


def write_state(last_index: int):
    payload = {"last_index": int(last_index)}
    STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def image_path_by_index(index: int):
    images = list_images()
    if not images:
        return None, images
    if index < 0:
        index = 0
    if index >= len(images):
        index = len(images) - 1
    return images[index], images


def output_image_path(name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def output_mask_path(name: str):
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    return MASK_DIR / f"{stem}.png"


def overlay_red(original_bgr: np.ndarray, mask_gray: np.ndarray):
    mask = mask_gray > 0
    result = original_bgr.copy()
    red = np.zeros_like(result)
    red[:, :, 2] = 255
    result[mask] = cv2.addWeighted(result[mask], 0.45, red[mask], 0.55, 0)
    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/images")
def api_images():
    images = list_images()
    state = read_state()
    last_index = int(state.get("last_index", 0))
    if images:
        last_index = max(0, min(last_index, len(images) - 1))
    else:
        last_index = 0
    return jsonify(
        {
            "count": len(images),
            "images": [p.name for p in images],
            "last_index": last_index,
        }
    )


@app.route("/api/state", methods=["POST"])
def api_state():
    data = request.get_json(silent=True) or {}
    index = int(data.get("last_index", 0))
    write_state(index)
    return jsonify({"ok": True})


@app.route("/api/image/<int:index>")
def api_image(index: int):
    path, images = image_path_by_index(index)
    if path is None:
        return jsonify({"error": "No images found in folder."}), 404

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": f"Failed to read image: {path.name}"}), 500

    mask_file = output_mask_path(path.name)
    if mask_file.exists():
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape[:2] != image.shape[:2]:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
    else:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

    ok_img, img_png = cv2.imencode(".png", image)
    ok_mask, mask_png = cv2.imencode(".png", mask)
    if not ok_img or not ok_mask:
        return jsonify({"error": "Failed to encode image."}), 500

    return jsonify(
        {
            "index": index,
            "name": path.name,
            "total": len(images),
            "image_data": base64.b64encode(img_png.tobytes()).decode("utf-8"),
            "mask_data": base64.b64encode(mask_png.tobytes()).decode("utf-8"),
        }
    )


@app.route("/api/save/<int:index>", methods=["POST"])
def api_save(index: int):
    path, images = image_path_by_index(index)
    if path is None:
        return jsonify({"error": "No images found."}), 404

    data = request.get_json(silent=True) or {}
    mask_data = data.get("mask_data")
    if not mask_data:
        return jsonify({"error": "mask_data is required."}), 400

    try:
        mask_bytes = base64.b64decode(mask_data)
        mask_pil = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask = np.array(mask_pil, dtype=np.uint8)
    except Exception as exc:
        return jsonify({"error": f"Invalid mask data: {exc}"}), 400

    original = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if original is None:
        return jsonify({"error": "Cannot load original image."}), 500
    if mask.shape[:2] != original.shape[:2]:
        return jsonify({"error": "Mask size does not match image size."}), 400

    colored = overlay_red(original, mask)
    out_img = output_image_path(path.name)
    out_mask = output_mask_path(path.name)

    cv2.imwrite(str(out_img), colored)
    cv2.imwrite(str(out_mask), mask)
    write_state(index)

    return jsonify({"ok": True, "saved_to": str(out_img)})


@app.route("/image/<int:index>")
def raw_image(index: int):
    path, _ = image_path_by_index(index)
    if path is None:
        return jsonify({"error": "No images found."}), 404
    return send_file(path)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=8000, debug=True)
