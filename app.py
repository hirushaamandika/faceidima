import base64
import os
import re
import shutil

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

import database
import model as face_model
import storage

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")  # local scratch space only; source of truth is S3
os.makedirs(DATASET_DIR, exist_ok=True)

# --- Face detector (Haar Cascade). Prefer a local copy bundled with this project. ---
LOCAL_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
PACKAGE_CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
CASCADE_PATH = LOCAL_CASCADE_PATH if os.path.isfile(LOCAL_CASCADE_PATH) else PACKAGE_CASCADE_PATH
if not os.path.isfile(CASCADE_PATH):
    raise FileNotFoundError(
        "Haar cascade file not found. Download it from:\n"
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml\n"
        "and save it as 'haarcascade_frontalface_default.xml' next to app.py."
    )
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

database.init_db()


def decode_data_url(data_url: str) -> np.ndarray:
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def detect_largest_face(img_bgr: np.ndarray):
    """Return a cropped grayscale face (largest detected) or None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest by area
    return gray[y:y + h, x:x + w]


def safe_folder_name(name: str) -> str:
    name = name.strip().lower()
    return re.sub(r"[^a-z0-9_-]+", "_", name)


@app.route("/")
def index():
    return render_template("index.html", users=database.get_all_users())


@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/api/capture_face", methods=["POST"])
def capture_face():
    try:
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        image_data_url = data.get("image")
        if not name:
            return jsonify({"success": False, "error": "Name is required"}), 400
        if not image_data_url:
            return jsonify({"success": False, "error": "No image provided"}), 400

        img = decode_data_url(image_data_url)
        if img is None:
            return jsonify({"success": False, "error": "Invalid image data"}), 400

        face = detect_largest_face(img)
        if face is None:
            return jsonify({"success": False, "error": "No face detected"})

        folder = safe_folder_name(name)
        existing = storage.count_dataset_images(folder)
        ok, jpg_bytes = cv2.imencode(".jpg", face)
        if not ok:
            return jsonify({"success": False, "error": "Could not encode captured face"}), 500

        filename = f"img_{existing + 1:03d}.jpg"
        storage.upload_dataset_image(folder, filename, jpg_bytes.tobytes())

        return jsonify({"success": True, "count": existing + 1})
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Error in /api/capture_face")
        return jsonify({"success": False, "error": f"Server error: {exc}"}), 500


@app.route("/api/train", methods=["POST"])
def train():
    try:
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        if name:
            database.add_user_if_missing(name)

        # Pull the latest full dataset down from S3 into a clean local scratch folder.
        if os.path.isdir(DATASET_DIR):
            shutil.rmtree(DATASET_DIR)
        os.makedirs(DATASET_DIR, exist_ok=True)
        storage.download_dataset_to_local(DATASET_DIR)

        result = face_model.train_and_save(DATASET_DIR)

        # Push the freshly trained model back up to S3 so every instance can use it.
        storage.upload_model_files(face_model.MODEL_DIR)
        face_model.reset_model_cache()

        for person in result["classes"]:
            database.add_user_if_missing(person)

        return jsonify({"success": True, **result})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Error in /api/train")
        return jsonify({"success": False, "error": f"Server error: {exc}"}), 500


@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")


@app.route("/api/mark_attendance", methods=["POST"])
def mark_attendance_route():
    try:
        data = request.get_json(silent=True) or {}
        image_data_url = data.get("image")
        if not image_data_url:
            return jsonify({"success": False, "error": "No image provided"}), 400

        img = decode_data_url(image_data_url)
        if img is None:
            return jsonify({"success": False, "error": "Invalid image data"}), 400

        face = detect_largest_face(img)
        if face is None:
            return jsonify({"success": False, "error": "No face detected"})

        name, confidence = face_model.predict_face(face)
        if name is None:
            return jsonify({
                "success": False,
                "error": f"Face not recognized (confidence {confidence:.2f}). Register first or retrain.",
            })

        newly_marked = database.mark_attendance(name)
        message = f"Attendance marked for {name}" if newly_marked else f"{name} was already marked today"
        return jsonify({
            "success": True,
            "name": name,
            "confidence": round(confidence, 3),
            "newly_marked": newly_marked,
            "message": message,
        })
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Error in /api/mark_attendance")
        return jsonify({"success": False, "error": f"Server error: {exc}"}), 500


@app.route("/api/reload_model", methods=["POST"])
def reload_model():
    """Force this instance to pull the latest trained model from S3.
    Useful after training happens on a different instance, or after a redeploy."""
    try:
        face_model.reset_model_cache()
        # Remove any stale local copies so load_trained_model() is forced to re-download.
        for path in (face_model.MODEL_PATH, face_model.LABELS_PATH):
            if os.path.isfile(path):
                os.remove(path)
        model, labels = face_model.load_trained_model()
        if model is None:
            return jsonify({"success": False, "error": "No trained model found in S3 yet."}), 404
        return jsonify({"success": True, "classes": labels})
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Error in /api/reload_model")
        return jsonify({"success": False, "error": f"Server error: {exc}"}), 500


@app.route("/records")
def records_page():
    return render_template("records.html", records=database.get_attendance_records())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
