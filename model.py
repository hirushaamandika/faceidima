import json
import os

import cv2
import numpy as np
from tensorflow.keras import layers, models

IMG_SIZE = 64
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "face_cnn.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
CONFIDENCE_THRESHOLD = 0.75  # below this, treat the face as "unknown"

os.makedirs(MODEL_DIR, exist_ok=True)


def build_model(num_classes: int) -> models.Sequential:
    """A small CNN: two conv blocks + dense head, ending in a softmax over registered people."""
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def load_dataset(dataset_dir: str):
    """Read every person's folder of face images into arrays for training."""
    people = sorted(
        d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
    )
    images, labels = [], []
    for idx, person in enumerate(people):
        person_dir = os.path.join(dataset_dir, person)
        for fname in os.listdir(person_dir):
            fpath = os.path.join(person_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(idx)

    X = np.array(images, dtype="float32") / 255.0
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(labels, dtype="int32")
    return X, y, people


def train_and_save(dataset_dir: str, epochs: int = 15):
    X, y, people = load_dataset(dataset_dir)

    if len(people) < 1 or len(X) == 0:
        raise ValueError("No registered face images found. Register at least one person first.")
    if len(people) < 2:
        raise ValueError(
            "Only one person is registered. Register at least two different people "
            "so the model has something to tell them apart from."
        )

    model = build_model(num_classes=len(people))

    # Use a validation split only if there's enough data to spare some for it.
    use_val = len(X) >= 20
    history = model.fit(
        X, y,
        epochs=epochs,
        validation_split=0.15 if use_val else 0.0,
        batch_size=16,
        verbose=2,
    )

    model.save(MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(people, f)

    final_acc = float(history.history["accuracy"][-1])
    return {"classes": people, "num_images": len(X), "final_train_accuracy": final_acc}


_cached_model = None
_cached_labels = None


def load_trained_model():
    """Load the model + label list, downloading from S3 first if not already cached locally
    (this covers fresh containers and worker processes that didn't do the training)."""
    global _cached_model, _cached_labels
    if _cached_model is not None:
        return _cached_model, _cached_labels

    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(LABELS_PATH):
        import storage
        found = storage.download_model_files(MODEL_DIR)
        if not found:
            return None, None

    from tensorflow.keras.models import load_model
    _cached_model = load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        _cached_labels = json.load(f)
    return _cached_model, _cached_labels


def reset_model_cache():
    """Call after retraining so the next prediction reloads the fresh model from disk."""
    global _cached_model, _cached_labels
    _cached_model = None
    _cached_labels = None


def predict_face(face_gray: np.ndarray):
    """Predict identity of a single cropped, grayscale face image.
    Returns (name_or_none, confidence).
    """
    model, labels = load_trained_model()
    if model is None:
        return None, 0.0

    face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    probs = model.predict(face, verbose=0)[0]
    best_idx = int(np.argmax(probs))
    confidence = float(probs[best_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return None, confidence

    return labels[best_idx], confidence
