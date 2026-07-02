import os
import base64
import numpy as np
import pickle
import cv2
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import deque

app = Flask(__name__)
CORS(app)

# Paths
DATA_DIR = "/app/data"
MODEL_DIR = "/app/model"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# In-memory prediction smoothing
prediction_history = deque(maxlen=10)

# --- Helpers ---

def load_model():
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    labels_path = os.path.join(MODEL_DIR, "labels.npy")
    if not all(os.path.exists(p) for p in [model_path, scaler_path, labels_path]):
        return None, None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    labels = np.load(labels_path, allow_pickle=True)
    return model, scaler, labels


def extract_landmarks(image_bgr):
    """Run MediaPipe on a BGR image and return normalized 63-feature vector."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None
    lms = results.multi_hand_landmarks[0].landmark
    wrist = lms[0]
    features = []
    for lm in lms:
        features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return np.array(features)


def decode_image(b64_string):
    """Decode a base64 image (data URI or raw) to a BGR numpy array."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


# --- Routes ---

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/signs", methods=["GET"])
def get_signs():
    """Return current list of signs (from existing data or default list)."""
    labels_path = os.path.join(DATA_DIR, "labels.npy")
    if os.path.exists(labels_path):
        labels = np.load(labels_path, allow_pickle=True).tolist()
    else:
        labels = ["pouce_leve", "paix", "ok", "stop", "rock"]
    return jsonify({"signs": labels})


@app.route("/signs", methods=["POST"])
def update_signs():
    """Update the sign list."""
    data = request.json
    signs = data.get("signs", [])
    if not signs:
        return jsonify({"error": "Liste vide"}), 400
    np.save(os.path.join(DATA_DIR, "labels.npy"), np.array(signs))
    return jsonify({"signs": signs})


@app.route("/collect", methods=["POST"])
def collect():
    """
    Receive a frame + sign label, extract landmarks, append to dataset.
    Body: { "image": "<base64>", "sign": "pouce_leve" }
    """
    data = request.json
    image = decode_image(data["image"])
    sign = data["sign"]

    features = extract_landmarks(image)
    if features is None:
        return jsonify({"error": "Aucune main détectée dans l'image"}), 400

    # Load existing data
    X_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")
    labels_path = os.path.join(DATA_DIR, "labels.npy")

    if os.path.exists(labels_path):
        labels = np.load(labels_path, allow_pickle=True).tolist()
    else:
        labels = ["pouce_leve", "paix", "ok", "stop", "rock"]
        np.save(labels_path, np.array(labels))

    if sign not in labels:
        return jsonify({"error": f"Signe '{sign}' inconnu"}), 400

    sign_idx = labels.index(sign)

    if os.path.exists(X_path) and os.path.exists(y_path):
        X = np.load(X_path)
        y = np.load(y_path)
        X = np.vstack([X, features])
        y = np.append(y, sign_idx)
    else:
        X = features.reshape(1, -1)
        y = np.array([sign_idx])

    np.save(X_path, X)
    np.save(y_path, y)

    # Count samples for this sign
    count = int(np.sum(y == sign_idx))
    return jsonify({"success": True, "sign": sign, "count": count, "total": len(y)})


@app.route("/data/stats", methods=["GET"])
def data_stats():
    """Return sample count per sign."""
    X_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")
    labels_path = os.path.join(DATA_DIR, "labels.npy")

    if not os.path.exists(X_path):
        return jsonify({"total": 0, "per_sign": {}})

    y = np.load(y_path)
    labels = np.load(labels_path, allow_pickle=True).tolist()
    per_sign = {label: int(np.sum(y == i)) for i, label in enumerate(labels)}
    return jsonify({"total": len(y), "per_sign": per_sign})


@app.route("/data/reset", methods=["DELETE"])
def reset_data():
    """Delete all collected data."""
    for fname in ["X.npy", "y.npy"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            os.remove(path)
    return jsonify({"success": True})


@app.route("/train", methods=["POST"])
def train():
    """Train the MLP on collected data."""
    X_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")
    labels_path = os.path.join(DATA_DIR, "labels.npy")

    if not os.path.exists(X_path):
        return jsonify({"error": "Aucune donnée collectée"}), 400

    X = np.load(X_path)
    y = np.load(y_path)
    labels = np.load(labels_path, allow_pickle=True).tolist()

    if len(X) < 20:
        return jsonify({"error": f"Pas assez de données ({len(X)} exemples, minimum 20)"}), 400

    # Check all classes have samples
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        return jsonify({"error": "Il faut des données pour au moins 2 signes différents"}), 400

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True, zero_division=0)

    # Save model
    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    np.save(os.path.join(MODEL_DIR, "labels.npy"), np.array(labels))

    per_sign = {}
    for sign in labels:
        if sign in report:
            per_sign[sign] = {
                "precision": round(report[sign]["precision"], 3),
                "recall": round(report[sign]["recall"], 3),
                "f1": round(report[sign]["f1-score"], 3),
                "support": int(report[sign]["support"]),
            }

    return jsonify({
        "success": True,
        "accuracy": round(accuracy * 100, 1),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "per_sign": per_sign,
    })


@app.route("/model/status", methods=["GET"])
def model_status():
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    return jsonify({"trained": os.path.exists(model_path)})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict sign from a frame.
    Body: { "image": "<base64>" }
    """
    model, scaler, labels = load_model()
    if model is None:
        return jsonify({"error": "Modèle non entraîné"}), 400

    data = request.json
    image = decode_image(data["image"])
    features = extract_landmarks(image)

    if features is None:
        prediction_history.clear()
        return jsonify({"sign": None, "confidence": 0, "hand_detected": False})

    features_scaled = scaler.transform(features.reshape(1, -1))
    proba = model.predict_proba(features_scaled)[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])

    # Temporal smoothing
    if confidence >= 0.70:
        prediction_history.append(pred_idx)
    else:
        prediction_history.clear()

    if len(prediction_history) > 0:
        smoothed_idx = max(set(prediction_history), key=list(prediction_history).count)
        smoothed_sign = labels[smoothed_idx]
        smoothed_conf = float(proba[smoothed_idx])
    else:
        smoothed_sign = None
        smoothed_conf = confidence

    all_proba = {labels[i]: round(float(p), 3) for i, p in enumerate(proba)}

    return jsonify({
        "sign": smoothed_sign,
        "confidence": round(smoothed_conf * 100, 1),
        "hand_detected": True,
        "all_probabilities": all_proba,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
