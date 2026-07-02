"""
main.py — Détection en temps réel

Charge le modèle entraîné et classifie les signes de main en direct
depuis la webcam. Le signe est affiché uniquement si la confiance
dépasse 70%. Les prédictions sont lissées sur les 10 dernières frames
pour éviter les fluctuations.

Contrôles : Q pour quitter
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import deque

MODEL_DIR = "model"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def extract_landmarks(hand_landmarks):
    """Vecteur de 63 valeurs normalisé par rapport au poignet (landmark 0)."""
    wrist = hand_landmarks.landmark[0]
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])
    return np.array(landmarks)


def load_model():
    """Charge le modèle, le scaler et les labels depuis model/."""
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    if not os.path.exists(model_path):
        print("Erreur : aucun modèle trouvé dans 'model/'.")
        print("Lancer d'abord : python train_model.py")
        exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    labels = np.load(os.path.join(MODEL_DIR, "labels.npy"))

    return model, scaler, labels


def draw_overlay(frame, sign_text, confidence):
    """Affiche le signe détecté et le score de confiance en bas de l'image."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 95), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if sign_text:
        label = sign_text.replace("_", " ").upper()
        cv2.putText(frame, label, (20, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 100), 3)
        cv2.putText(frame, f"Confiance : {confidence * 100:.0f}%", (20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
    else:
        cv2.putText(frame, "Aucune main detectee", (20, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)


def main():
    model, scaler, labels = load_model()
    print(f"Modèle chargé. Signes reconnus : {list(labels)}")
    print("Détection en cours... (Q pour quitter)\n")

    history = deque(maxlen=10)  # Lissage temporel des prédictions

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        return

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            sign_text = None
            confidence = 0.0

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                landmarks = extract_landmarks(hand)
                landmarks_scaled = scaler.transform([landmarks])

                proba = model.predict_proba(landmarks_scaled)[0]
                pred_idx = int(np.argmax(proba))
                confidence = proba[pred_idx]

                history.append(pred_idx)

                # Signe le plus fréquent dans l'historique (vote majoritaire)
                smoothed_idx = max(set(history), key=history.count)

                if confidence > 0.70:
                    sign_text = labels[smoothed_idx]
            else:
                history.clear()

            draw_overlay(frame, sign_text, confidence)
            cv2.imshow("Detection de signes de main", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
