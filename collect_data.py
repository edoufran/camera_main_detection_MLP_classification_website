"""
collect_data.py — Collecte des données d'entraînement

Pour chaque signe défini dans SIGNS, MediaPipe extrait les 21 landmarks
de la main depuis la webcam. Les landmarks sont normalisés et sauvegardés
dans data/ sous forme de tableaux numpy.

Contrôles : ESPACE pour démarrer la capture | Q pour quitter
"""

import cv2
import mediapipe as mp
import numpy as np
import os

SIGNS = ["pouce_leve", "paix", "ok", "stop", "rock"]
SAMPLES_PER_SIGN = 150
DATA_DIR = "data"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def extract_landmarks(hand_landmarks):
    """
    Retourne un vecteur de 63 valeurs (21 points × 3 coordonnées)
    normalisé par rapport au poignet (landmark 0).

    La normalisation rend les features invariantes à la position
    de la main dans l'image — seule la forme du signe compte.
    """
    wrist = hand_landmarks.landmark[0]
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])
    return np.array(landmarks)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        return

    all_X = []
    all_y = []

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        for sign_idx, sign in enumerate(SIGNS):
            samples = []
            capturing = False

            print(f"\n[{sign_idx + 1}/{len(SIGNS)}] Signe : {sign}")
            print("Faire le signe devant la caméra, puis appuyer sur ESPACE.")

            while len(samples) < SAMPLES_PER_SIGN:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                    if capturing:
                        landmarks = extract_landmarks(hand)
                        samples.append(landmarks)

                cv2.putText(frame, f"Signe : {sign.replace('_', ' ')}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                if capturing:
                    status = f"Capture : {len(samples)}/{SAMPLES_PER_SIGN}"
                    color = (0, 255, 0)
                    progress = int((len(samples) / SAMPLES_PER_SIGN) * 400)
                    cv2.rectangle(frame, (20, 105), (420, 130), (50, 50, 50), -1)
                    cv2.rectangle(frame, (20, 105), (20 + progress, 130), (0, 255, 0), -1)
                else:
                    status = "ESPACE pour commencer"
                    color = (0, 165, 255)

                cv2.putText(frame, status, (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                cv2.imshow("Collecte de donnees", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' ') and not capturing:
                    capturing = True
                    print("  Capture démarrée...")
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            all_X.append(np.array(samples))
            all_y.append(np.full(len(samples), sign_idx))
            print(f"  ✓ {len(samples)} échantillons — '{sign}'")

    cap.release()
    cv2.destroyAllWindows()

    X = np.vstack(all_X)       # (total_samples, 63)
    y = np.concatenate(all_y)  # (total_samples,)

    np.save(os.path.join(DATA_DIR, "X.npy"), X)
    np.save(os.path.join(DATA_DIR, "y.npy"), y)
    np.save(os.path.join(DATA_DIR, "labels.npy"), np.array(SIGNS))

    print(f"\n✓ Dataset sauvegardé dans '{DATA_DIR}/'")
    print(f"  {X.shape[0]} exemples | {X.shape[1]} features")


if __name__ == "__main__":
    main()
