"""
train_model.py — Entraînement du classificateur

Charge les données de data/, entraîne un MLP (Multi-Layer Perceptron)
et sauvegarde le modèle et le scaler dans model/.

Architecture : Entrée (63) → 128 → 64 → Sortie (N signes)
"""

import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

DATA_DIR = "data"
MODEL_DIR = "model"


def main():
    print("Chargement des données...")
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))

    print(f"  {X.shape[0]} exemples | {X.shape[1]} features | {len(labels)} signes")
    print(f"  Signes : {list(labels)}")

    # Découpage 80% train / 20% test, stratifié pour équilibrer les classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nDécoupage : {len(X_train)} train | {len(X_test)} test")

    # Normalisation : centre chaque feature à 0, écart-type à 1
    # Le scaler est appris sur le train uniquement, puis appliqué au test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nEntraînement du modèle MLP...")
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train, y_train)
    print("  Terminé.")

    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    print(f"\n{'='*45}")
    print(f"  Précision globale : {accuracy * 100:.1f}%")
    print(f"{'='*45}")
    print("\nRapport par signe :")
    print(classification_report(y_test, y_pred, target_names=labels))

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    np.save(os.path.join(MODEL_DIR, "labels.npy"), labels)

    print(f"✓ Modèle sauvegardé dans '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
