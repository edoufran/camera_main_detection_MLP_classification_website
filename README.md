# Détection de signes de main en temps réel

Détection et classification de signes de main depuis une webcam, en temps réel. Le projet combine **MediaPipe** pour l'extraction des points clés de la main et un **réseau de neurones MLP** pour la classification.

---

## Signes reconnus

| Signe | Geste |
|-------|-------|
| `pouce_leve` | Pouce vers le haut |
| `paix` | Index et majeur levés |
| `ok` | Pouce et index formant un cercle |
| `stop` | Main ouverte, doigts tendus |
| `rock` | Index et auriculaire levés |

---

## Installation

```bash
pip install -r requirements.txt
```

**Dépendances :**

| Package | Version | Rôle |
|---------|---------|------|
| `mediapipe` | 0.10.21 | Détection des landmarks de la main |
| `opencv-python` | ≥ 4.8 | Capture webcam et affichage |
| `numpy` | ≥ 1.24 | Manipulation des données |
| `scikit-learn` | ≥ 1.3 | Entraînement et évaluation du modèle |

---

## Utilisation

### 1. Collecter les données

```bash
python collect_data.py
```

Le script parcourt chaque signe un par un. Faire le signe devant la caméra, puis appuyer sur **ESPACE** pour démarrer la capture. Le script enregistre automatiquement 150 exemples, puis passe au signe suivant.

**Contrôles :** `ESPACE` pour démarrer la capture | `Q` pour quitter

**Fichiers générés dans `data/` :**
```
data/
├── X.npy       # Landmarks (750 × 63)
├── y.npy       # Labels encodés (750,)
└── labels.npy  # Noms des signes ['pouce_leve', 'paix', ...]
```

---

### 2. Entraîner le modèle

```bash
python train_model.py
```

Entraîne le classificateur sur les données collectées et affiche les métriques.

**Exemple de sortie :**
```
Découpage : 600 train | 150 test

=============================================
  Précision globale : 98.7%
=============================================

Rapport par signe :
              precision  recall  f1-score   support
  pouce_leve       1.00    1.00      1.00        30
        paix       0.97    1.00      0.98        30
          ok       1.00    0.97      0.98        30
        stop       1.00    1.00      1.00        30
        rock       0.97    0.97      0.97        30
```

**Fichiers générés dans `model/` :**
```
model/
├── model.pkl    # Classificateur MLP sérialisé
├── scaler.pkl   # StandardScaler sérialisé
└── labels.npy   # Noms des signes
```

---

### 3. Lancer la détection en temps réel

```bash
python main.py
```

Ouvre la webcam et affiche en temps réel le signe détecté avec son score de confiance. Le signe n'est affiché que si la confiance dépasse **70%**.

**Contrôles :** `Q` pour quitter

---

## Structure du projet

```
signe-main-detection/
├── collect_data.py    # Collecte des données via webcam
├── train_model.py     # Entraînement du classificateur
├── main.py            # Détection en temps réel
├── requirements.txt
├── data/              # Généré après collect_data.py
└── model/             # Généré après train_model.py
```

---

## Fonctionnement technique

### Pipeline complet

```
Webcam
  └─► MediaPipe Hands
        └─► 21 landmarks (x, y, z)
              └─► Normalisation / StandardScaler
                    └─► MLP Classifier
                          └─► Signe prédit + confiance
```

---

### MediaPipe Hands

MediaPipe détecte la main dans l'image et retourne **21 points de repère (landmarks)**, chacun avec des coordonnées `(x, y, z)` normalisées entre 0 et 1 par rapport aux dimensions de l'image.

```
                    8   12  16  20
                    |   |   |   |
                7   |   |   |   19
                |  11   |  15   |
             6  | /  \  | /  \ 18
             | 10    14  |    17
          5  |/       \ |   /
           \ 9         13  16
            \|          | /
             4     2    |/
              \   / \   3
               \ /   \ /
                1     0  ← poignet (référence)
```

**Normalisation des landmarks :** toutes les coordonnées sont exprimées relativement au poignet (point 0). Ainsi, déplacer la main dans l'image ne change pas les features — seule la forme du geste compte.

```python
# Avant normalisation : position absolue dans l'image
lm.x = 0.67, lm.y = 0.42

# Après normalisation : position relative au poignet
lm.x - wrist.x = 0.12, lm.y - wrist.y = -0.23
```

Résultat : un vecteur de **21 × 3 = 63 features** par frame.

---

### MLP Classifier (scikit-learn)

Un **MLP (Multi-Layer Perceptron)** est un réseau de neurones entièrement connecté. Chaque neurone calcule une somme pondérée de ses entrées, puis applique une fonction d'activation non-linéaire (`ReLU`).

```
Entrée       Couche 1     Couche 2     Sortie
  63    →      128    →      64    →     5
neurones     neurones     neurones    classes
```

Pourquoi un MLP plutôt qu'un CNN ?
- Les features (landmarks normalisés) sont déjà extraites par MediaPipe.
- Il n'y a pas d'image brute à analyser, seulement 63 valeurs numériques.
- Un MLP est suffisant, rapide à entraîner et léger à l'inférence.

---

### StandardScaler

Avant l'entraînement, chaque feature est centrée et réduite :

```
x_scaled = (x - mean) / std
```

Sans cette étape, les features avec de grandes valeurs absolues domineraient l'entraînement. Le scaler est appris uniquement sur les données d'entraînement, puis appliqué à l'identique sur les données de test et lors de l'inférence en temps réel.

---

### Lissage temporel (vote majoritaire)

Le modèle prédit un signe à chaque frame (~30 fois par seconde). Sans lissage, l'affichage fluctuerait constamment. Un `deque` de taille 10 stocke les 10 dernières prédictions ; le signe affiché est celui qui apparaît le plus souvent dans cette fenêtre.

```python
history = deque(maxlen=10)
history.append(pred_idx)
smoothed = max(set(history), key=history.count)  # vote majoritaire
```

---

### Seuil de confiance

Le modèle retourne une distribution de probabilités sur les 5 classes (ex: `[0.02, 0.01, 0.94, 0.01, 0.02]`). Si la probabilité maximale est inférieure à **0.70**, aucun signe n'est affiché — évitant les faux positifs lorsque la main est dans une position ambiguë ou en transition.

---

## Comprendre les métriques d'entraînement

| Métrique | Définition |
|----------|-----------|
| **Precision** | Parmi les fois où le modèle a prédit ce signe, combien étaient correctes |
| **Recall** | Parmi les vraies occurrences de ce signe, combien ont été détectées |
| **F1-score** | Moyenne harmonique de precision et recall (équilibre les deux) |
| **Support** | Nombre d'exemples de ce signe dans le jeu de test |

Une précision globale **> 95%** est attendue avec 150 exemples par signe bien capturés.

---

## Ajouter de nouveaux signes

1. Ajouter le nom du signe dans `SIGNS` dans `collect_data.py`
2. Relancer `python collect_data.py`
3. Relancer `python train_model.py`

Le reste du code s'adapte automatiquement.
