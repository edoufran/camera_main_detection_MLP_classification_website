# Image de base Python légère
FROM python:3.11-slim

# Dossier de travail dans le container
WORKDIR /app

# Dépendances système nécessaires pour OpenCV et MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
# On copie requirements.txt en premier pour profiter du cache Docker :
# si requirements.txt ne change pas, Docker réutilise le layer déjà construit
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY app.py .

# Créer les dossiers de données (ils seront montés en volume)
RUN mkdir -p /app/data /app/model

# Exposer le port Flask
EXPOSE 5000

# Lancer le serveur avec gunicorn (plus robuste que flask run en prod)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
