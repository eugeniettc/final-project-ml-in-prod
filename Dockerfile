# 1. Image de base officielle Python légère
FROM python:3.11-slim

# Variables d'environnement pour optimiser l'exécution de Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Répertoire de travail dans le conteneur
WORKDIR /app

# 2. Installation de curl pour le test de santé (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Création d'un utilisateur non-root pour la sécurité (Bonne pratique industrielle)
RUN useradd -m -u 1000 appuser

# 4. Copie et installation des dépendances (optimisation du cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copie du code de l'application
COPY app/ app/

# Attribution des droits à l'utilisateur non-root
RUN chown -R appuser:appuser /app

# Bascule vers l'utilisateur sécurisé
USER appuser

# Exposition du port
EXPOSE 8000

# Sonde de santé intégrée (Healthcheck Docker)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ready || exit 1

# 6. Commande de démarrage avec port dynamique pour le Cloud
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]