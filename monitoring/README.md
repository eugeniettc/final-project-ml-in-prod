# PredictMaint AI — Observabilité, Monitoring & Alerting (Bonus)

Ce dossier contient la stack complète d'observabilité en temps réel (Logging structuré, Métriques Prometheus, Tableaux de bord Grafana et Détection de dérive des capteurs) adaptée aux exigences industrielles du projet.

---

## 1. Architecture de Surveillance

---

## 2. Démarrage de la Stack

Assurez-vous que **Docker Desktop** est lancé, puis exécutez à la racine du projet :

```bash
docker compose up --build -d

Accès aux interfaces :
API FastAPI & Swagger : http://localhost:8000/docs
Métriques brutes Prometheus : http://localhost:8000/metrics
Prometheus UI : http://localhost:9090 (Vérifier que la cible api est UP dans Status -> Targets)
Grafana UI : http://localhost:3000 (Identifiants par défaut : admin / admin)
```
