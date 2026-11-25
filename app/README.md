# PredictMaint AI — Documentation de l'Application Web & Inférence

Ce document détaille l'architecture, le cycle de vie du modèle de Machine Learning, le contrat d'interface et la procédure de test du module `app/`.

---

## 1. Vue d'ensemble de l'Architecture

Le module `app/` fournit le service d'inférence en production pour le projet **PredictMaint AI**. Il intègre :
- **Une API REST haute performance** développée avec **FastAPI**.
- **Une validation stricte des données d'entrée** via **Pydantic v2**.
- **Le chargement sécurisé et traçable du modèle** depuis le registre **MLflow** hébergé sur **DagsHub**.
- **Une interface utilisateur web interactive** servie directement (`app/static/index.html`) avec des cas de test préconfigurés pour l'évaluation.

---

## 2. Cycle de vie et Chargement du Modèle (`model_loader.py`)

Conformément aux exigences du sujet et aux règles d'or fixées avec l'équipe MLOps :

### Mode Production (Par défaut)
- **Serveur MLflow** : `https://dagshub.com/Abdel380/predictmaint-ai.mlflow`
- **Nom du modèle** : `predictmaint-ai`
- **Alias de production** : `champion`
- **Résolution dynamique** : Au démarrage de l'application (ou lors d'un rafraîchissement), le service interroge le registre MLflow via `MlflowClient.get_model_version_by_alias()`. Il résout l'alias `champion` en un **numéro de version concret** (ex : version `1`), puis charge ce modèle précis avec `mlflow.sklearn.load_model()`.
- **Règle d'or (Aucun mock silencieux en production)** :
  Si le serveur MLflow est inaccessible ou qu'aucun modèle approuvé n'est disponible, l'application **ne charge aucun faux modèle de secours**. La sonde de préparation `/ready` renvoie strictement une erreur **`503 Service Unavailable`**, garantissant que le trafic n'est servi que lorsque le modèle champion est opérationnel.

### Mode Mock / Développement Local
- Activé **uniquement de manière explicite** via la variable d'environnement :
  ```bash
  ENABLE_MOCK_MODEL=true
  ```
- Dans ce mode, un modèle simulé (`MockModelPipeline`) est utilisé en mémoire. Cela permet de :
  1. Exécuter l'intégralité de la suite de tests automatisés `pytest` en local et en CI sans dépendre du réseau.
  2. Développer et tester l'interface graphique sans nécessiter les tokens secrets DagsHub.

---

## 3. Contrat d'Interface & Endpoints de l'API

### Endpoints disponibles

| Méthode | Endpoint | Rôle | Code Succès | Code Erreur |
|---|---|---|---|---|
| `GET` | `/` | Interface web utilisateur (HTML/CSS) | `200` | - |
| `GET` | `/ready` | Sonde de disponibilité (*Readiness probe*) | `200` | `503` si non prêt |
| `GET` | `/model-info` | Affiche la version réelle du modèle chargé | `200` | `503` si non prêt |
| `POST` | `/predict` | Analyse les données machine et retourne le diagnostic | `200` | `422` (invalide) / `503` |
| `GET` | `/docs` | Documentation interactive Swagger OpenAPI | `200` | - |

---

### Format des Données (`app/schemas.py`)

#### Requête d'entrée (`POST /predict`)
L'API accepte des clés JSON modernes en snake_case :
```json
{
  "type": "L",
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1450.0,
  "torque": 48.0,
  "tool_wear": 180.0
}
```

#### Reconstruction interne pour le modèle Scikit-learn
Avant d'appeler le modèle entraîné par l'équipe ML, l'application reconstruit automatiquement un DataFrame Pandas avec les **6 noms de colonnes exacts du dataset AI4I 2020** :
```python
["Type", "Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
```

#### Réponse de l'API
```json
{
  "failure_prediction": 0,
  "risk_score": 0.12,
  "model_name": "predictmaint-ai",
  "model_version": "1"
}
```
- `failure_prediction` : `0` (machine saine) ou `1` (panne imminente).
- `risk_score` : score compris entre `0.0` et `1.0` (représentant le niveau de risque de défaillance).
- `model_name` : identifiant du modèle dans le registre.
- `model_version` : version concrète réellement chargée en mémoire (garantit la traçabilité).

---

## 4. Tests Automatisés (`tests/test_app.py`)

Le projet inclut une suite de tests automatisés couvrant tous les cas critiques requis par le barème du projet :

1. **Test des prédictions nominales et format** (`test_prediction_success_and_format`) :
   - Vérifie la réponse pour une machine saine (score bas).
   - Vérifie la détection d'une machine à risque d'usure critique (score > 0.5, prédiction 1).
   - Valide le type et la présence de tous les champs de sortie.

2. **Test des cas limites et valeurs aberrantes** (`test_invalid_inputs_and_edge_cases`) :
   - Rejet immédiat (erreur `422 Unprocessable Entity`) en cas de :
     - Type de produit non reconnu (différent de `L`, `M`, `H`).
     - Température négative ($\le 0$ K).
     - Vitesse de rotation négative ou nulle ($\le 0$ rpm).
     - Couple négatif ($< 0$ Nm).
     - Temps d'usure négatif ($< 0$ min).
     - Donnée obligatoire manquante.

3. **Test de disponibilité et non-régression du Champion** (`test_readiness_failure...` & `test_champion_loaded...`) :
   - Vérifie que `/ready` et `/predict` renvoient strictement une erreur `503` si le modèle n'est pas prêt.
   - Vérifie que l'alias `champion` est toujours respecté même si une version candidate plus récente (ex: v2) est présente dans le registre.

---

## 5. Guide d'Exécution

### Lancer la suite de tests :
```bash
pytest -v
```

### Lancer l'application en mode local de test (Mock) :
```powershell
$env:ENABLE_MOCK_MODEL="true"; uvicorn app.main:app --reload
```
Puis ouvrir : `http://localhost:8000`

### Lancer l'application connectée au vrai MLflow DagsHub :
Configurer d'abord les identifiants d'accès dans l'environnement ou dans un fichier `.env` :
```bash
MLFLOW_TRACKING_USERNAME="votre_username_dagshub"
MLFLOW_TRACKING_PASSWORD="votre_token_dagshub"
```
Puis exécuter :
```bash
uvicorn app.main:app --reload
```
