# PredictMaint AI — Documentation de Déploiement & Twelve-Factor App

Ce document détaille l'architecture de déploiement, l'application de la méthodologie des **12 facteurs** (Twelve-Factor App) et les procédures de reproduction du déploiement en local et en production pour le projet **PredictMaint AI**.

---

## 1. Méthodologie Twelve-Factor App

L'application web et le service d'inférence ont été conçus selon les principes de la méthodologie _Twelve-Factor App_ pour assurer leur portabilité, leur observabilité et leur résilience en production.

| N°       | Facteur                                                    | Implémentation dans PredictMaint AI                                                                                                                                                                                                                                                   |
| -------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **I**    | **Codebase** (Base de code)                                | Un seul dépôt Git centralisé partagé par l'équipe. Le code source, les tests et les définitions de workflows y sont versionnés avec un flux de branches strict (`dev` -> `staging` -> `main`).                                                                                        |
| **II**   | **Dependencies** (Dépendances)                             | Dépendances Python explicitement déclarées dans `requirements.txt` avec des versions isolées au sein d'une image Docker `python:3.11-slim`. Aucune dépendance implicite au système hôte.                                                                                              |
| **III**  | **Config** (Configuration)                                 | Séparation stricte du code et de la configuration. Tous les paramètres sensibles ou variables (`MLFLOW_TRACKING_URI`, identifiants DagsHub/Azure, port `PORT`) sont injectés à l'exécution via des variables d'environnement. Zéro secret stocké dans l'image Docker ou le dépôt Git. |
| **IV**   | **Backing Services** (Services d'appui)                    | Les services tiers (Azure Blob Storage pour DVC, DagsHub pour MLflow Tracking et Registry) sont traités comme des ressources attachées, accessibles via des URL et identifiants configurables sans modification de code.                                                              |
| **V**    | **Build, Release, Run** (Assemblage, livraison, exécution) | Séparation stricte des étapes : le _Build_ fabrique l'image Docker identifiée par le Git commit SHA ; la _Release_ associe cette image immuable aux variables d'environnement cibles ; le _Run_ exécute le conteneur sur l'infrastructure cloud.                                      |
| **VI**   | **Processes** (Processus)                                  | Le service d'inférence FastAPI s'exécute comme un processus sans état (_stateless_). Aucune donnée d'état n'est conservée localement sur le disque du conteneur, permettant un redémarrage ou un remplacement immédiat.                                                               |
| **VII**  | **Port Binding** (Association de ports)                    | L'application est autonome et s'associe directement à un port HTTP via le serveur ASGI Uvicorn (`--host 0.0.0.0 --port ${PORT:-8000}`), sans nécessiter de serveur web tiers pré-installé.                                                                                            |
| **VIII** | **Concurrency** (Concurrence)                              | Passage à l'échelle horizontal par multiplication des conteneurs légers derrière le répartiteur de charge de la plateforme cloud (Render, Azure Container Apps ou Cloud Run).                                                                                                         |
| **IX**   | **Disposability** (Jetabilité)                             | Processus à démarrage rapide (chargement optimisé du modèle au démarrage via FastAPI lifespan) et arrêt propre (_graceful shutdown_) géré nativement par Uvicorn à la réception d'un signal SIGTERM.                                                                                  |
| **X**    | **Dev/Prod Parity** (Parité dev/prod)                      | Utilisation de la même image Docker en environnement local de test, sur les runners de CI/CD et en production sur le cloud, garantissant un comportement rigoureusement identique.                                                                                                    |
| **XI**   | **Logs** (Journaux d'événements)                           | Les journaux applicatifs sont traités comme un flux continu d'événements émis directement sur la sortie standard (`stdout`/`stderr`), capturés et centralisés par la plateforme d'hébergement sans fichier de log local.                                                              |
| **XII**  | **Admin Processes** (Processus d'administration)           | Les tâches de maintenance ponctuelles (réentraînement continu `ml/retrain.py`, audit ou rollback `ml/rollback.py`) sont exécutées sous forme de scripts indépendants ou via des jobs GitHub Actions (`workflow_dispatch`), sans impacter le processus d'inférence en ligne.           |

### Facteurs à pertinence limitée

- **Facteur VIII (Concurrency - clustering au niveau OS)** : Notre application exécutant un pipeline d'inférence léger, la concurrence interne par sous-processus complexes ou workers locaux est limitée au profit d'un dimensionnement horizontal géré directement par la plateforme conteneurisée (Cloud Autoscaling).

---

## 2. Déclencheurs de Branches & Pipeline CI/CD

Le cycle de vie du code suit un flux Git Flow rigoureux documenté comme suit :

1. **Pull Request vers `dev`, `staging` ou `main`** :
   - Exécution du linting et de l'analyse statique (`ruff`).
   - Lancement de la suite complète de tests applicatifs (`pytest`).
   - Construction de l'image Docker (`docker build`).
   - Exécution d'un _smoke test_ du conteneur pour valider le démarrage et la réponse de `/ready`.
   - Règle de protection : merge bloqué si un seul check échoue.

2. **Merge vers `staging`** :
   - Validation de la version candidate (_release candidate_) dans un environnement de staging isolé.

3. **Merge vers `main`** :
   - Construction et étiquetage de l'image Docker finale avec le commit SHA (`predictmaint-ai:sha-<commit>`).
   - Déploiement automatique sur la plateforme de production en ligne (Render / Azure / Cloud Run).
   - Attente de la disponibilité de la sonde de préparation (`/ready`).
   - Exécution d'un test de fumée HTTP réel en direct sur l'URL publique de production vérifiant une prédiction et la version du modèle.

---

## 3. Procédure de Reproduction du Déploiement

### Reproduction en Local (avec Docker)

1. **Construction de l'image Docker de production** :
   ```bash
   docker build -t predictmaint-ai:local .
   ```
2. **Exécution du conteneur en mode test (Mock activé)** :
   ```bash
   docker run -d -p 8000:8000 -e ENABLE_MOCK_MODEL=true --name predictmaint predictmaint-ai:local
   ```
3. **Validation de la disponibilité (Smoke test local)** :
   ```bash
   curl -f http://localhost:8000/ready
   curl -f http://localhost:8000/model-info
   ```
4. **Arrêt et nettoyage du conteneur** :
   ```bash
   docker stop predictmaint && docker rm predictmaint
   ```

### Reproduction en Production (sur le Cloud)

1. Configurer les variables d'environnement secrètes sur la plateforme cible (Render / Cloud Run) :
   - `MLFLOW_TRACKING_URI` : `https://dagshub.com/Abdel380/predictmaint-ai.mlflow`
   - `MLFLOW_TRACKING_USERNAME` : `<votre_username_dagshub>`
   - `MLFLOW_TRACKING_PASSWORD` : `<votre_token_dagshub>`
   - `MLFLOW_MODEL_NAME` : `predictmaint-ai`
   - `MLFLOW_MODEL_ALIAS` : `champion`
2. Déployer l'image étiquetée avec le Git SHA.
3. Vérifier le bon chargement du modèle champion via l'URL publique :
   ```bash
   curl -f https://<votre-app-url>/ready
   ```
