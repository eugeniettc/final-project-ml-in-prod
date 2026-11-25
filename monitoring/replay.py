"""
Simulateur de trafic industriel :
Envoie les observations de machines à l'API à haute cadence.

Exemples d'utilisation :
  python monitoring/replay.py                               # Trafic normal
  python monitoring/replay.py --rate 30 --limit 500         # 30 req/s sur 500 machines
  python monitoring/replay.py --broken                      # Simule des erreurs (pour tester les alertes)
  python monitoring/replay.py --drift                       # Simule une surchauffe (pour tester le Data Drift)
"""

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data/splits/test.csv"

parser = argparse.ArgumentParser(description="Rejeu de données machine vers l'API.")
parser.add_argument("--file", default=str(DEFAULT_CSV), help="Chemin du CSV de test")
parser.add_argument("--url", default="http://localhost:8000/predict", help="URL du endpoint /predict")
parser.add_argument("--rate", type=float, default=20.0, help="Nombre de requêtes par seconde (défaut: 20)")
parser.add_argument("--limit", type=int, default=500, help="Nombre total de lignes à envoyer (défaut: 500)")
parser.add_argument("--shuffle", action="store_true", default=True, help="Mélanger l'ordre des machines")
parser.add_argument("--broken", action="store_true", help="Génère des requêtes invalides pour tester les alertes d'erreurs")
parser.add_argument("--drift", action="store_true", help="Simule une surchauffe pour faire grimper la dérive (Data Drift > 0.20)")

args = parser.parse_args()

# 1. Chargement des données
df = pd.read_csv(args.file)

if args.shuffle:
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

df = df.head(args.limit)

print(f"\n Démarrage de la simulation : {len(df)} requêtes à un rythme de {args.rate} req/s...")
if args.broken:
    print(" Mode BROKEN activé : envoi de données erronées pour tester les alertes d'erreurs.")
if args.drift:
    print(" Mode DRIFT activé : injection de surchauffe (340 K) pour faire réagir le radar de dérive.")

statuses = Counter()
start_time = time.time()

# 2. Envoi des requêtes une par une
for idx, row in df.iterrows():
    # Mapping vers les champs JSON attendus par l'API
    payload = {
        "type": str(row.get("Type", "L")),
        "air_temperature": float(row.get("Air temperature", 300.0)),
        "process_temperature": float(row.get("Process temperature", 310.0)),
        "rotational_speed": float(row.get("Rotational speed", 1500.0)),
        "torque": float(row.get("Torque", 40.0)),
        "tool_wear": float(row.get("Tool wear", 50.0))
    }

    # Simulation d'erreur
    if args.broken:
        payload["type"] = "TYPE_INVALIDE"  # Provoque une erreur 422
        payload["air_temperature"] = -50.0

    # Simulation de dérive (surchauffe machine)
    if args.drift:
        payload["air_temperature"] = 345.0  # +45 K au-dessus de la normale
        payload["process_temperature"] = 355.0
        payload["torque"] = 75.0

    # Requête HTTP avec X-Request-ID unique
    req = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": uuid.uuid4().hex[:12]
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            statuses[response.status] += 1
    except urllib.error.HTTPError as err:
        statuses[err.code] += 1
    except urllib.error.URLError:
        statuses["unreachable"] += 1

    time.sleep(1.0 / args.rate)

duration = round(time.time() - start_time, 2)
print(f"\n Simulation terminée en {duration}s !")
print(" Bilan des réponses HTTP reçues :", dict(statuses))