# NER Transport Béninois

Projet d’extraction d’entités nommées (NER) pour des requêtes de transport au Bénin, basé sur **GLiNER2** avec un fine-tuning **LoRA**.

## Objectif

Extraire automatiquement les entités suivantes dans des phrases en français :

- `Departure` : lieu de départ
- `Destination` : lieu d’arrivée
- `Via` : lieu(x) de passage
- `Time` : heure
- `Date` : date
- `Passengers` : nombre/type de passagers
- `TripType` : type de trajet (aller, retour, aller-retour)
- `Purpose` : motif du voyage

## Structure principale

- `/api.py` : API FastAPI pour l’inférence NER
- `/train_benin_transport_ner_v2.py` : script de fine-tuning
- `/train_gliner2_ner_10000_v2.jsonl` : dataset d’entraînement
- `/benin_transport_ner_model/` : checkpoints et adapter LoRA entraîné
- `/test_api.py` : client de test HTTP pour l’API

## Prérequis

- Python 3.10+ recommandé
- Dépendances API :

```bash
pip install -r requirements_api.txt
```

## Lancer l’API

### Option 1 — via Python

```bash
python api.py
```

### Option 2 — via Uvicorn

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

L’API est ensuite disponible sur :

- Swagger UI : `http://localhost:8000/docs`
- ReDoc : `http://localhost:8000/redoc`

## Endpoints

- `POST /api/v1/ner/extract` : extraction d’entités
- `GET /api/v1/ner/health` : état de l’API/modèle
- `GET /api/v1/ner/entity-types` : liste des types d’entités

### Exemple de requête

```json
{
  "text": "Je veux aller de Cotonou à Parakou demain à 8h avec 2 adultes"
}
```

## Tester l’API

Dans un autre terminal (après avoir lancé l’API) :

```bash
python test_api.py
```

## Entraîner ou réentraîner le modèle

Le script de fine-tuning utilise `fastino/gliner2-multi-v1` et applique LoRA :

```bash
python train_benin_transport_ner_v2.py
```

Le modèle/adapter est sauvegardé dans :

- `./benin_transport_ner_model/best`
- `./benin_transport_ner_model/checkpoint-*`

## Configuration utile (variables d’environnement)

- `NER_MODEL_DIR` : chemin vers l’adapter LoRA (défaut : `./benin_transport_ner_model/best`)
- `NER_THRESHOLD` : seuil de confiance par défaut (défaut : `0.5`)
- `NER_HOST` : host API (défaut : `0.0.0.0`)
- `NER_PORT` : port API (défaut : `8000`)
