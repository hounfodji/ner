#!/usr/bin/env python3
"""
🚀 API FastAPI — Extraction d'entités NER pour le transport béninois
====================================================================
Groupe 4 : Hospice Hounfodji, Juste Hodonou

Endpoints:
  POST /api/v1/ner/extract    → Extraire les entités d'une phrase
  GET  /api/v1/ner/health     → Vérifier que l'API fonctionne
  GET  /api/v1/ner/entity-types → Lister les types d'entités supportés

Lancer le serveur:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Ou directement:
  python api.py
"""

import os
import re
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_BASE = "fastino/gliner2-multi-v1"
DEFAULT_THRESHOLD = float(os.environ.get("NER_THRESHOLD", "0.5"))
MAX_TEXT_LENGTH = 500
HOST = os.environ.get("NER_HOST", "0.0.0.0")
PORT = int(os.environ.get("NER_PORT", "8000"))

ALL_ENTITY_TYPES = [
    "Departure", "Destination", "Via", "Time",
    "Date", "Passengers", "TripType", "Purpose",
]

# Villes béninoises connues (pour distinguer "Cotonou-Parakou" de "Porto-Novo")
BENIN_CITIES = {
    "cotonou", "parakou", "abomey", "porto-novo", "ouidah", "djougou",
    "natitingou", "bohicon", "kandi", "lokossa", "malanville", "nikki",
    "savalou", "dassa", "savè", "godomey", "calavi", "jonquet",
    "abomey-calavi", "comè", "dogbo", "allada", "azovè",
    "banikoara", "bassila", "bembèrèkè", "bétérou", "cobly", "covè",
    "glazoué", "grand-popo", "kétou", "kouandé", "pobè", "sakété",
    "tchaourou", "tanguiéta",
}

# Pattern : Mot1-Mot2 ou Mot1/Mot2 (mots capitalisés)
ROUTE_PATTERN = re.compile(
    r'\b([A-ZÀ-Ü][a-zà-ü]+(?:-[A-ZÀ-Ü][a-zà-ü]+)*)'
    r'\s*[-/]\s*'
    r'([A-ZÀ-Ü][a-zà-ü]+(?:-[A-ZÀ-Ü][a-zà-ü]+)*)\b',
    re.UNICODE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# POST-PROCESSING (filtrage des hallucinations)
# ============================================================

TYPE_PRIORITY = {
    "Departure": 1, "Destination": 2, "Via": 3,
    "Passengers": 4, "Time": 5, "Date": 6,
    "TripType": 7, "Purpose": 8,
}

VALID_TRIP_TYPES = {"aller", "retour", "aller-retour", "aller simple"}

INVALID_TRIPTYPE = {
    "je", "tu", "il", "on", "nous", "vous", "ils",
    "pars", "part", "voyage", "billet", "verra", "va",
    "prends", "cherche", "veux", "dois", "faut",
}

INVALID_PURPOSE_PATTERNS = [
    "frontière", "gare", "aéroport", "marché",
    "adulte", "enfant", "personne", "bébé",
    "si possible", "verra", "après", "on",
]

INVALID_DEPARTURE = {
    "mon patron", "ma mère", "mon père", "le chauffeur",
    "mon ami", "quelqu'un",
    "ici", "là", "là-bas", "d'ici", "quelque part",
}

INVALID_DATE_PATTERNS = [
    "cotonou", "parakou", "abomey", "porto-novo", "ouidah", "djougou",
    "natitingou", "bohicon", "kandi", "lokossa", "malanville", "nikki",
    "savalou", "dassa", "savè", "godomey", "calavi", "jonquet",
]


def post_process(entities_dict: dict) -> dict:
    """
    Filtre les faux positifs et déduplique les entités.

    Args:
        entities_dict: {type: [(mention, score), ...]}

    Returns:
        dict nettoyé {type: [mention, ...]}
    """
    # --- Étape 1 : Filtrer les spans invalides ---
    filtered = {}
    for etype, mentions in entities_dict.items():
        valid = []
        for mention, score in mentions:
            m_lower = mention.lower().strip()

            if etype == "TripType":
                if m_lower in INVALID_TRIPTYPE:
                    continue
                if m_lower not in VALID_TRIP_TYPES:
                    continue

            if etype == "Purpose":
                if any(p in m_lower for p in INVALID_PURPOSE_PATTERNS):
                    continue
                if len(m_lower) < 3:
                    continue

            if etype == "Departure":
                if m_lower in INVALID_DEPARTURE:
                    continue

            if etype == "Date":
                if any(c in m_lower for c in INVALID_DATE_PATTERNS):
                    continue

            valid.append((mention, score))

        if valid:
            filtered[etype] = valid

    # --- Étape 2 : Déduplication inter-types ---
    all_spans = {}
    for etype, mentions in filtered.items():
        for mention, score in mentions:
            key = mention.lower()
            if key not in all_spans:
                all_spans[key] = []
            all_spans[key].append((etype, mention, score))

    deduplicated = {}
    for span_key, entries in all_spans.items():
        if len(entries) > 1:
            best = min(entries, key=lambda x: (TYPE_PRIORITY.get(x[0], 99), -x[2]))
            etype, mention, _ = best
            deduplicated.setdefault(etype, []).append(mention)
        else:
            etype, mention, _ = entries[0]
            deduplicated.setdefault(etype, []).append(mention)

    return deduplicated


# ============================================================
# CHARGEMENT DU MODÈLE (au démarrage)
# ============================================================

model = None


def load_model():
    """Charge le modèle GLiNER2 + adapter LoRA."""
    global model

    logger.info("📦 Chargement du modèle de base : %s", MODEL_BASE)
    from gliner2 import GLiNER2
    from gliner2.training.lora import load_lora_adapter

    model = GLiNER2.from_pretrained(MODEL_BASE)

    if os.path.exists(ADAPTER_DIR):
        logger.info("🔌 Chargement de l'adapter LoRA : %s", ADAPTER_DIR)
        load_lora_adapter(model, ADAPTER_DIR)
        logger.info("✅ Modèle + adapter chargés avec succès")
    else:
        logger.warning("⚠️  Adapter introuvable (%s), utilisation du modèle de base", ADAPTER_DIR)

    return model


def extract_entities(text: str, threshold: float, entity_types: list[str]) -> dict:
    schema = model.create_schema().entities(entity_types)
    result = model.extract(text, schema)

    # DEBUG — à supprimer après diagnostic
    logger.info("RAW TYPE: %s", type(result))
    logger.info("RAW KEYS: %s", result.keys() if isinstance(result, dict) else "NOT DICT")
    logger.info("RAW DATA: %s", str(result)[:500])

    raw = {}
    for etype, mentions_data in result["entities"].items():
        if not mentions_data:
            continue
        processed = []
        for item in mentions_data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                processed.append((item[0], item[1]))
            else:
                processed.append((str(item), 1.0))
        raw[etype] = processed

    logger.info("PARSED: %s", raw)
    return post_process(raw)


# ============================================================
# SCHEMAS PYDANTIC
# ============================================================

class NERRequest(BaseModel):
    """Corps de la requête d'extraction."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Phrase en français à analyser",
        examples=["Je veux aller de Cotonou à Parakou demain à 8h avec 2 adultes"],
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Seuil de confiance (0-1). Défaut : 0.5",
    )
    entity_types: Optional[list[str]] = Field(
        default=None,
        description="Types d'entités à extraire. Défaut : tous les types",
        examples=[["Departure", "Destination", "Passengers"]],
    )


class NERResponse(BaseModel):
    """Réponse de l'extraction."""
    text: str = Field(description="Phrase analysée")
    entities: dict[str, list[str]] = Field(description="Entités extraites par type")
    processing_time_ms: float = Field(description="Temps de traitement en millisecondes")


class HealthResponse(BaseModel):
    """Réponse du health check."""
    status: str
    model: str
    adapter: str
    default_threshold: float


class EntityTypesResponse(BaseModel):
    """Liste des types d'entités."""
    entity_types: list[dict[str, str]]


class ErrorResponse(BaseModel):
    """Réponse d'erreur."""
    error: str
    code: int


# ============================================================
# APPLICATION FASTAPI
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, libère à l'arrêt."""
    logger.info("🚀 Démarrage de l'API NER Transport Béninois")
    load_model()
    yield
    logger.info("🛑 Arrêt de l'API")


app = FastAPI(
    title="API NER — Transport Béninois",
    description=(
        "Extraction d'entités nommées (NER) pour les requêtes de transport au Bénin.\n\n"
        "**Groupe 4** : Hospice Hounfodji, Juste Hodonou\n\n"
        "Modèle : GLiNER2 (fastino/gliner2-multi-v1) fine-tuné avec LoRA\n\n"
        "Entités supportées : Departure, Destination, Via, Time, Date, Passengers, TripType, Purpose"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — autoriser tous les origines pour le dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.post(
    "/api/v1/ner/extract",
    response_model=NERResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Requête invalide"},
        422: {"model": ErrorResponse, "description": "Texte trop long"},
        500: {"model": ErrorResponse, "description": "Erreur serveur"},
    },
    summary="Extraire les entités d'une phrase",
    description=(
        "Analyse une phrase en français et extrait les entités liées au transport.\n\n"
        "**Exemple :**\n"
        "```json\n"
        '{"text": "Je veux aller de Cotonou à Parakou demain à 8h"}\n'
        "```"
    ),
)
async def extract(request: NERRequest):
    """
    Endpoint principal : extraction d'entités NER.

    - **text** : phrase en français (obligatoire, max 500 caractères)
    - **threshold** : seuil de confiance entre 0 et 1 (optionnel, défaut 0.5)
    - **entity_types** : sous-ensemble de types à extraire (optionnel, défaut tous)
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    # Valider les types d'entités demandés
    entity_types = request.entity_types or ALL_ENTITY_TYPES
    invalid_types = [t for t in entity_types if t not in ALL_ENTITY_TYPES]
    if invalid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Types d'entités invalides : {invalid_types}. "
                   f"Types supportés : {ALL_ENTITY_TYPES}",
        )

    threshold = request.threshold if request.threshold is not None else DEFAULT_THRESHOLD

    try:
        start = time.perf_counter()
        entities = extract_entities(request.text, threshold, entity_types)
        elapsed_ms = (time.perf_counter() - start) * 1000
    except Exception as e:
        logger.error("Erreur extraction : %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur du modèle : {str(e)}")

    return NERResponse(
        text=request.text,
        entities=entities,
        processing_time_ms=round(elapsed_ms, 1),
    )


@app.get(
    "/api/v1/ner/health",
    response_model=HealthResponse,
    summary="Vérifier l'état de l'API",
)
async def health():
    """Retourne l'état du serveur et du modèle."""
    return HealthResponse(
        status="ok" if model is not None else "model_not_loaded",
        model=MODEL_BASE,
        adapter=ADAPTER_DIR if os.path.exists(ADAPTER_DIR) else "non chargé",
        default_threshold=DEFAULT_THRESHOLD,
    )


@app.get(
    "/api/v1/ner/entity-types",
    response_model=EntityTypesResponse,
    summary="Lister les types d'entités supportés",
)
async def entity_types():
    """Retourne la liste complète des types d'entités avec descriptions."""
    descriptions = {
        "Departure": "Lieu de départ",
        "Destination": "Lieu d'arrivée",
        "Via": "Lieu(x) de passage intermédiaire",
        "Time": "Heure de départ ou d'arrivée",
        "Date": "Date du voyage",
        "Passengers": "Nombre et type de passagers",
        "TripType": "Type de trajet (aller, retour, aller-retour)",
        "Purpose": "Motif du voyage (affaires, tourisme, etc.)",
    }
    return EntityTypesResponse(
        entity_types=[
            {"name": t, "description": descriptions.get(t, "")}
            for t in ALL_ENTITY_TYPES
        ]
    )


# ============================================================
# LANCEMENT DIRECT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("🚀 API NER Transport Béninois")
    logger.info("=" * 60)
    logger.info("📡 Swagger UI : http://%s:%d/docs", HOST, PORT)
    logger.info("📡 ReDoc      : http://%s:%d/redoc", HOST, PORT)
    logger.info("=" * 60)

    uvicorn.run(
        "api:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
