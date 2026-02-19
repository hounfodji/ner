#!/usr/bin/env python3
"""
Test du modèle GLiNER2 fine-tuné pour le transport béninois.
"""

from gliner2 import GLiNER2
from gliner2.training.lora import load_lora_adapter

MODEL_DIR = "./benin_transport_ner_model"
ENTITY_TYPES = ["Departure", "Destination", "Via", "Time", "Date", "Passengers", "TripType", "Purpose"]

test_queries = [
    "Je veux aller de Cotonou à Parakou",
    "Un billet pour Abomey, je pars d'ici",
    "Direction Natitingou en passant par Djougou et Parakou si possible",
    "Je pars de Calavi",
    "Cotonou-Parakou aller-retour pour 2 adultes et 1 enfant",
    "Je veux pas aller à Porto-Novo finalement, plutôt Ouidah",
    "Gare de Jonquet vers la frontière nigériane",
    "Le car qui part à 6h du matin pour le nord",
    "Mon patron m'envoie chercher un colis à Parakou, retour le jour même, après Ouidah",
    "PK10 jusqu'à Godomey, après on verra",
]

print("=" * 60)
print("🧪 TEST DU MODÈLE FINE-TUNÉ - TRANSPORT BÉNINOIS")
print("=" * 60)

print("\n📦 Chargement du modèle de base + adapter LoRA...")
model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
load_lora_adapter(model, f"{MODEL_DIR}/best")
print("   ✅ Modèle chargé")

print("\n🔍 Extraction d'entités:")
print("-" * 60)

for query in test_queries:
    schema = model.create_schema().entities(ENTITY_TYPES)
    result = model.extract(query, schema)

    print(f"\n📝 \"{query}\"")

    entities_found = False
    for entity_type, mentions in result["entities"].items():
        if mentions:
            entities_found = True
            print(f"   • {entity_type:12} : {mentions}")

    if not entities_found:
        print("   (Aucune entité détectée)")

    print("-" * 60)
