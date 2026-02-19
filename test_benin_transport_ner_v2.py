#!/usr/bin/env python3
"""
Test amélioré du modèle GLiNER2 fine-tuné pour le transport béninois.

Améliorations par rapport à test_benin_transport_ner.py:
  1. Seuil de confiance configurable (threshold)
  2. Déduplication inter-types (même span → garder le type prioritaire)
  3. Filtrage des spans invalides (verbes comme TripType, etc.)
  4. Comparaison avec les résultats attendus
"""

import json
from gliner2 import GLiNER2
from gliner2.training.lora import load_lora_adapter

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "./benin_transport_ner_model"
ENTITY_TYPES = ["Departure", "Destination", "Via", "Time", "Date", "Passengers", "TripType", "Purpose"]

# Seuil de confiance — augmenter si trop de faux positifs
# GLiNER2 par défaut utilise 0.5, essaie 0.6 ou 0.7
THRESHOLD = 0.5

# ============================================================
# POST-PROCESSING
# ============================================================

# Priorité des types (plus petit = plus prioritaire)
TYPE_PRIORITY = {
    "Departure": 1, "Destination": 2, "Via": 3,
    "Passengers": 4, "Time": 5, "Date": 6,
    "TripType": 7, "Purpose": 8
}

# Mots qui ne devraient PAS être TripType
INVALID_TRIPTYPE = {
    "je", "tu", "il", "on", "nous", "vous", "ils",
    "pars", "part", "voyage", "billet", "verra", "va",
    "prends", "cherche", "veux", "dois", "faut",
}

# Mots qui ne devraient PAS être Purpose
INVALID_PURPOSE_PATTERNS = [
    # Lieux (ne pas confondre avec Purpose)
    "frontière", "gare", "aéroport", "marché",
    # Passagers
    "adulte", "enfant", "personne", "bébé",
    # Fragments sans sens
    "si possible", "verra", "après", "on",
]

# Mots qui ne devraient PAS être Departure
INVALID_DEPARTURE = {
    "mon patron", "ma mère", "mon père", "le chauffeur",
    "mon ami", "quelqu'un",
}


def post_process(entities_dict):
    """
    Post-processing des entités extraites.
    
    entities_dict: dict de {type: [(mention, score), ...]}
    Retourne: dict nettoyé {type: [mention, ...]}
    """
    # 1. Filtrer les spans invalides
    filtered = {}
    for etype, mentions in entities_dict.items():
        valid_mentions = []
        for mention, score in mentions:
            m_lower = mention.lower().strip()

            # TripType: filtrer les verbes et mots non-pertinents
            if etype == "TripType":
                if m_lower in INVALID_TRIPTYPE:
                    continue
                # Garder seulement: aller, retour, aller-retour (et variantes)
                valid_tt = ["aller", "retour", "aller-retour", "aller simple"]
                if m_lower not in valid_tt:
                    continue

            # Purpose: filtrer les lieux et passagers
            if etype == "Purpose":
                skip = False
                for pattern in INVALID_PURPOSE_PATTERNS:
                    if pattern in m_lower:
                        skip = True
                        break
                if skip:
                    continue
                # Purpose trop court (1-2 chars) → probablement bruit
                if len(m_lower) < 3:
                    continue

            # Departure: filtrer les non-lieux
            if etype == "Departure":
                if m_lower in INVALID_DEPARTURE:
                    continue

            valid_mentions.append((mention, score))

        if valid_mentions:
            filtered[etype] = valid_mentions

    # 2. Déduplication inter-types: si même span dans 2 types, garder le prioritaire
    all_spans = {}  # span_lower → [(type, mention, score)]
    for etype, mentions in filtered.items():
        for mention, score in mentions:
            key = mention.lower()
            if key not in all_spans:
                all_spans[key] = []
            all_spans[key].append((etype, mention, score))

    # Pour chaque span dupliqué, garder seulement le type le plus prioritaire
    deduplicated = {}
    for span_key, entries in all_spans.items():
        if len(entries) > 1:
            # Garder le type avec la meilleure priorité (ou meilleur score si même priorité)
            best = min(entries, key=lambda x: (TYPE_PRIORITY.get(x[0], 99), -x[2]))
            for etype, mention, score in entries:
                if etype == best[0]:
                    if etype not in deduplicated:
                        deduplicated[etype] = []
                    deduplicated[etype].append(mention)
        else:
            etype, mention, score = entries[0]
            if etype not in deduplicated:
                deduplicated[etype] = []
            deduplicated[etype].append(mention)

    return deduplicated


# ============================================================
# EXEMPLES DE TEST AVEC RÉSULTATS ATTENDUS
# ============================================================

test_cases = [
    {
        "input": "Je veux aller de Cotonou à Parakou",
        "expected": {"Departure": ["Cotonou"], "Destination": ["Parakou"]}
    },
    {
        "input": "Un billet pour Abomey, je pars d'ici",
        "expected": {"Destination": ["Abomey"]}
    },
    {
        "input": "Direction Natitingou en passant par Djougou et Parakou si possible",
        "expected": {"Destination": ["Natitingou"], "Via": ["Djougou", "Parakou"]}
    },
    {
        "input": "Je pars de Calavi",
        "expected": {"Departure": ["Calavi"]}
    },
    {
        "input": "Cotonou-Parakou aller-retour pour 2 adultes et 1 enfant",
        "expected": {"Departure": ["Cotonou"], "Destination": ["Parakou"],
                     "TripType": ["aller-retour"], "Passengers": ["2 adultes", "1 enfant"]}
    },
    {
        "input": "Je veux pas aller à Porto-Novo finalement, plutôt Ouidah",
        "expected": {"Destination": ["Ouidah"]}
    },
    {
        "input": "Gare de Jonquet vers la frontière nigériane",
        "expected": {"Departure": ["Gare de Jonquet"], "Destination": ["frontière nigériane"]}
    },
    {
        "input": "Le car qui part à 6h du matin pour le nord",
        "expected": {"Destination": ["le nord"], "Time": ["6h du matin"]}
    },
    {
        "input": "Mon patron m'envoie chercher un colis à Parakou, retour le jour même, après Ouidah",
        "expected": {"Destination": ["Parakou"], "Date": ["le jour même"],
                     "Purpose": ["chercher un colis"]}
    },
    {
        "input": "PK10 jusqu'à Godomey, après on verra",
        "expected": {"Departure": ["PK10"], "Destination": ["Godomey"]}
    },
]


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("🧪 TEST AMÉLIORÉ - TRANSPORT BÉNINOIS")
    print("=" * 60)

    # Charger le modèle
    print(f"\n📦 Chargement du modèle...")
    model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
    load_lora_adapter(model, f"{MODEL_DIR}/best")
    print("   ✅ Modèle chargé")
    print(f"   Seuil de confiance: {THRESHOLD}")

    # Tester
    print(f"\n🔍 Test sur {len(test_cases)} exemples:")
    print("-" * 60)

    total_correct = 0
    total_partial = 0
    total_wrong = 0

    for tc in test_cases:
        query = tc["input"]
        expected = tc["expected"]

        # Extraction
        schema = model.create_schema().entities(ENTITY_TYPES)
        result = model.extract(query, schema, threshold=THRESHOLD)

        # Récupérer les entités avec scores
        raw_entities = {}
        for entity_type, mentions_data in result['entities'].items():
            if mentions_data:
                # GLiNER2 peut retourner des tuples (mention, score) ou juste des strings
                processed = []
                for item in mentions_data:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        processed.append((item[0], item[1]))
                    else:
                        processed.append((str(item), 1.0))
                raw_entities[entity_type] = processed

        # Post-processing
        entities = post_process(raw_entities)

        # Évaluation
        print(f"\n📝 \"{query}\"")

        # Afficher résultats
        all_types = sorted(set(list(entities.keys()) + list(expected.keys())),
                          key=lambda x: TYPE_PRIORITY.get(x, 99))

        match_status = "✅"
        for etype in all_types:
            got = set(m.lower() for m in entities.get(etype, []))
            exp = set(m.lower() for m in expected.get(etype, []))

            if got == exp:
                marker = "✅"
            elif got & exp:  # partial match
                marker = "⚠️"
                match_status = "⚠️" if match_status == "✅" else match_status
            elif etype in entities and etype not in expected:
                marker = "❌ faux positif"
                match_status = "❌"
            elif etype not in entities and etype in expected:
                marker = "❌ manquant"
                match_status = "❌"
            else:
                marker = "❌"
                match_status = "❌"

            if etype in entities:
                got_str = entities[etype]
                exp_str = expected.get(etype, [])
                if exp_str:
                    print(f"   {marker} {etype:12}: {got_str}  (attendu: {exp_str})")
                else:
                    print(f"   {marker} {etype:12}: {got_str}  (non attendu!)")
            elif etype in expected:
                print(f"   {marker} {etype:12}: []  (attendu: {expected[etype]})")

        if match_status == "✅":
            total_correct += 1
        elif match_status == "⚠️":
            total_partial += 1
        else:
            total_wrong += 1

        print(f"   → {match_status}")
        print("-" * 60)

    # Résumé
    total = len(test_cases)
    print(f"\n{'=' * 60}")
    print(f"📊 RÉSUMÉ: {total_correct}/{total} parfaits, {total_partial}/{total} partiels, {total_wrong}/{total} incorrects")
    print(f"{'=' * 60}")

    # Conseils
    if total_wrong > total // 2:
        print("\n💡 Conseils pour améliorer:")
        print("   1. Augmenter le seuil (THRESHOLD) si trop de faux positifs")
        print("   2. Réentraîner avec le dataset nettoyé v2")
        print("   3. Augmenter early_stopping_patience à 5-7")
        print("   4. Essayer un learning rate plus bas (2e-4)")


if __name__ == "__main__":
    main()
