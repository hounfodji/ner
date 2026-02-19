#!/usr/bin/env python3
"""
🧪 Client de test pour l'API NER Transport Béninois
=====================================================
Usage:
  1. Lancer l'API :  python api.py
  2. Dans un autre terminal :  python test_api.py
"""

import requests
import json

API_URL = "http://localhost:8000"


def test_extract(text, threshold=None, entity_types=None):
    """Appelle l'endpoint d'extraction et affiche les résultats."""
    payload = {"text": text}
    if threshold is not None:
        payload["threshold"] = threshold
    if entity_types is not None:
        payload["entity_types"] = entity_types

    print(f'\n📝 "{text}"')
    try:
        resp = requests.post(f"{API_URL}/api/v1/ner/extract", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for etype, mentions in data["entities"].items():
            print(f"   • {etype:12} : {mentions}")
        if not data["entities"]:
            print("   (Aucune entité détectée)")
        print(f"   ⏱ {data['processing_time_ms']}ms")
    except requests.exceptions.ConnectionError:
        print("   ❌ Impossible de se connecter. L'API tourne-t-elle sur le port 8000 ?")
    except requests.exceptions.HTTPError as e:
        print(f"   ❌ Erreur HTTP {resp.status_code}: {resp.text}")
    print("-" * 60)


def main():
    print("=" * 60)
    print("🧪 TEST DE L'API NER")
    print("=" * 60)

    # Health check
    print("\n🏥 Health check...")
    try:
        resp = requests.get(f"{API_URL}/api/v1/ner/health", timeout=5)
        print(f"   {resp.json()}")
    except requests.exceptions.ConnectionError:
        print("   ❌ API non disponible. Lance d'abord : python api.py")
        return

    # Entity types
    print("\n📋 Types d'entités disponibles :")
    resp = requests.get(f"{API_URL}/api/v1/ner/entity-types", timeout=5)
    for et in resp.json()["entity_types"]:
        print(f"   • {et['name']:12} — {et['description']}")

    # Tests d'extraction
    print("\n" + "=" * 60)
    print("🔍 Tests d'extraction")
    print("=" * 60)

    test_phrases = [
        "Je veux aller de Cotonou à Parakou demain à 8h avec 2 adultes",
        "Un billet pour Abomey, je pars d'ici",
        "Direction Natitingou en passant par Djougou et Parakou si possible",
        "Je pars de Calavi",
        "Cotonou-Parakou aller-retour pour 2 adultes et 1 enfant",
        "Je veux pas aller à Porto-Novo finalement, plutôt Ouidah",
        "Gare de Jonquet vers la frontière nigériane, 3 personnes",
        "Le car qui part à 6h du matin pour le nord",
        "PK10 jusqu'à Godomey, après on verra",
        "Voyage d'affaires à Porto-Novo le 15 mars",
    ]

    for phrase in test_phrases:
        test_extract(phrase)

    # Test avec seuil personnalisé
    print("\n🔧 Test avec seuil élevé (0.7) :")
    test_extract("Je veux aller de Cotonou à Parakou demain", threshold=0.7)

    # Test avec types d'entités spécifiques
    print("🔧 Test avec types spécifiques (Departure, Destination) :")
    test_extract(
        "Je veux aller de Cotonou à Parakou demain à 8h",
        entity_types=["Departure", "Destination"],
    )

    # Test d'erreur : texte vide
    print("\n⚠️ Test d'erreur (texte vide) :")
    try:
        resp = requests.post(f"{API_URL}/api/v1/ner/extract", json={"text": ""}, timeout=5)
        print(f"   Status: {resp.status_code} — {resp.json()}")
    except Exception as e:
        print(f"   {e}")

    print("\n✅ Tests terminés !")
    print(f"📡 Documentation Swagger : {API_URL}/docs")


if __name__ == "__main__":
    main()
