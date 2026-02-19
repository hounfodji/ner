import json
import requests
import time
import re

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gpt-oss:latest" 

REF_FILE = "recherches_ner.md"
OUTPUT_FILE = "train_gliner2_ner_clean_5000.jsonl"
TOTAL_SAMPLES = 5000
BATCH_SIZE = 10 

# --- EXEMPLES FEW-SHOT CORRIGÉS ---
FEW_SHOT_EXAMPLES = """
{"input": "Je veux aller de Cotonou à Parakou", "output": {"entities": {"Departure": ["Cotonou"], "Destination": ["Parakou"]}}}
{"input": "Un billet pour Abomey, je pars d'ici", "output": {"entities": {"Destination": ["Abomey"]}}}
{"input": "Direction Natitingou en passant par Djougou et Parakou si possible", "output": {"entities": {"Destination": ["Natitingou"], "Via": ["Djougou", "Parakou"]}}}
{"input": "Cotonou-Parakou aller-retour pour 2 adultes et 1 enfant", "output": {"entities": {"Departure": ["Cotonou"], "Destination": ["Parakou"], "TripType": ["aller-retour"], "Passengers": ["2 adultes", "1 enfant"]}}}
{"input": "Je ne veux plus aller à Porto-Novo, je préfère Ouidah", "output": {"entities": {"Destination": ["Ouidah"]}}}
{"input": "Gare de Jonquet vers la frontière nigériane", "output": {"entities": {"Departure": ["Gare de Jonquet"], "Destination": ["frontière nigériane"]}}}
{"input": "Le car qui part à 6h du matin pour le nord", "output": {"entities": {"Time": ["6h du matin"], "Destination": ["nord du Bénin"]}}}
{"input": "PK10 jusqu'à Godomey, après on verra", "output": {"entities": {"Departure": ["PK10"], "Destination": ["Godomey"]}}}
{"input": "Je pars demain matin à 5h avec 3 personnes", "output": {"entities": {"Date": ["demain matin"], "Time": ["5h"], "Passengers": ["3 adultes"]}}}
{"input": "Voyage d'affaires à Parakou le 15 mars", "output": {"entities": {"Destination": ["Parakou"], "Date": ["15 mars"], "Purpose": ["affaires"]}}}
{"input": "De Cotonou à Abomey via Dantokpa avec 4 adultes et 2 enfants", "output": {"entities": {"Departure": ["Cotonou"], "Destination": ["Abomey"], "Via": ["Dantokpa"], "Passengers": ["4 adultes", "2 enfants"]}}}
"""

def load_and_chunk_reference(filepath, chunk_size=2500):
    """Découpe le document en morceaux pour éviter de saturer le contexte d'Ollama."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    except FileNotFoundError:
        print(f"⚠️  Erreur: {filepath} introuvable.")
        return [""]

def generate_with_ollama(context_chunk):
    prompt = f"""
Tu es un expert en annotation NER pour le transport au Bénin.
Utilise ce référentiel géographique PARTIEL :
---
{context_chunk}
---

Tâche : Génère {BATCH_SIZE} exemples RÉALISTES de requêtes de transport au Bénin au format JSONL GLiNER2.

Format de sortie STRICT :
{{"input": "phrase en français", "output": {{"entities": {{"EntityType": ["mention1", "mention2"]}}}}}}

Entités disponibles :
- Departure : lieu de départ explicite
- Destination : lieu d'arrivée final
- Via : lieux de passage (liste)
- Time : heure de départ/arrivée (format: "Xh", "Xh du matin/soir", "XhXX")
- Date : date du voyage (jour, date précise, expression temporelle)
- Passengers : mentions de passagers (IMPORTANT: "1 adulte", "2 adultes", "1 enfant", "3 enfants")
- TripType : type de trajet ("aller", "aller-retour", "retour")
- Purpose : motif du voyage ("affaires", "visite familiale", "tourisme", etc.)

RÈGLES CRITIQUES - RESPECTE-LES ABSOLUMENT :

1. SINGULIER/PLURIEL CORRECT :
   ✅ "1 adulte" (PAS "1 adultes")
   ✅ "2 adultes"
   ✅ "1 enfant" (PAS "1 enfants")
   ✅ "3 enfants"

2. LOGIQUE DE CHANGEMENT D'AVIS :
   ❌ "Je ne veux plus aller à X, je vais à Y" → {{"Departure": ["X"], "Destination": ["Y"]}}
   ✅ "Je ne veux plus aller à X, je vais à Y" → {{"Destination": ["Y"]}}
   (X n'est PAS un départ, c'est une destination annulée)

3. TIME vs DURÉE :
   ✅ "Time": ["6h du matin"] (heure de départ)
   ❌ "Time": ["2h"] (c'est une durée, pas une heure)

4. DESTINATION = LIEU FINAL :
   - Si "je reste à Cotonou" → PAS DE TRAJET, ne génère pas cet exemple
   - Si "je vais au nord" → {{"Destination": ["nord du Bénin"]}}

5. INCOMPLÉTUDE RÉALISTE :
   - 40% des exemples : 1-2 entités seulement
   - 30% des exemples : 3-4 entités
   - 30% des exemples : 5+ entités

6. VARIÉTÉ DE FORMULATION :
   - Mélange langage familier béninois et formel
   - Utilise les points de repère du référentiel
   - Varie les structures de phrases

Exemples Few-Shot (FORMAT EXACT À RESPECTER) :
{FEW_SHOT_EXAMPLES}

GÉNÈRE UNIQUEMENT LE JSONL, UN OBJET PAR LIGNE.
NE PRODUIS AUCUN TEXTE AVANT OU APRÈS.
"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 4096, 
            "temperature": 0.8  # Légèrement réduit pour plus de cohérence
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        return response.json().get('response', '')
    except Exception as e:
        print(f"⚠️  Erreur lors de l'appel Ollama : {e}")
        return ""

def validate_passengers_grammar(passengers_list):
    """
    Valide la grammaire singulier/pluriel dans Passengers.
    Retourne True si correct, False sinon.
    """
    for passenger in passengers_list:
        # Vérifications strictes
        if re.match(r"^1 (adultes|enfants)", passenger):
            return False  # "1 adultes" ou "1 enfants" est INCORRECT
        if re.match(r"^[2-9]\d* (adulte|enfant)$", passenger):
            return False  # "2 adulte" ou "3 enfant" est INCORRECT
    return True

def validate_time_format(time_list):
    """
    Valide que Time contient bien des heures, pas des durées.
    """
    for time_str in time_list:
        # Rejeter les durées type "2h", "3h30"
        if re.match(r"^\d+h(\d+)?$", time_str) and not any(word in time_str.lower() for word in ["matin", "soir", "après-midi", "midi"]):
            # Si c'est juste "2h" sans contexte, c'est probablement une durée
            if int(re.match(r"^(\d+)", time_str).group(1)) < 4:
                return False
    return True

def validate_logic(data):
    """
    Valide la logique sémantique de l'exemple.
    """
    input_text = data.get("input", "").lower()
    entities = data.get("output", {}).get("entities", {})
    
    # Cas 1: "je reste à X" → pas de trajet
    if "reste" in input_text or "rester" in input_text:
        return False
    
    # Cas 2: Departure == Destination (sauf cas particuliers)
    departure = entities.get("Departure", [])
    destination = entities.get("Destination", [])
    if departure and destination and departure == destination:
        # Sauf si c'est un aller-retour explicite
        if "aller-retour" not in entities.get("TripType", []):
            return False
    
    return True

def validate_gliner2_format(line_data):
    """Vérifie que la ligne respecte le format GLiNER2 NER avec validation stricte."""
    try:
        # Structure de base
        if 'input' not in line_data or 'output' not in line_data:
            return False, "Manque 'input' ou 'output'"
        
        if 'entities' not in line_data['output']:
            return False, "Manque 'entities'"
        
        entities = line_data['output']['entities']
        if not isinstance(entities, dict):
            return False, "'entities' n'est pas un dict"
        
        # Vérifier que chaque entité est une liste
        for entity_type, mentions in entities.items():
            if not isinstance(mentions, list):
                return False, f"{entity_type} n'est pas une liste"
        
        # Validation grammaticale des Passengers
        if "Passengers" in entities:
            if not validate_passengers_grammar(entities["Passengers"]):
                return False, "Erreur singulier/pluriel dans Passengers"
        
        # Validation du format Time
        if "Time" in entities:
            if not validate_time_format(entities["Time"]):
                return False, "Time contient une durée au lieu d'une heure"
        
        # Validation de la logique
        if not validate_logic(line_data):
            return False, "Logique sémantique incohérente"
        
        return True, "OK"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def main():
    chunks = load_and_chunk_reference(REF_FILE)
    if not chunks or chunks == [""]: 
        print("⚠️  Aucun référentiel chargé, utilisation sans contexte géographique")
        chunks = [""]

    count = 0
    rejected_count = 0
    rejection_reasons = {}
    
    print(f"🚀 Démarrage de la génération de {TOTAL_SAMPLES} lignes au format GLiNER2 NER...")
    print(f"   Validation stricte activée (grammaire + logique)")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        while count < TOTAL_SAMPLES:
            # Rotation des chunks
            current_chunk = chunks[(count // BATCH_SIZE) % len(chunks)]
            
            content = generate_with_ollama(current_chunk)
            
            if content:
                lines = content.replace("```jsonl", "").replace("```json", "").replace("```", "").strip().split('\n')
                batch_added = 0
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            data = json.loads(line)
                            
                            # Validation stricte
                            is_valid, reason = validate_gliner2_format(data)
                            
                            if is_valid:
                                f.write(line + "\n")
                                count += 1
                                batch_added += 1
                            else:
                                rejected_count += 1
                                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                        except json.JSONDecodeError:
                            rejected_count += 1
                            rejection_reasons["JSON invalide"] = rejection_reasons.get("JSON invalide", 0) + 1
                
                if batch_added > 0:
                    print(f"✅ Progression : {count}/{TOTAL_SAMPLES} (+{batch_added} ajoutés, {rejected_count} rejetés)")
                else:
                    print(f"⚠️  Lot entièrement rejeté ({rejected_count} rejets cumulés)")
            
            time.sleep(0.1)

    print(f"\n🎉 Terminé ! {count} exemples sauvegardés dans {OUTPUT_FILE}")
    print(f"📊 Statistiques de validation :")
    print(f"   • Exemples valides : {count}")
    print(f"   • Exemples rejetés : {rejected_count}")
    print(f"\n📋 Raisons de rejet :")
    for reason, freq in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
        print(f"   • {reason:40} : {freq:4} fois")
    
    # Affichage d'un échantillon
    print(f"\n📋 Aperçu des 3 premiers exemples :")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            data = json.loads(line.strip())
            print(f"\nExemple {i+1}:")
            print(f"  Input: {data['input']}")
            print(f"  Entities:")
            for ent_type, mentions in data['output']['entities'].items():
                print(f"    • {ent_type:12} : {mentions}")

if __name__ == "__main__":
    main()