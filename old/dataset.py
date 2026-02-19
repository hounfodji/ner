import json
import requests
import time

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gpt-oss:latest" 

REF_FILE = "recherches_ner.md"
OUTPUT_FILE = "train_ollama_final_5000.jsonl"
TOTAL_SAMPLES = 5000
BATCH_SIZE = 10 

# --- EXEMPLES FEW-SHOT (VOS RÉFÉRENCES N01-N10) ---
FEW_SHOT_EXAMPLES = """
{"input": "Je veux aller de Cotonou à Parakou", "output": {"departure": "Cotonou", "destination": "Parakou", "via": null, "time": null, "date": null, "passengers": null, "trip_type": null, "purpose": null}}
{"input": "Un billet pour Abomey, je pars d'ici", "output": {"departure": "Cotonou", "destination": "Abomey", "via": null, "time": null, "date": null, "passengers": null, "trip_type": null, "purpose": null}}
{"input": "Direction Natitingou en passant par Djougou et Parakou si possible", "output": {"departure": null, "destination": "Natitingou", "via": ["Djougou", "Parakou"], "time": null, "date": null, "passengers": null, "trip_type": null, "purpose": null}}
{"input": "Cotonou-Parakou aller-retour pour 2 adultes et 1 enfant", "output": {"departure": "Cotonou", "destination": "Parakou", "via": null, "time": null, "date": null, "passengers": {"adults": 2, "children": 1}, "trip_type": "aller-retour", "purpose": null}}
{"input": "Je veux pas aller à Porto-Novo finalement, plutôt Ouidah", "output": {"departure": "Porto-Novo", "destination": "Ouidah", "via": null, "time": null, "date": null, "passengers": null, "trip_type": null, "purpose": null}}
{"input": "Gare de Jonquet vers la frontière nigériane", "output": {"departure": "Gare de Jonquet", "destination": "Frontière nigériane", "via": null, "time": null, "date": null, "passengers": null, "trip_type": null, "purpose": null}}
{"input": "Le car qui part à 6h du matin pour le nord", "output": {"departure": null, "destination": "Nord du Bénin", "via": null, "time": "6h du matin", "date": null, "passengers": null, "trip_type": null, "purpose": null}}
{"input": "PK10 jusqu'à Godomey, après on verra", "output": {"departure": "PK10", "destination": "Godomey", "via": null, "time": null, "date": null, "passengers": null, "trip_type": null, "purpose": null}}
"""

def load_and_chunk_reference(filepath, chunk_size=2500):
    """Découpe le document en morceaux pour éviter de saturer le contexte d'Ollama."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        # On découpe par blocs de caractères pour que chaque appel soit rapide
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    except FileNotFoundError:
        print(f"Erreur: {filepath} introuvable.")
        return []

def generate_with_ollama(context_chunk):
    prompt = f"""
Tu es un expert en annotation de données pour le transport au Bénin.
Utilise ce référentiel géographique PARTIEL :
---
{context_chunk}
---

Tâche : Génère {BATCH_SIZE} exemples uniques de requêtes de transport au Bénin au format JSONL.

Règles de Nuance :
1. Incomplétude : 50% des phrases doivent avoir des valeurs `null` pour plusieurs entités.
2. Complexité : Utilise des objets pour `passengers` (ex: {{"adults": 2}}) et des listes pour `via`.
3. Réalité Terrain : Utilise le langage familier et les points de repère du référentiel.
4. Logique Négative : Inclus des changements d'avis ("finalement") ou des destinations vagues.

Exemples Few-Shot à imiter :
{FEW_SHOT_EXAMPLES}

Génère uniquement le JSONL, un objet par ligne.
"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 4096, 
            "temperature": 0.85
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        return response.json().get('response', '')
    except Exception as e:
        print(f"Erreur lors de l'appel Ollama : {e}")
        return ""

def main():
    chunks = load_and_chunk_reference(REF_FILE)
    if not chunks: return

    count = 0
    print(f"Démarrage de la génération de {TOTAL_SAMPLES} lignes...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        while count < TOTAL_SAMPLES:
            # On change de morceau de document à chaque lot pour varier les lieux
            current_chunk = chunks[(count // BATCH_SIZE) % len(chunks)]
            
            content = generate_with_ollama(current_chunk)
            
            if content:
                lines = content.replace("```jsonl", "").replace("```json", "").replace("```", "").strip().split('\n')
                batch_added = 0
                for line in lines:
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            json.loads(line) # Validation
                            f.write(line + "\n")
                            count += 1
                            batch_added += 1
                        except: continue
                
                if batch_added > 0:
                    print(f"Progression : {count}/{TOTAL_SAMPLES}")
                else:
                    print("Lot vide ou mal formé, nouvelle tentative...")
            
            time.sleep(0.1)

    print(f"Terminé ! {count} exemples sauvegardés dans {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
