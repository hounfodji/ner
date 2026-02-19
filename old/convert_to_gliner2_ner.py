#!/usr/bin/env python3
"""
Script de conversion : Format personnalisé → Format GLiNER2 NER

Entrée : {"input": "...", "output": {"departure": "X", "destination": "Y", ...}}
Sortie : {"input": "...", "output": {"entities": {"Departure": ["X"], "Destination": ["Y"], ...}}}
"""

import json

def convert_to_gliner2_ner(input_file, output_file):
    """
    Convertit le dataset au format GLiNER2 NER.
    
    Mapping des champs :
    - departure → Departure
    - destination → Destination  
    - via → Via (liste conservée)
    - time → Time
    - date → Date
    - passengers → Passengers (converti en texte)
    - trip_type → TripType
    - purpose → Purpose
    """
    
    converted_count = 0
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                input_text = data.get('input', '')
                old_output = data.get('output', {})
                
                # Nouvelle structure pour GLiNER2
                entities = {}
                
                # Conversion champ par champ
                if old_output.get('departure') is not None:
                    entities['Departure'] = [old_output['departure']]
                
                if old_output.get('destination') is not None:
                    entities['Destination'] = [old_output['destination']]
                
                if old_output.get('via') is not None:
                    # 'via' est déjà une liste
                    entities['Via'] = old_output['via']
                
                if old_output.get('time') is not None:
                    entities['Time'] = [old_output['time']]
                
                if old_output.get('date') is not None:
                    entities['Date'] = [old_output['date']]
                
                # Passagers : convertir l'objet en liste de mentions textuelles
                if old_output.get('passengers') is not None:
                    passengers = old_output['passengers']
                    passenger_mentions = []
                    
                    if isinstance(passengers, dict):
                        if passengers.get('adults'):
                            passenger_mentions.append(f"{passengers['adults']} adultes")
                        if passengers.get('children'):
                            passenger_mentions.append(f"{passengers['children']} enfants")
                    
                    if passenger_mentions:
                        entities['Passengers'] = passenger_mentions
                
                if old_output.get('trip_type') is not None:
                    entities['TripType'] = [old_output['trip_type']]
                
                if old_output.get('purpose') is not None:
                    entities['Purpose'] = [old_output['purpose']]
                
                # Créer la nouvelle ligne au format GLiNER2
                new_data = {
                    "input": input_text,
                    "output": {
                        "entities": entities
                    }
                }
                
                # Écrire dans le fichier de sortie
                f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                converted_count += 1
                
            except json.JSONDecodeError:
                print(f"⚠️  Ligne {line_num} : JSON invalide, ignorée")
                skipped_count += 1
            except Exception as e:
                print(f"⚠️  Ligne {line_num} : Erreur {e}, ignorée")
                skipped_count += 1
    
    print(f"\n✅ Conversion terminée !")
    print(f"   • Exemples convertis : {converted_count}")
    print(f"   • Exemples ignorés    : {skipped_count}")
    print(f"   • Fichier de sortie   : {output_file}")


def validate_gliner2_format(filepath, num_samples=5):
    """Affiche quelques exemples pour vérifier le format."""
    print(f"\n📋 Aperçu des {num_samples} premiers exemples :\n")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            
            data = json.loads(line.strip())
            print(f"Exemple {i+1}:")
            print(f"  Input : {data['input'][:80]}...")
            print(f"  Entities :")
            for entity_type, mentions in data['output']['entities'].items():
                print(f"    • {entity_type:12} : {mentions}")
            print("-" * 60)


if __name__ == "__main__":
    # Fichiers
    INPUT_FILE = "train_ollama_final_corrige_5000.jsonl"
    OUTPUT_FILE = "train_gliner2_ner_5000.jsonl"
    
    # Conversion
    print("🔄 Conversion du dataset au format GLiNER2 NER...")
    convert_to_gliner2_ner(INPUT_FILE, OUTPUT_FILE)
    
    # Validation visuelle
    validate_gliner2_format(OUTPUT_FILE)
    
    print("\n✨ Vous pouvez maintenant utiliser ce fichier pour entraîner GLiNER2 !")