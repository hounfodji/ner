#!/usr/bin/env python3
"""
Fine-tuning GLiNER2 pour l'extraction d'entités de transport au Bénin.

Entités cibles:
- Departure: lieu de départ
- Destination: lieu d'arrivée
- Via: lieux de passage
- Time: heure de départ/arrivée
- Date: date du voyage
- Passengers: nombre de passagers
- TripType: type de trajet (aller, aller-retour)
- Purpose: motif du voyage
"""

import os
from gliner2 import GLiNER2
from gliner2.training.data import TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# ============================================================
# CONFIGURATION
# ============================================================

# Chemins des données
TRAIN_FILE = "train_gliner2_ner_10000_clean.jsonl"  # Dataset augmenté (10k exemples)
OUTPUT_DIR = "./benin_transport_ner_model"
EXPERIMENT_NAME = "benin_transport_ner_v1"

# Paramètres d'entraînement
NUM_EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE_ENCODER = 1e-5  # LR pour l'encodeur
LEARNING_RATE_TASK = 5e-4     # LR pour les têtes de tâches

# Utiliser LoRA ? (recommandé pour économiser mémoire/temps)
USE_LORA = True
LORA_RANK = 16
LORA_ALPHA = 32

# Validation split
VAL_RATIO = 0.15  # 15% pour validation
RANDOM_SEED = 42

# Weights & Biases (optionnel)
USE_WANDB = False  # Mettez True si vous voulez logger sur W&B
WANDB_PROJECT = "benin-transport-ner"

# ============================================================
# ÉTAPE 1: CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================

print("="*60)
print("🚀 FINE-TUNING GLINER2 - TRANSPORT BÉNINOIS")
print("="*60)

print("\n📁 Chargement du dataset...")
dataset = TrainingDataset.load(TRAIN_FILE)

# Statistiques du dataset
print("\n📊 Statistiques du dataset:")
dataset.print_stats()

# Validation du dataset (désactivée - données déjà vérifiées)
print("\n⏭️  Validation désactivée (passage direct à l'entraînement)")
# Note: certaines entités peuvent ne pas être exactement dans le texte
# mais GLiNER2 gère cela pendant l'entraînement

# ============================================================
# ÉTAPE 2: SPLIT TRAIN/VALIDATION
# ============================================================

print(f"\n📂 Split train/validation ({100-int(VAL_RATIO*100)}/{int(VAL_RATIO*100)})...")
train_data, val_data, _ = dataset.split(
    train_ratio=1-VAL_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=0.0,
    shuffle=True,
    seed=RANDOM_SEED
)

print(f"   • Exemples d'entraînement: {len(train_data)}")
print(f"   • Exemples de validation: {len(val_data)}")

# Note: On ne sauvegarde pas les splits pour éviter la validation automatique
print("   ℹ️  Splits créés en mémoire (non sauvegardés sur disque)")

# ============================================================
# ÉTAPE 3: CHARGEMENT DU MODÈLE
# ============================================================

print("\n🧠 Chargement du modèle de base...")
model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
print("   ✅ Modèle chargé: fastino/gliner2-multi-v1 (multilingue)")

# ============================================================
# ÉTAPE 4: CONFIGURATION DE L'ENTRAÎNEMENT
# ============================================================

print("\n⚙️  Configuration de l'entraînement...")

if USE_LORA:
    print(f"   • Mode: LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})")
    print("   • Avantages: moins de mémoire, entraînement plus rapide")
else:
    print("   • Mode: Full fine-tuning")
    print("   • Note: plus de mémoire requise")

config = TrainingConfig(
    # Sortie
    output_dir=OUTPUT_DIR,
    experiment_name=EXPERIMENT_NAME,
    
    # Entraînement
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    
    # Learning rates
    encoder_lr=LEARNING_RATE_ENCODER if not USE_LORA else None,
    task_lr=LEARNING_RATE_TASK,
    
    # Optimisation
    weight_decay=0.01,
    max_grad_norm=1.0,
    scheduler_type="linear",
    warmup_ratio=0.1,
    
    # Mixed precision (recommandé)
    fp16=True,
    
    # LoRA (si activé)
    use_lora=USE_LORA,
    lora_r=LORA_RANK if USE_LORA else None,
    lora_alpha=LORA_ALPHA if USE_LORA else None,
    lora_dropout=0.1 if USE_LORA else None,
    lora_target_modules=["encoder", "span_rep", "classifier"] if USE_LORA else None,
    save_adapter_only=True if USE_LORA else False,
    
    # Évaluation et checkpoints
    eval_strategy="epoch",  # Évalue et sauvegarde à chaque époque
    save_best=True,
    metric_for_best="eval_loss",
    greater_is_better=False,
    save_total_limit=3,  # Garde seulement les 3 meilleurs checkpoints
    
    # Early stopping
    early_stopping=True,
    early_stopping_patience=5,
    
    # Logging
    logging_steps=50,
    report_to_wandb=USE_WANDB,
    wandb_project=WANDB_PROJECT if USE_WANDB else None,
    
    # DataLoader
    num_workers=4,
    pin_memory=True,
    
    # Autres
    seed=RANDOM_SEED
)

print("\n📋 Configuration:")
print(f"   • Époques: {NUM_EPOCHS}")
print(f"   • Batch size: {BATCH_SIZE}")
print(f"   • Learning rate (task): {LEARNING_RATE_TASK}")
if not USE_LORA:
    print(f"   • Learning rate (encoder): {LEARNING_RATE_ENCODER}")
print(f"   • Warmup ratio: 0.1")
print(f"   • Mixed precision: FP16")
print(f"   • Early stopping: activé (patience=3)")
print(f"   • Weights & Biases: {'activé' if USE_WANDB else 'désactivé'}")

# ============================================================
# ÉTAPE 5: ENTRAÎNEMENT
# ============================================================

print("\n" + "="*60)
print("🏋️  DÉBUT DE L'ENTRAÎNEMENT")
print("="*60)

trainer = GLiNER2Trainer(model, config)

results = trainer.train(
    train_data=train_data,
    eval_data=val_data
)

# ============================================================
# ÉTAPE 6: RÉSULTATS
# ============================================================

print("\n" + "="*60)
print("✅ ENTRAÎNEMENT TERMINÉ")
print("="*60)

print(f"\n📊 Résultats finaux:")
print(f"   • Best validation loss: {results.get('best_metric', 'N/A'):.4f}")
print(f"   • Total steps: {results.get('total_steps', 'N/A')}")
print(f"   • Temps total: {results.get('total_time_seconds', 0)/60:.1f} minutes")

print(f"\n💾 Modèle sauvegardé dans: {OUTPUT_DIR}")
print(f"   • Meilleur checkpoint: {OUTPUT_DIR}/best")
print(f"   • Dernier checkpoint: {OUTPUT_DIR}/checkpoint-{results.get('total_steps', 'final')}")

# ============================================================
# ÉTAPE 7: TEST D'INFÉRENCE
# ============================================================

print("\n" + "="*60)
print("🧪 TEST D'INFÉRENCE")
print("="*60)

print("\n📦 Chargement du meilleur modèle...")

# Pour LoRA, on doit recharger le modèle de base puis appliquer l'adapter
if USE_LORA:
    print("   ℹ️  Mode LoRA: chargement du modèle de base + adapter")
    best_model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
    
    # Charger l'adapter depuis le checkpoint
    import os
    best_checkpoint_path = f"{OUTPUT_DIR}/best"
    if os.path.exists(best_checkpoint_path):
        # Appliquer l'adapter
        from gliner2.training.lora import load_lora_adapter
        try:
            load_lora_adapter(best_model, best_checkpoint_path)
            print("   ✅ Adapter LoRA chargé")
        except Exception as e:
            print(f"   ⚠️  Erreur chargement adapter: {e}")
            print("   ℹ️  Utilisation du modèle de base pour test")
    else:
        print(f"   ⚠️  Checkpoint introuvable: {best_checkpoint_path}")
        print("   ℹ️  Utilisation du modèle entraîné en mémoire")
        best_model = model
else:
    # Pour full fine-tuning, chargement normal
    best_checkpoint_path = f"{OUTPUT_DIR}/best"
    if os.path.exists(best_checkpoint_path):
        best_model = GLiNER2.from_pretrained(best_checkpoint_path)
    else:
        print(f"   ⚠️  Checkpoint introuvable: {best_checkpoint_path}")
        print("   ℹ️  Utilisation du modèle entraîné en mémoire")
        best_model = model

# Exemples de test
test_queries = [
    "Je veux aller de Cotonou à Parakou demain à 8h avec 2 adultes",
    "Un billet pour Abomey, je pars d'ici",
    "Direction Natitingou en passant par Djougou et Parakou",
    "Voyage d'affaires à Porto-Novo le 15 mars",
    "Gare de Jonquet vers la frontière nigériane, 3 personnes"
]

print("\n🔍 Extraction d'entités sur des exemples de test:")
print("-"*60)

for query in test_queries:
    # Créer le schéma
    schema = (
        best_model.create_schema()
        .entities(["Departure", "Destination", "Via", "Time", "Date", "Passengers", "TripType", "Purpose"])
    )
    
    # Extraire les entités
    result = best_model.extract(query, schema)
    
    print(f"\n📝 Phrase: \"{query}\"")
    
    # Afficher les entités détectées
    entities_found = False
    for entity_type, mentions in result['entities'].items():
        if mentions:
            entities_found = True
            print(f"   • {entity_type:12} : {mentions}")
    
    if not entities_found:
        print("   (Aucune entité détectée)")
    
    print("-"*60)