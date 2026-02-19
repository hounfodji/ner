#!/usr/bin/env python3
"""
Fine-tuning GLiNER2 pour l'extraction d'entitÃ©s de transport au BÃ©nin.

EntitÃ©s cibles:
- Departure: lieu de dÃ©part
- Destination: lieu d'arrivÃ©e
- Via: lieux de passage
- Time: heure de dÃ©part/arrivÃ©e
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

# Chemins des donnÃ©es
TRAIN_FILE = "train_gliner2_ner_10000_v2.jsonl"  # Dataset nettoyé v2
OUTPUT_DIR = "./benin_transport_ner_model"
EXPERIMENT_NAME = "benin_transport_ner_v2"

# ParamÃ¨tres d'entraÃ®nement
NUM_EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE_ENCODER = 1e-5  # LR pour l'encodeur
LEARNING_RATE_TASK = 3e-4     # LR réduit pour meilleure convergence (était 5e-4)

# Utiliser LoRA ? (recommandÃ© pour Ã©conomiser mÃ©moire/temps)
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
# Ã‰TAPE 1: CHARGEMENT ET PRÃ‰PARATION DES DONNÃ‰ES
# ============================================================

print("="*60)
print("ðŸš€ FINE-TUNING GLINER2 - TRANSPORT BÃ‰NINOIS")
print("="*60)

print("\nðŸ“ Chargement du dataset...")
dataset = TrainingDataset.load(TRAIN_FILE)

# Statistiques du dataset
print("\nðŸ“Š Statistiques du dataset:")
dataset.print_stats()

# Validation du dataset (dÃ©sactivÃ©e - donnÃ©es dÃ©jÃ  vÃ©rifiÃ©es)
print("\nâ­ï¸  Validation dÃ©sactivÃ©e (passage direct Ã  l'entraÃ®nement)")
# Note: certaines entitÃ©s peuvent ne pas Ãªtre exactement dans le texte
# mais GLiNER2 gÃ¨re cela pendant l'entraÃ®nement

# ============================================================
# Ã‰TAPE 2: SPLIT TRAIN/VALIDATION
# ============================================================

print(f"\nðŸ“‚ Split train/validation ({100-int(VAL_RATIO*100)}/{int(VAL_RATIO*100)})...")
train_data, val_data, _ = dataset.split(
    train_ratio=1-VAL_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=0.0,
    shuffle=True,
    seed=RANDOM_SEED
)

print(f"   â€¢ Exemples d'entraÃ®nement: {len(train_data)}")
print(f"   â€¢ Exemples de validation: {len(val_data)}")

# Note: On ne sauvegarde pas les splits pour Ã©viter la validation automatique
print("   â„¹ï¸  Splits crÃ©Ã©s en mÃ©moire (non sauvegardÃ©s sur disque)")

# ============================================================
# Ã‰TAPE 3: CHARGEMENT DU MODÃˆLE
# ============================================================

print("\nðŸ§  Chargement du modÃ¨le de base...")
model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
print("   âœ… ModÃ¨le chargÃ©: fastino/gliner2-multi-v1 (multilingue)")

# ============================================================
# Ã‰TAPE 4: CONFIGURATION DE L'ENTRAÃŽNEMENT
# ============================================================

print("\nâš™ï¸  Configuration de l'entraÃ®nement...")

if USE_LORA:
    print(f"   â€¢ Mode: LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})")
    print("   â€¢ Avantages: moins de mÃ©moire, entraÃ®nement plus rapide")
else:
    print("   â€¢ Mode: Full fine-tuning")
    print("   â€¢ Note: plus de mÃ©moire requise")

config = TrainingConfig(
    # Sortie
    output_dir=OUTPUT_DIR,
    experiment_name=EXPERIMENT_NAME,
    
    # EntraÃ®nement
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
    
    # Mixed precision (recommandÃ©)
    fp16=True,
    
    # LoRA (si activÃ©)
    use_lora=USE_LORA,
    lora_r=LORA_RANK if USE_LORA else None,
    lora_alpha=LORA_ALPHA if USE_LORA else None,
    lora_dropout=0.1 if USE_LORA else None,
    lora_target_modules=["encoder", "span_rep", "classifier"] if USE_LORA else None,
    save_adapter_only=True if USE_LORA else False,
    
    # Ã‰valuation et checkpoints
    eval_strategy="epoch",  # Ã‰value et sauvegarde Ã  chaque Ã©poque
    save_best=True,
    metric_for_best="eval_loss",
    greater_is_better=False,
    save_total_limit=3,  # Garde seulement les 3 meilleurs checkpoints
    
    # Early stopping
    early_stopping=True,
    early_stopping_patience=5,  # Augmenté de 3 à 5 pour laisser converger
    
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

print("\nðŸ“‹ Configuration:")
print(f"   â€¢ Ã‰poques: {NUM_EPOCHS}")
print(f"   â€¢ Batch size: {BATCH_SIZE}")
print(f"   â€¢ Learning rate (task): {LEARNING_RATE_TASK}")
if not USE_LORA:
    print(f"   â€¢ Learning rate (encoder): {LEARNING_RATE_ENCODER}")
print(f"   â€¢ Warmup ratio: 0.1")
print(f"   â€¢ Mixed precision: FP16")
print(f"   â€¢ Early stopping: activÃ© (patience=3)")
print(f"   â€¢ Weights & Biases: {'activÃ©' if USE_WANDB else 'dÃ©sactivÃ©'}")

# ============================================================
# Ã‰TAPE 5: ENTRAÃŽNEMENT
# ============================================================

print("\n" + "="*60)
print("ðŸ‹ï¸  DÃ‰BUT DE L'ENTRAÃŽNEMENT")
print("="*60)

trainer = GLiNER2Trainer(model, config)

results = trainer.train(
    train_data=train_data,
    eval_data=val_data
)

# ============================================================
# Ã‰TAPE 6: RÃ‰SULTATS
# ============================================================

print("\n" + "="*60)
print("âœ… ENTRAÃŽNEMENT TERMINÃ‰")
print("="*60)

print(f"\nðŸ“Š RÃ©sultats finaux:")
print(f"   â€¢ Best validation loss: {results.get('best_metric', 'N/A'):.4f}")
print(f"   â€¢ Total steps: {results.get('total_steps', 'N/A')}")
print(f"   â€¢ Temps total: {results.get('total_time_seconds', 0)/60:.1f} minutes")

print(f"\nðŸ’¾ ModÃ¨le sauvegardÃ© dans: {OUTPUT_DIR}")
print(f"   â€¢ Meilleur checkpoint: {OUTPUT_DIR}/best")
print(f"   â€¢ Dernier checkpoint: {OUTPUT_DIR}/checkpoint-{results.get('total_steps', 'final')}")

# ============================================================
# Ã‰TAPE 7: TEST D'INFÃ‰RENCE
# ============================================================

print("\n" + "="*60)
print("ðŸ§ª TEST D'INFÃ‰RENCE")
print("="*60)

print("\nðŸ“¦ Chargement du meilleur modÃ¨le...")

# Pour LoRA, on doit recharger le modÃ¨le de base puis appliquer l'adapter
if USE_LORA:
    print("   â„¹ï¸  Mode LoRA: chargement du modÃ¨le de base + adapter")
    best_model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
    
    # Charger l'adapter depuis le checkpoint
    import os
    best_checkpoint_path = f"{OUTPUT_DIR}/best"
    if os.path.exists(best_checkpoint_path):
        # Appliquer l'adapter
        from gliner2.training.lora import load_lora_adapter
        try:
            load_lora_adapter(best_model, best_checkpoint_path)
            print("   âœ… Adapter LoRA chargÃ©")
        except Exception as e:
            print(f"   âš ï¸  Erreur chargement adapter: {e}")
            print("   â„¹ï¸  Utilisation du modÃ¨le de base pour test")
    else:
        print(f"   âš ï¸  Checkpoint introuvable: {best_checkpoint_path}")
        print("   â„¹ï¸  Utilisation du modÃ¨le entraÃ®nÃ© en mÃ©moire")
        best_model = model
else:
    # Pour full fine-tuning, chargement normal
    best_checkpoint_path = f"{OUTPUT_DIR}/best"
    if os.path.exists(best_checkpoint_path):
        best_model = GLiNER2.from_pretrained(best_checkpoint_path)
    else:
        print(f"   âš ï¸  Checkpoint introuvable: {best_checkpoint_path}")
        print("   â„¹ï¸  Utilisation du modÃ¨le entraÃ®nÃ© en mÃ©moire")
        best_model = model

# Exemples de test
test_queries = [
    "Je veux aller de Cotonou Ã  Parakou demain Ã  8h avec 2 adultes",
    "Un billet pour Abomey, je pars d'ici",
    "Direction Natitingou en passant par Djougou et Parakou",
    "Voyage d'affaires Ã  Porto-Novo le 15 mars",
    "Gare de Jonquet vers la frontiÃ¨re nigÃ©riane, 3 personnes"
]

print("\nðŸ” Extraction d'entitÃ©s sur des exemples de test:")
print("-"*60)

for query in test_queries:
    # CrÃ©er le schÃ©ma
    schema = (
        best_model.create_schema()
        .entities(["Departure", "Destination", "Via", "Time", "Date", "Passengers", "TripType", "Purpose"])
    )
    
    # Extraire les entitÃ©s
    result = best_model.extract(query, schema)
    
    print(f"\nðŸ“ Phrase: \"{query}\"")
    
    # Afficher les entitÃ©s dÃ©tectÃ©es
    entities_found = False
    for entity_type, mentions in result['entities'].items():
        if mentions:
            entities_found = True
            print(f"   â€¢ {entity_type:12} : {mentions}")
    
    if not entities_found:
        print("   (Aucune entitÃ© dÃ©tectÃ©e)")
    
    print("-"*60)