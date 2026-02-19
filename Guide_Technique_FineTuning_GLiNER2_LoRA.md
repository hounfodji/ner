# Guide Technique Complet : Fine-tuning de GLiNER2 avec LoRA

> **Projet** : Travel Order Resolver (T-AIA-901) — Groupe 4 : Extraction d'entités NER  
> **Auteurs** : Hospice Hounfodji, Juste Hodonou  
> **Modèle** : fastino/gliner2-multi-v1 + LoRA  
> **Date** : Février 2026

---

## Table des matières

1. [Qu'est-ce que GLiNER2 ?](#1-quest-ce-que-gliner2)
2. [Architecture interne de GLiNER2](#2-architecture-interne-de-gliner2)
3. [Qu'est-ce que LoRA ?](#3-quest-ce-que-lora)
4. [Comment LoRA modifie GLiNER2](#4-comment-lora-modifie-gliner2)
5. [Le dataset d'entraînement](#5-le-dataset-dentraînement)
6. [Le processus d'entraînement pas à pas](#6-le-processus-dentraînement-pas-à-pas)
7. [Chaque paramètre expliqué](#7-chaque-paramètre-expliqué)
8. [Ce qui se passe à chaque époque](#8-ce-qui-se-passe-à-chaque-époque)
9. [Early stopping et sélection du meilleur modèle](#9-early-stopping-et-sélection-du-meilleur-modèle)
10. [Inférence : comment le modèle fait une prédiction](#10-inférence--comment-le-modèle-fait-une-prédiction)
11. [Résumé visuel du pipeline complet](#11-résumé-visuel-du-pipeline-complet)

---

## 1. Qu'est-ce que GLiNER2 ?

### Le problème du NER classique

Les modèles NER classiques (spaCy, CamemBERT-NER) sont entraînés pour reconnaître des types d'entités **fixes** : `PER` (personne), `LOC` (lieu), `ORG` (organisation). Si tu veux un nouveau type comme `Departure` ou `TripType`, il faut **réentraîner tout le modèle** avec une nouvelle couche de classification.

### L'innovation de GLiNER

GLiNER (Generalist and Lightweight model for NER) résout ce problème en reformulant le NER comme un **problème de matching texte ↔ type**. Au lieu d'avoir une couche fixe qui dit "ce token est LOC ou PER", GLiNER calcule un **score de similarité** entre chaque segment du texte et chaque description de type d'entité.

Concrètement :
- Tu donnes au modèle une phrase : `"Je veux aller de Cotonou à Parakou"`
- Tu donnes une liste de types : `["Departure", "Destination", "Via", "Time"]`
- Le modèle calcule : **quel segment de texte correspond le mieux à quel type ?**

C'est pourquoi GLiNER est "zero-shot" : tu peux lui demander n'importe quel type d'entité **sans réentraînement**, juste en changeant la liste de types.

### GLiNER2 vs GLiNER1

GLiNER2 est la version améliorée avec :
- Un encodeur plus puissant : **mDeBERTa-v3-base** (multilingue, 100+ langues)
- Un mécanisme de comptage : **count_lstm** (gère mieux les entités multiples du même type)
- Un meilleur token pooling : **first** (utilise le premier sous-token pour représenter un mot)

---

## 2. Architecture interne de GLiNER2

Voici ce qui se passe à l'intérieur du modèle quand il traite une phrase :

```
┌─────────────────────────────────────────────────────────────┐
│                        ENTRÉE                                │
│  Phrase : "Je veux aller de Cotonou à Parakou"              │
│  Types  : ["Departure", "Destination", "Via", "Time"]       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               TOKENIZER (mDeBERTa)                          │
│                                                              │
│  Phrase → [CLS] Je veux aller de Cotonou à Parakou [SEP]   │
│  Types  → [CLS] Departure [SEP] Destination [SEP] ...      │
│                                                              │
│  Les deux sont tokenisés et concaténés en une seule         │
│  séquence d'entrée.                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           ENCODEUR TRANSFORMER (mDeBERTa-v3-base)           │
│                                                              │
│  12 couches de Transformer avec self-attention              │
│  Chaque token reçoit un vecteur contextuel de dim 768       │
│                                                              │
│  "Cotonou" voit "de" à sa gauche → comprend que c'est un   │
│  point de départ (grâce à l'attention sur "de")             │
│                                                              │
│  "Parakou" voit "à" à sa gauche → comprend que c'est une   │
│  destination (grâce à l'attention sur "à")                  │
│                                                              │
│  Paramètres : 279 M (86% du total)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           SPAN REPRESENTATION (span_rep)                     │
│                                                              │
│  Génère toutes les paires (début, fin) possibles :          │
│                                                              │
│  Span 1 : "Je"           (pos 1-1)                          │
│  Span 2 : "Je veux"      (pos 1-2)                          │
│  Span 3 : "Cotonou"      (pos 5-5)  ← candidat Departure   │
│  Span 4 : "Cotonou à"    (pos 5-6)                          │
│  Span 5 : "Parakou"      (pos 7-7)  ← candidat Destination │
│  ...                                                         │
│                                                              │
│  Pour chaque span, combine les vecteurs du premier et       │
│  dernier token → vecteur de span (dim 768)                  │
│                                                              │
│  Paramètres : ~2 M                                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           CLASSIFIER (scoring)                               │
│                                                              │
│  Calcule le score de matching entre chaque span et          │
│  chaque type d'entité :                                     │
│                                                              │
│  score("Cotonou", "Departure")   = 0.92  ✅ HIGH           │
│  score("Cotonou", "Destination") = 0.12  ❌ LOW            │
│  score("Parakou", "Departure")   = 0.08  ❌ LOW            │
│  score("Parakou", "Destination") = 0.87  ✅ HIGH           │
│  score("veux",    "Departure")   = 0.01  ❌ LOW            │
│                                                              │
│  Formule : score = sigmoid(span_vector · type_vector)       │
│                                                              │
│  Paramètres : ~29 M                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           COUNT LSTM                                         │
│                                                              │
│  Prédit combien d'entités de chaque type exister dans       │
│  la phrase. Aide le modèle à ne pas sur-prédire.            │
│                                                              │
│  count("Departure")   = 1                                   │
│  count("Destination") = 1                                   │
│  count("Via")         = 0                                   │
│  count("Time")        = 0                                   │
│                                                              │
│  Paramètres : ~200 K                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           SORTIE                                             │
│                                                              │
│  {                                                           │
│    "Departure":   ["Cotonou"],                              │
│    "Destination": ["Parakou"]                               │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

### Récapitulatif des paramètres

| Composant | Paramètres | % du total | Rôle |
|-----------|-----------|------------|------|
| Encodeur mDeBERTa | 279 M | 90% | Comprendre le contexte de chaque mot |
| Classifier | 29 M | 9.3% | Matcher spans ↔ types |
| Span representation | 2 M | 0.6% | Représenter les segments de texte |
| Count LSTM | 200 K | 0.1% | Compter les entités par type |
| **Total** | **310 M** | **100%** | |

---

## 3. Qu'est-ce que LoRA ?

### Le problème du fine-tuning classique

Quand on fine-tune un modèle classiquement (full fine-tuning), on modifie **tous les 310 millions de paramètres**. Cela pose 3 problèmes :

1. **Mémoire GPU** : il faut stocker les gradients pour 310M paramètres → ~2.5 GB juste pour les gradients
2. **Temps** : calculer les gradients pour 310M paramètres est lent
3. **Stockage** : le modèle fine-tuné fait 1.2 GB (copie complète)

### L'idée de LoRA

LoRA (Low-Rank Adaptation) est une technique publiée par Microsoft Research en 2021. L'idée fondamentale est :

> **Quand on fine-tune un modèle, les modifications nécessaires aux poids sont de faible rang.**

En termes simples : on n'a pas besoin de modifier **tous** les poids. On peut approximer les modifications avec des matrices beaucoup plus petites.

### Comment ça marche mathématiquement

Prenons une matrice de poids W d'une couche du Transformer :

```
W original : matrice 768 × 768 = 589 824 paramètres
```

En full fine-tuning, on calcule :

```
W_nouveau = W_original + ΔW

où ΔW est la matrice de mise à jour (768 × 768 = 589 824 paramètres)
```

Avec LoRA, on **décompose** ΔW en deux petites matrices :

```
ΔW = A × B

où :
  A : matrice 768 × r    (768 × 16 = 12 288 paramètres)
  B : matrice r × 768    (16 × 768 = 12 288 paramètres)

Total LoRA : 12 288 + 12 288 = 24 576 paramètres (au lieu de 589 824)
```

**r** est le **rang** de LoRA (`lora_r=16` dans notre config). C'est la dimension de la décomposition.

### Visualisation

```
                   Full fine-tuning                    LoRA
                   ────────────────                    ────

Poids originaux :  W (768 × 768)                      W (768 × 768) ← GELÉS
                   ↓                                   ↓
Mise à jour :      ΔW (768 × 768)                     A (768 × 16) × B (16 × 768)
                   ↓                                   ↓
Résultat :         W + ΔW                              W + A × B

Paramètres à      589 824                              24 576
entraîner :       (100%)                               (4.2%)
```

### Le facteur d'échelle alpha

Il y a un paramètre supplémentaire : `lora_alpha=32`. Il contrôle **l'amplitude** de la modification LoRA :

```
W_nouveau = W_original + (alpha / r) × A × B
          = W_original + (32 / 16) × A × B
          = W_original + 2.0 × A × B
```

Le ratio `alpha/r = 32/16 = 2.0` est le facteur de mise à l'échelle. Un ratio plus grand signifie que les modifications LoRA ont plus d'impact sur le modèle. Dans notre cas, `2.0` est un ratio standard qui donne un bon équilibre entre adaptation et stabilité.

### Le dropout LoRA

`lora_dropout=0.1` signifie que pendant l'entraînement, 10% des connexions LoRA sont aléatoirement désactivées à chaque pas. Cela force le modèle à ne pas trop dépendre de connexions spécifiques, ce qui améliore la **généralisation** (capacité à bien fonctionner sur des phrases jamais vues).

---

## 4. Comment LoRA modifie GLiNER2

### Quelles couches sont modifiées ?

Dans notre configuration, LoRA est appliqué à 3 composants :

```python
lora_target_modules = ["encoder", "span_rep", "classifier"]
```

| Composant | Poids originaux | Paramètres LoRA | Ce que LoRA apprend |
|-----------|----------------|-----------------|---------------------|
| `encoder` | 279 M (gelés) | ~2.5 M | Mieux comprendre le contexte transport béninois |
| `span_rep` | 2 M (gelés) | ~300 K | Mieux représenter les segments géographiques |
| `classifier` | 29 M (gelés) | ~300 K | Mieux scorer les matchings span↔type |
| **Total** | **310 M (gelés)** | **3.1 M (entraînés)** | **1% du modèle** |

### Ce qui se passe concrètement

```
AVANT LoRA (zero-shot) :
  "frontière nigériane" → Destination score : 0.35 → en dessous du seuil → PAS DÉTECTÉ

APRÈS LoRA (fine-tuné) :
  "frontière nigériane" → Destination score : 0.87 → au dessus du seuil → DÉTECTÉ ✅
```

LoRA a modifié subtilement les matrices de l'encodeur pour que les représentations vectorielles de "frontière nigériane" soient plus proches de celles de "Destination". Le modèle a **appris** que dans le contexte du transport béninois, "frontière nigériane" est une destination valide.

### L'adapter sauvegardé

Quand on sauvegarde le modèle, on ne sauvegarde **que les matrices LoRA** (A et B), pas tout le modèle :

```
benin_transport_ner_model/best/
├── adapter_weights.safetensors   ← les matrices A et B (11.8 MB)
└── adapter_config.json            ← la configuration LoRA (r, alpha, etc.)
```

Pour l'inférence, on charge le modèle de base (1.2 GB depuis HuggingFace), puis on **applique** l'adapter (11.8 MB) dessus. C'est comme mettre un "patch" sur le modèle original.

---

## 5. Le dataset d'entraînement

### Format des données

Chaque exemple du dataset est une ligne JSON (format JSONL) :

```json
{
  "text": "Je veux aller de Cotonou à Parakou demain à 8h",
  "entities": [
    {"start": 20, "end": 27, "label": "Departure", "text": "Cotonou"},
    {"start": 30, "end": 37, "label": "Destination", "text": "Parakou"},
    {"start": 38, "end": 44, "label": "Date", "text": "demain"},
    {"start": 47, "end": 49, "label": "Time", "text": "8h"}
  ]
}
```

Chaque entité est définie par :
- `start` / `end` : position en caractères dans le texte
- `label` : le type d'entité (parmi nos 8 types)
- `text` : le texte exact de l'entité (doit correspondre à `text[start:end]`)

### Statistiques

| Statistique | Valeur |
|------------|--------|
| Nombre d'exemples | 10 061 |
| Total de mentions d'entités | 36 462 |
| Moyenne d'entités par exemple | 3.6 |
| Split entraînement | 85% (8 552 exemples) |
| Split validation | 15% (1 509 exemples) |

### Distribution des types d'entités

| Type | Occurrences | % du total |
|------|------------|------------|
| Destination | 9 686 | 26.6% |
| Departure | 8 823 | 24.2% |
| Date | 4 105 | 11.3% |
| Time | 3 790 | 10.4% |
| Via | 2 712 | 7.4% |
| Passengers | 2 961 | 8.1% |
| TripType | 2 714 | 7.4% |
| Purpose | 1 671 | 4.6% |

### Problèmes de qualité rencontrés

Le dataset initial contenait **612 erreurs d'annotation** (1.7%) qui dégradaient fortement les performances. Trois catégories de problèmes :

**1. Erreurs de spans (décalage de position)**

```json
// ❌ AVANT : le texte dit "touristique" mais l'annotation dit "tourisme"
{"start": 45, "end": 52, "label": "Purpose", "text": "tourisme"}
// Le texte réel à ces positions : "tourist" (pas "tourisme")

// ✅ APRÈS correction : réalignement automatique
{"start": 45, "end": 57, "label": "Purpose", "text": "touristique"}
```

**2. Erreurs sémantiques (mauvais type)**

```json
// ❌ "aller" annoté comme TripType alors que c'est un verbe
// Phrase : "Je veux aller de Cotonou à Parakou"
{"start": 9, "end": 14, "label": "TripType", "text": "aller"}
// Ici "aller" = verbe (je veux ALLER), pas un type de trajet (billet ALLER)

// ✅ Solution : suppression de ces annotations quand "aller" est un verbe
// 126 cas corrigés
```

**3. Double encodage UTF-8 (mojibake)**

```
❌ "CotÃ´nou"  → double-encodé
✅ "Cotonou"   → corrigé
```

Un script de nettoyage automatique (`clean_dataset_v2.py`) a corrigé 592 annotations et supprimé 274 annotations irréparables.

---

## 6. Le processus d'entraînement pas à pas

### Vue d'ensemble

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Dataset     │     │   Modèle     │     │   Sortie     │
│  10 061      │────▶│   GLiNER2    │────▶│   Adapter    │
│  exemples    │     │   + LoRA     │     │   LoRA       │
│  (JSONL)     │     │              │     │   (11.8 MB)  │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │
       ▼                    ▼
  Split 85/15          Entraînement
  train / val          5 époques
                       ~3.3 minutes
```

### Ce qui se passe à chaque pas (step)

Un **pas** (step) correspond au traitement d'un batch de 16 exemples :

```
Pas 1 du training :

1. FORWARD PASS (passage avant)
   ─────────────────────────────
   - Prend 16 phrases du dataset
   - Les passe à travers le modèle
   - Pour chaque phrase, le modèle prédit les entités
   - Compare les prédictions avec les vraies annotations
   - Calcule la PERTE (loss) : écart entre prédiction et réalité

   Exemple :
   Phrase : "Je veux aller de Cotonou à Parakou"
   Prédit : Departure="Je" (score 0.6), Destination="Parakou" (score 0.7)
   Vrai   : Departure="Cotonou", Destination="Parakou"
   Loss   : élevée (car "Je" ≠ "Cotonou")

2. BACKWARD PASS (rétropropagation)
   ─────────────────────────────────
   - Calcule les GRADIENTS : dans quelle direction modifier les poids
     pour réduire la perte ?
   - Les gradients ne sont calculés QUE pour les matrices LoRA (A et B)
   - Les 310M paramètres du modèle de base sont GELÉS (pas de gradient)

3. OPTIMIZER STEP (mise à jour)
   ────────────────────────────
   - L'optimiseur AdamW utilise les gradients pour modifier A et B
   - Taux d'apprentissage : 3e-4 (chaque poids bouge de 0.0003 × gradient)
   - Weight decay : 0.01 (légère pénalité sur les gros poids → régularisation)

4. RÉSULTAT
   ────────
   Le modèle est légèrement meilleur qu'avant ce pas.
   On passe au batch suivant.
```

### Timeline complète de l'entraînement

```
Époque 1 (534 steps) ─────────────────────────────────────────
│ Step 1-50    : Loss ~8.0 → le modèle apprend les bases
│ Step 50-100  : Loss ~6.5 → commence à reconnaître les villes
│ Step 100-534 : Loss ~5.0 → reconnaît Departure vs Destination
│ 
│ → Évaluation sur validation : val_loss = 4.20
│ → Sauvegarde checkpoint-epoch-1
└──────────────────────────────────────────────────────────────

Époque 2 (534 steps) ─────────────────────────────────────────
│ Le modèle voit les MÊMES données mais dans un ORDRE DIFFÉRENT
│ Il affine sa compréhension des patterns linguistiques
│ 
│ → Évaluation : val_loss = 3.45 (↓ meilleur !)
│ → Sauvegarde checkpoint-epoch-2 + best/
└──────────────────────────────────────────────────────────────

Époque 3 (534 steps) ─────────────────────────────────────────
│ → Évaluation : val_loss = 3.10 (↓ encore meilleur !)
│ → Sauvegarde checkpoint-epoch-3 + mise à jour best/
└──────────────────────────────────────────────────────────────

Époque 4 (534 steps) ─────────────────────────────────────────
│ → Évaluation : val_loss = 2.97 (↓ MEILLEUR SCORE !)
│ → Sauvegarde checkpoint-epoch-4 + mise à jour best/
└──────────────────────────────────────────────────────────────

Époque 5 (534 steps) ─────────────────────────────────────────
│ → Évaluation : val_loss = 3.15 (↑ régression !)
│ → Le modèle commence à SURAPPRENDRE (overfitting)
│ → Early stopping déclenché (patience 5 non atteinte, mais
│   on observe la tendance)
└──────────────────────────────────────────────────────────────

RÉSULTAT FINAL :
  Meilleur checkpoint = époque 4, val_loss = 2.97
  Sauvegardé dans benin_transport_ner_model/best/
```

---

## 7. Chaque paramètre expliqué

### Paramètres d'entraînement

| Paramètre | Valeur | Explication détaillée |
|-----------|--------|----------------------|
| `num_epochs` | 15 | Nombre maximum de passages sur tout le dataset. En pratique, l'early stopping arrête avant (époque 5). |
| `batch_size` | 16 | Nombre de phrases traitées simultanément. 16 = bon compromis mémoire/vitesse. Plus grand = plus stable mais plus de mémoire GPU. |
| `task_lr` | 3e-4 (0.0003) | Learning rate : vitesse à laquelle les poids LoRA sont modifiés. Trop grand (0.01) = le modèle "saute" et n'apprend rien. Trop petit (0.00001) = le modèle apprend trop lentement. |
| `weight_decay` | 0.01 | Régularisation L2 : pénalise les poids trop grands pour éviter l'overfitting. À chaque pas, les poids sont multipliés par `(1 - 0.01) = 0.99`. |
| `max_grad_norm` | 1.0 | Gradient clipping : si le gradient est trop grand (>1.0), il est renormalisé. Empêche les "explosions de gradient" qui déstabilisent l'entraînement. |
| `warmup_ratio` | 0.1 | Les 10% premiers pas utilisent un learning rate progressif (de 0 à 3e-4). Cela stabilise le début de l'entraînement quand les gradients sont bruités. |
| `scheduler_type` | linear | Après le warmup, le learning rate décroît linéairement jusqu'à 0. L'idée : au début on fait de gros ajustements, à la fin de petits ajustements fins. |
| `fp16` | True | Mixed precision : utilise des nombres 16-bit au lieu de 32-bit pour les calculs. Divise la mémoire par ~2 et accélère de ~30% avec une perte de précision négligeable. |

### Paramètres LoRA

| Paramètre | Valeur | Explication détaillée |
|-----------|--------|----------------------|
| `lora_r` | 16 | Rang de la décomposition. Plus grand = plus expressif mais plus de paramètres. 16 est le standard pour NER. Des valeurs 4-64 sont typiques. |
| `lora_alpha` | 32 | Facteur d'échelle. Le ratio alpha/r = 32/16 = 2.0 contrôle l'amplitude des modifications. Standard = alpha = 2 × r. |
| `lora_dropout` | 0.1 | 10% des connexions LoRA sont aléatoirement coupées pendant l'entraînement. Régularisation qui empêche l'overfitting. |
| `lora_target_modules` | encoder, span_rep, classifier | Quelles parties du modèle sont adaptées. On adapte les 3 composants principaux. |
| `save_adapter_only` | True | Ne sauvegarde que les matrices LoRA (11.8 MB), pas le modèle complet (1.2 GB). |

### Paramètres d'évaluation

| Paramètre | Valeur | Explication détaillée |
|-----------|--------|----------------------|
| `eval_strategy` | epoch | Évaluer sur le set de validation à la fin de chaque époque (pas à chaque step). |
| `save_best` | True | Garder le meilleur checkpoint en mémoire (celui avec la plus basse val_loss). |
| `metric_for_best` | eval_loss | La métrique utilisée pour choisir le "meilleur" modèle. On utilise la loss de validation. |
| `greater_is_better` | False | Plus la loss est basse, mieux c'est (contrairement à l'accuracy où plus c'est haut, mieux c'est). |
| `save_total_limit` | 3 | Garde au maximum 3 checkpoints sur disque (les 3 meilleurs + best). Économise de l'espace disque. |
| `early_stopping_patience` | 5 | Si la val_loss ne s'améliore pas pendant 5 époques consécutives, on arrête l'entraînement. |

---

## 8. Ce qui se passe à chaque époque

### Époque = 1 passage complet sur le dataset

```
Dataset : 8 552 exemples d'entraînement
Batch size : 16
Steps par époque : 8 552 / 16 = 534 steps

À chaque step, le modèle :
1. Lit 16 phrases
2. Prédit les entités
3. Compare avec la vérité
4. Calcule les gradients (seulement sur les params LoRA)
5. Met à jour les poids LoRA
```

### Pourquoi plusieurs époques ?

Le modèle voit les données **plusieurs fois** parce que :
- À la première passe, il n'apprend que les patterns les plus évidents ("de X" → Departure)
- À la deuxième passe, il affine sur les cas plus subtils ("en passant par X" → Via)
- À chaque passe, les données sont dans un **ordre aléatoire différent** (shuffle), ce qui force le modèle à généraliser plutôt que mémoriser

### L'overfitting : pourquoi il faut s'arrêter

```
Époque  │ Train Loss │ Val Loss │ Interprétation
────────┼────────────┼──────────┼──────────────────────────
   1    │    12.5    │   4.20   │ Apprend les bases
   2    │     9.8    │   3.45   │ S'améliore
   3    │     8.2    │   3.10   │ S'améliore encore
   4    │     7.0    │   2.97   │ ★ MEILLEUR (best checkpoint)
   5    │     5.5    │   3.15   │ ↑ Val monte = OVERFITTING
   6    │     4.0    │   3.40   │ ↑ Continue de monter
   7    │     2.5    │   3.80   │ ↑ Overfitting sévère
────────┴────────────┴──────────┴──────────────────────────

Observation :
- Train loss baisse TOUJOURS (le modèle mémorise les exemples)
- Val loss baisse d'abord puis REMONTE (le modèle ne généralise plus)
```

Le **sweet spot** est l'époque 4 : le moment où le modèle a le plus appris **sans avoir commencé à mémoriser**.

---

## 9. Early stopping et sélection du meilleur modèle

### Comment ça marche

```python
early_stopping_patience = 5
```

Le mécanisme :
1. À chaque fin d'époque, on évalue la val_loss
2. Si elle est **meilleure** que la meilleure précédente → on sauvegarde dans `best/` et on remet le compteur à 0
3. Si elle est **pire** → on incrémente le compteur de patience
4. Si le compteur atteint 5 → on arrête l'entraînement

```
Époque 1 : val_loss = 4.20 → MEILLEUR (compteur = 0, save best/)
Époque 2 : val_loss = 3.45 → MEILLEUR (compteur = 0, save best/)
Époque 3 : val_loss = 3.10 → MEILLEUR (compteur = 0, save best/)
Époque 4 : val_loss = 2.97 → MEILLEUR (compteur = 0, save best/)
Époque 5 : val_loss = 3.15 → PIRE     (compteur = 1)
Époque 6 : val_loss = 3.40 → PIRE     (compteur = 2)
...
Époque 9 : val_loss = 4.10 → PIRE     (compteur = 5) → STOP !

Le modèle dans best/ est celui de l'époque 4 (val_loss = 2.97)
```

### Pourquoi c'est important

Sans early stopping, le modèle continuerait à s'entraîner jusqu'à l'époque 15, **dégradant** ses performances sur les données réelles. L'early stopping économise du temps de calcul et garantit qu'on utilise le modèle au moment où il est le plus performant.

---

## 10. Inférence : comment le modèle fait une prédiction

### En production (API FastAPI)

Quand quelqu'un envoie une requête à l'API :

```
POST /api/v1/ner/extract
{"text": "Je veux aller de Cotonou à Parakou demain à 8h"}
```

Voici ce qui se passe :

```
Étape 1 : CHARGEMENT (au démarrage, une seule fois)
───────────────────────────────────────────────────
model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")  # 1.2 GB
load_lora_adapter(model, "benin_transport_ner_model/best")    # 11.8 MB
# Le modèle est maintenant : base + patch LoRA = fine-tuné

Étape 2 : SCHÉMA
────────────────
schema = model.create_schema().entities([
    "Departure", "Destination", "Via", "Time",
    "Date", "Passengers", "TripType", "Purpose"
])
# Dit au modèle quels types chercher

Étape 3 : EXTRACTION (~45ms)
───────────────────────────
result = model.extract(text, schema)
# Le modèle tokenise, encode, génère les spans,
# calcule les scores, et retourne les résultats

Étape 4 : POST-PROCESSING
─────────────────────────
# Filtre les faux positifs :
# - TripType "aller" si c'est un verbe
# - Departure "ici" (pas un vrai lieu)
# - Date contenant un nom de ville
# - Déduplication si le même span apparaît dans 2 types

Étape 5 : RÉPONSE
─────────────────
{
  "entities": {
    "Departure": ["Cotonou"],
    "Destination": ["Parakou"],
    "Date": ["demain"],
    "Time": ["8h"]
  },
  "processing_time_ms": 45
}
```

---

## 11. Résumé visuel du pipeline complet

```
╔══════════════════════════════════════════════════════════════╗
║                PIPELINE COMPLET                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ENTRAÎNEMENT (fait une seule fois, ~3 min)                  ║
║  ─────────────────────────────────────────                   ║
║                                                              ║
║  Dataset JSONL ──▶ GLiNER2 base ──▶ LoRA fine-tuning         ║
║  (10 061 ex.)      (310M params)     (3.1M params)           ║
║                                          │                   ║
║                                          ▼                   ║
║                                    adapter_weights           ║
║                                    (11.8 MB)                 ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  INFÉRENCE (à chaque requête, ~45 ms)                        ║
║  ────────────────────────────────────                        ║
║                                                              ║
║  Phrase ──▶ GLiNER2 base + LoRA ──▶ Post-processing ──▶ JSON ║
║             (modèle patché)          (filtrage FP)           ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  MÉTRIQUES                                                   ║
║  ─────────                                                   ║
║  • Val loss : 4.75 → 2.97 (−37%)                            ║
║  • Score test : 30% → 80% (+167%)                            ║
║  • Params entraînés : 3.1M / 310M (1%)                      ║
║  • Temps entraînement : 3.3 minutes                          ║
║  • Temps inférence : ~45 ms/requête                          ║
║  • Taille adapter : 11.8 MB                                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Glossaire rapide

| Terme | Définition simple |
|-------|-------------------|
| **Transformer** | Architecture de réseau de neurones basée sur le mécanisme d'attention. Permet à chaque mot de "voir" tous les autres mots de la phrase. |
| **Self-attention** | Mécanisme qui calcule l'importance de chaque mot par rapport à tous les autres. "Cotonou" fait plus attention à "de" qu'à "Je". |
| **mDeBERTa** | Modèle de type BERT multilingue développé par Microsoft, utilisé comme encodeur dans GLiNER2. Pré-entraîné sur 100+ langues. |
| **Tokenisation** | Découpage du texte en sous-unités. "Cotonou" reste un seul token, mais "Natitingou" peut devenir ["Nati", "##tingou"]. |
| **Span** | Segment continu de texte. "Gare de Jonquet" est un span de 3 tokens. |
| **Loss (perte)** | Mesure de l'erreur du modèle. Plus c'est bas, mieux le modèle prédit. |
| **Gradient** | Direction dans laquelle modifier les poids pour réduire la loss. Calculé par rétropropagation. |
| **Batch** | Groupe de phrases traitées ensemble. Batch de 16 = 16 phrases par step. |
| **Époque** | Un passage complet sur tout le dataset d'entraînement. |
| **Overfitting** | Le modèle mémorise les données d'entraînement au lieu de généraliser. Se détecte quand la val_loss remonte. |
| **Early stopping** | Arrête l'entraînement automatiquement quand le modèle commence à overfitter. |
| **Checkpoint** | Sauvegarde de l'état du modèle à un instant donné. Le "best" est le meilleur. |
| **Adapter** | Le fichier contenant uniquement les poids LoRA (11.8 MB). Se "patch" sur le modèle de base. |
| **Zero-shot** | Utiliser le modèle sans aucun entraînement spécifique au domaine. |
| **Fine-tuning** | Entraînement supplémentaire sur des données spécifiques à notre domaine (transport béninois). |
| **FP16** | Calculs en nombres flottants 16-bit (au lieu de 32-bit). Plus rapide, moins de mémoire. |
| **AdamW** | Algorithme d'optimisation qui adapte le learning rate pour chaque paramètre individuellement. |
