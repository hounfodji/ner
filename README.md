# Benin Transport NER — GLiNER2 + LoRA

Encoder fine-tuning of GLiNER2 with LoRA for named-entity recognition on French-language
public-transport queries in Benin, served through a FastAPI inference API.

## Overview

The system extracts structured trip information (origin, destination, stops, time, date,
passengers, trip type, purpose) from free-form French sentences such as *"Je veux aller de
Cotonou à Parakou demain à 8h avec 2 adultes"*. It is the NER component of a "Travel Order
Resolver": turning a natural-language travel request into machine-readable fields. The target
is a **low-resource setting** — French as used in Benin, with local toponyms and informal
phrasing — for which no off-the-shelf labelled corpus exists, so the training data was
generated and validated programmatically.

Built by Groupe 4 (Hospice Hounfodji, Juste Hodonou) for the *Travel Order Resolver
(T-AIA-901)* project.

## Technical approach

### Entity schema

Eight domain entity types, modelled as a flat span-extraction schema:

| Entity | Meaning |
|---|---|
| `Departure` | Origin location |
| `Destination` | Final destination |
| `Via` | Intermediate stop(s) |
| `Time` | Departure/arrival time |
| `Date` | Travel date |
| `Passengers` | Passenger count/type (e.g. "2 adultes", "1 enfant") |
| `TripType` | Trip type (aller, retour, aller-retour) |
| `Purpose` | Trip purpose (business, tourism, …) |

### Model and architecture

The base model is **GLiNER2** (`fastino/gliner2-multi-v1`), a zero-shot NER model that
reformulates entity extraction as text-span ↔ entity-type matching over a multilingual
**mDeBERTa-v3-base** encoder. Rather than a fixed label head, the model scores the similarity
between text spans and natural-language entity-type names, so the entity set can be changed at
inference time without retraining. This is what makes it a good fit for a custom domain schema.

### Fine-tuning method (LoRA)

The encoder is adapted to the transport domain with **LoRA** (low-rank adapters), training only
a small set of adapter weights while keeping the base model frozen. Configuration
(`train_benin_transport_ner_v2.py`, `benin_transport_ner_model/training_config.json`):

- LoRA rank `r = 16`, `alpha = 32`, dropout `0.1`
- Target modules: `encoder`, `span_rep`, `classifier`
- Adapter-only checkpointing (`save_adapter_only = true`) — the trained adapter is ~12 MB
- 15 epochs, batch size 16, task learning rate `3e-4`, weight decay `0.01`, gradient clipping `1.0`
- Linear scheduler, warmup ratio `0.1`, FP16 mixed precision
- Per-epoch evaluation, best checkpoint selected on `eval_loss`, `save_total_limit = 3`
- Early stopping with patience 5; 85/15 train/validation split; seed 42

Trained artifacts are committed under `benin_transport_ner_model/` (`best/`, `final/`,
per-epoch checkpoints). Only the LoRA adapter weights are stored — the base model is pulled
from the Hugging Face Hub at load time.

### Dataset

Training data is `train_gliner2_ner_10000_v2.jsonl`: **10,047 examples** in GLiNER2 JSONL
format (`{"input": ..., "output": {"entities": {...}}}`), averaging 3.6 entities per example.
Because no labelled in-domain corpus exists, the dataset was **synthesised** rather than
hand-annotated (`old/generate_dataset.py`):

1. A local LLM (via Ollama) generates candidate query/annotation pairs from few-shot examples,
   grounded on a Benin geographic reference document (toponyms, landmarks, routes) so that
   locations are realistic.
2. Every generated line passes a **strict programmatic validator** before being kept:
   JSON/schema shape, singular/plural agreement on passenger counts, time-vs-duration
   disambiguation, and semantic-consistency rules (e.g. rejecting "I'll stay in X" non-trips
   and degenerate departure==destination cases). Invalid lines are dropped with tracked
   rejection reasons.

This validator-in-the-loop pipeline is the main mechanism for keeping a fully synthetic
dataset clean. Approximate per-type coverage (examples containing the type):

```
Destination 9618   Departure 6793   Passengers 4075   Time 3883
Date 3427          TripType 2136    Via 2121          Purpose 1654
```

### Inference and serving

`api.py` is a **FastAPI** service that loads the base model plus the LoRA adapter once at
startup (ASGI lifespan) and exposes:

- `POST /api/v1/ner/extract` — extract entities from a sentence (optional `threshold` and
  `entity_types` subset)
- `GET /api/v1/ner/health` — service/model status
- `GET /api/v1/ner/entity-types` — supported types with descriptions

Raw model output is passed through a **rule-based post-processing** layer that the project
adds on top of the model: it filters known false positives (verbs mislabelled as `TripType`,
locations mislabelled as `Purpose`/`Date`, non-place `Departure` spans) and performs
cross-type deduplication, keeping the highest-priority type when one span matches several.
Requests are validated with Pydantic v2 (max 500 chars), CORS is open for development, and
runtime is configurable via environment variables (`NER_MODEL_DIR`, `NER_THRESHOLD`,
`NER_HOST`, `NER_PORT`).

### What is third-party vs. built here

- **Third-party**: GLiNER2 and its mDeBERTa-v3-base encoder (`fastino/gliner2-multi-v1`), the
  GLiNER2 LoRA training utilities, FastAPI/uvicorn/Pydantic, and the local LLM used for data
  generation.
- **Built in this project**: the 8-entity domain schema, the synthetic data-generation and
  validation pipeline, the LoRA fine-tuning configuration and trained adapter, the
  post-processing/deduplication logic, the FastAPI service, and the evaluation harness.

## Results / status

Work-in-progress; the fine-tuned LoRA adapter trains end-to-end and serves through the API.
Quality is currently assessed **qualitatively** via `test_benin_transport_ner_v2.py`, which
runs 10 hand-labelled cases and reports exact / partial / incorrect matches per example.
**No formal precision/recall/F1 benchmark is committed to the repository**, and no held-out
test split is persisted, so headline accuracy numbers are not claimed here. The
post-processing layer was added specifically to suppress recurring false positives observed
during this qualitative testing. Establishing a labelled evaluation set and reporting span-level
F1 is the main open item (see Roadmap).

## Tech stack

- **Language**: Python 3.11
- **Modelling**: GLiNER2 (`gliner2`), mDeBERTa-v3-base encoder, LoRA, PyTorch
- **Serving**: FastAPI, uvicorn, Pydantic v2
- **Data generation**: Ollama (local LLM) + `requests`

## Repository structure

```
api.py                              FastAPI inference service
train_benin_transport_ner_v2.py     LoRA fine-tuning script
test_benin_transport_ner_v2.py      Qualitative evaluation harness
test_api.py                         HTTP client for the running API
train_gliner2_ner_10000_v2.jsonl    Training dataset (10,047 examples)
benin_transport_ner_model/          Trained LoRA adapter + checkpoints + training_config.json
requirements_api.txt                API runtime dependencies
Guide_Technique_FineTuning_GLiNER2_LoRA.md   In-depth technical write-up (French)
old/                                Earlier dataset-generation / training iterations
t-aia/                              Deployment copy (adapter + server start script)
```

## Setup & usage

Python 3.10+ recommended.

```bash
pip install -r requirements_api.txt
```

### Run the API

```bash
python api.py
# or
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Then open Swagger UI at `http://localhost:8000/docs` (ReDoc at `/redoc`).

Example request:

```bash
curl -X POST http://localhost:8000/api/v1/ner/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Je veux aller de Cotonou à Parakou demain à 8h avec 2 adultes"}'
```

Test the running API from another terminal:

```bash
python test_api.py
```

### Configuration (environment variables)

- `NER_MODEL_DIR` — path to the LoRA adapter (default `./benin_transport_ner_model/best`)
- `NER_THRESHOLD` — default confidence threshold (default `0.5`)
- `NER_HOST` / `NER_PORT` — API bind address (defaults `0.0.0.0` / `8000`)

### Retrain

The fine-tuning script loads `fastino/gliner2-multi-v1` and applies LoRA; it requires the
`gliner2` training extras and a GPU is recommended (FP16):

```bash
python train_benin_transport_ner_v2.py
```

Checkpoints and the best adapter are written to `./benin_transport_ner_model/`.

## Limitations / roadmap

- **No quantitative benchmark yet** — add a held-out labelled test set and report span-level
  precision/recall/F1.
- **Fully synthetic training data** — generated by an LLM and validated by rules; it has not
  been audited against real user queries and may under-represent genuine informal phrasing.
- **Heuristic post-processing** — the false-positive filters use hard-coded word/toponym lists
  that need maintenance and do not generalise beyond the current schema.
- Single-language (French) and single-domain (Benin transport) scope.
