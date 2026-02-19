#!/bin/bash

# Activation de l'environnement conda
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate aia

# Navigation vers le répertoire du projet
cd /root/projects/ner/t-aia

# Démarrage de l'API NER avec uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
