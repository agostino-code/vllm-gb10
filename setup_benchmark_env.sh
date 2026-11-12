#!/bin/bash
# Script per creare un ambiente virtuale e installare le dipendenze per il benchmark

VENV_DIR="venv"

echo "--- Creazione virtual environment in $VENV_DIR ---"
python3 -m venv $VENV_DIR

echo "--- Attivazione ambiente e installazione dipendenze ---"
source $VENV_DIR/bin/activate

pip install --upgrade pip
pip install openai

echo "--- Ambiente pronto ---"
echo "Per attivare l'ambiente, esegui: source $VENV_DIR/bin/activate"
