#!/bin/bash
# setup.sh
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies inside venv..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready."
echo "To use it, run: source .venv/bin/activate"
