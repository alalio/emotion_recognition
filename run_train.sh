#!/bin/sh

PYTHONPATH=$PYTHONPATH:src/

echo "python3 src/train_model.py"
PYTHONPATH=src:$PYTHONPATH python3 src/train_model.py

