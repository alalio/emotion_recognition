#!/bin/sh

echo "[Run Display]"
echo "    python3 src/random_display.py"
PYTHONPATH=/src:src:.:$PYTHONPATH python3 src/random_display.py
