#!/bin/bash
set -e

if [ "$AUTO_TRAIN" = "true" ]; then
    echo "Running model training before starting server..."
    python train.py
fi

echo "Starting server with uvicorn..."
exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8003} --workers ${WORKERS:-1}
