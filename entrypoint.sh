#!/bin/bash
set -e

echo "=========================================="
echo "Book Recommendation System - Starting"
echo "=========================================="

# Check if we should train models on startup
if [ "$AUTO_TRAIN" = "true" ]; then
    echo "AUTO_TRAIN enabled - checking for models..."
    
    MODEL_DIR="./artifacts"
    TRAIN_SCRIPT="train.py"
    MODEL_NAME="Implicit SBERT"
    
    # Check if model artifacts exist
    if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
        echo "⚠️  $MODEL_NAME model not found in $MODEL_DIR"
        echo "🚂 Training model(s)..."
        
        if [ -z "$DB_PASSWORD" ]; then
            echo "❌ ERROR: DB_PASSWORD environment variable is required for training"
            exit 1
        fi
        
        # Train the model
        python $TRAIN_SCRIPT
        
        if [ $? -eq 0 ]; then
            echo "✅ Model training completed successfully!"
        else
            echo "❌ Model training failed!"
            exit 1
        fi
    else
        echo "✅ $MODEL_NAME model found in $MODEL_DIR"
    fi
fi

# Start the application
echo "🚀 Starting server: $APP_MODULE"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Workers: $WORKERS"
echo "=========================================="

exec uvicorn $APP_MODULE --host $HOST --port $PORT --workers $WORKERS
