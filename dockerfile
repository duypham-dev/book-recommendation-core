FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build tools for scientific packages and clean up cache
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential libpq-dev pkg-config cargo curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first so Docker can cache installs
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip maturin \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download SBERT model so runtime doesn't rely on HF network
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('keepitreal/vietnamese-sbert')"

# Copy application source
COPY . .

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Ensure artifact folders exist (volumes may mount over them)
RUN mkdir -p /app/artifacts /app/artifacts /app/

# Environment variables
ENV APP_MODULE=server:app \
    HOST=0.0.0.0 \
    PORT=8003 \
    WORKERS=1 \
    AUTO_TRAIN=false

# Port configuration (can be overridden)
EXPOSE 8003

# Use entrypoint script to optionally train models before starting server
ENTRYPOINT ["./entrypoint.sh"]
