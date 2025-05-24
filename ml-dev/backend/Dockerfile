FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Python + outils de base
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip                 \
        openjdk-17-jre-headless     \
        curl && \
        apt-get clean && rm -rf /var/lib/apt/lists/*


# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (exclude unnecessary files with .dockerignore)
COPY . .

# Create necessary directories (si vraiment nécessaire)
RUN mkdir -p /app/data /app/mlruns

# Expose ports for FastAPI and H2O
EXPOSE 8000 54321

# Copy and prepare startup script
RUN chmod +x /app/start.sh

# Healthcheck (ajout recommandé)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# Execute startup script when container starts
CMD ["/app/start.sh"]