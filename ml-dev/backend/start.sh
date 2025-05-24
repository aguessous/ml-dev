#!/usr/bin/env bash
set -e

# Récupère la première adresse IP du conteneur
IP_ADDR=$(hostname -I | awk '{print $1}')

# Localise dynamiquement le JAR H2O dans le site-packages actif
H2O_JAR=$(python3 - << 'EOF'
import glob, site
sp = site.getsitepackages()[0]
matches = glob.glob(f"{sp}/h2o/backend/bin/h2o.jar")
print(matches[0] if matches else "")
EOF
)

echo "H2O jar path: $H2O_JAR"
echo "Starting H2O server on $IP_ADDR:54321..."
java -jar "$H2O_JAR" \
     -port 54321 \
     -ip "$IP_ADDR" \
     -name automl-cluster &

echo "Starting FastAPI..."
exec uvicorn main:app \
     --host 0.0.0.0 \
     --port 8000 \
     --workers 1
