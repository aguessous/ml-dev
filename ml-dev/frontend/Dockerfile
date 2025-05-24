FROM python:3.9-slim

# Répertoire de travail
WORKDIR /app

# Dépendances système minimales
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Port Streamlit
EXPOSE 8501

# Lancement de l'application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]