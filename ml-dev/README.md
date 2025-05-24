# ML Development Environment

This project provides a complete ML development environment with GPU support, persistence, and full-stack integration. It includes:

- Development environment with NVIDIA GPU support
- FastAPI backend with H2O AutoML
- Streamlit frontend
- MLflow tracking with PostgreSQL backend
- Monitoring with Prometheus and Grafana
- Database management with pgAdmin

## Getting Started

1. Make sure you have Docker and Docker Compose installed
2. If you want to use GPU support, ensure you have NVIDIA Docker runtime installed
3. Run the environment with:

```bash
docker-compose up -d
```

## Available Services

| Service | URL | Description |
|---------|-----|-------------|
| Jupyter Lab | http://localhost:8888 | Development environment |
| Streamlit UI | http://localhost:8501 | Frontend application |
| FastAPI Backend | http://localhost:8000 | Backend API |
| MLflow | http://localhost:5000 | Experiment tracking |
| pgAdmin | http://localhost:8080 | Database management |
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3000 | Monitoring dashboards |

## Directory Structure

- `ml-dev/`: Development environment with Jupyter
- `backend/`: FastAPI and H2O service
- `frontend/`: Streamlit UI
- `mlflow/`: MLflow tracking server
- `prometheus/`: Metrics collection
- `grafana/`: Monitoring dashboards
- `postgres_data/`: PostgreSQL data persistence
- `pgadmin_data/`: pgAdmin configuration persistence

## Environment Configuration

The environment is configured to provide:
- GPU acceleration for ML tasks
- Persistent storage for all services
- Inter-container communication
- Health checks for service dependencies
- Resource limits for stability

## Customization

You can customize this environment by:
1. Modifying the Dockerfiles to add specific packages
2. Adjusting resource limits in docker-compose.yml
3. Adding your code to the appropriate directories
4. Extending the monitoring configuration
