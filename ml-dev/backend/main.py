from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import io
import h2o
import mlflow
import json
import logging
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType 
from typing import Optional
from utils.data_processing import preprocess_for_model, separate_id_col
from mlflow.exceptions import MlflowException

# Configuration initiale
app = FastAPI()
h2o.init()

# Configuration MLflow
MLFLOW_TRACKING_URI = "http://mlflow:5000"  # Utilisez le nom du service Docker
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

EXPERIMENT_NAME = "demomlops3-insurance-cross-sell"
ARTIFACT_ROOT = "file:///mlflow/artifacts" 

try:
    mlflow.set_experiment(EXPERIMENT_NAME)
except MlflowException:
    # L’expérience est supprimée ? => on la restaure.
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp and exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
    else:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(EXPERIMENT_NAME)

logger = logging.getLogger("uvicorn")

# Variable globale pour le modèle
best_model = None

def load_best_model():
    global best_model
    try:
        experiments = client.search_experiments()
        if not experiments:
            logger.warning("Aucune expérience MLflow trouvée")
            return None
            
        runs = mlflow.search_runs(
            experiment_ids=[e.experiment_id for e in experiments],
            filter_string="tags.environment='production'",
            run_view_type=ViewType.ACTIVE_ONLY
        )
        
        if runs.empty:
            logger.warning("Aucun run MLflow en production")
            return None
            
        best_run = runs.iloc[runs['metrics.log_loss'].idxmin()]
        model_uri = f"runs:/{best_run.run_id}/model"
        logger.info(f"Chargement du modèle : {model_uri}")
        return mlflow.h2o.load_model(model_uri)
        
    except Exception as e:
        logger.error(f"Erreur de chargement du modèle : {str(e)}")
        return None

# Charger le modèle au démarrage
best_model = load_best_model()

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target: str = "Response",
    max_models: int = 5,
    exclude_algos: Optional[str] = "GLM,DRF"
):
    try:
        # Lire et prétraiter les données
        contents = await file.read()
        train_df = pd.read_csv(io.BytesIO(contents))
        train_df = preprocess_for_model(train_df)
        
        # Convertir en H2OFrame
        train_h2o = h2o.H2OFrame(train_df)
        train_h2o[target] = train_h2o[target].asfactor()

        # Configurer AutoML
        excluded_algorithms = exclude_algos.split(",") if exclude_algos else []
        
        with mlflow.start_run(run_name="AutoML‑H2O"):
            mlflow.set_tag("environment", "production")             
            aml = h2o.automl.H2OAutoML(
                max_models=max_models,
                seed=42,
                exclude_algos=excluded_algorithms,
                sort_metric="logloss"
            )
            
            aml.train(y=target, training_frame=train_h2o)
            
            # Loguer le modèle et les métriques
            mlflow.log_params({
                "max_models": max_models,
                "target": target,
                "excluded_algos": exclude_algos
            })
            
            mlflow.log_metrics({
                "log_loss": aml.leader.logloss(),
                "auc": aml.leader.auc()
            })
            
            mlflow.h2o.log_model(aml.leader, "model")
            model_uri = mlflow.get_artifact_uri("model")
            
        # Recharger le meilleur modèle
        global best_model
        best_model = load_best_model()
        
        return {
            "status": "success",
            "model_uri": model_uri,
            "leaderboard": aml.leaderboard.as_data_frame().to_dict()
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not best_model:
        return JSONResponse(
            status_code=503,
            content={"error": "Aucun modèle entraîné - Veuillez d'abord entraîner un modèle"}
        )
    
    try:
        # Lire et prétraiter les données
        contents = await file.read()
        test_df = pd.read_csv(io.BytesIO(contents))
        test_df = preprocess_for_model(test_df)
        
        # Conversion en H2OFrame
        test_h2o = h2o.H2OFrame(test_df)
        
        # Prédictions
        preds = best_model.predict(test_h2o)
        predictions = preds.as_data_frame().to_dict()
        
        return {
            "predictions": predictions,
            "model_version": best_model.model_id,
            "average_probability": preds["p1"].mean(),
            "targeted_customers": sum(preds["p1"] > 0.5)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if best_model else "no_model",
        "mlflow_uri": MLFLOW_TRACKING_URI
    }