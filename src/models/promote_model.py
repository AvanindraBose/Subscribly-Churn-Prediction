import sys
import logging
import mlflow
from pathlib import Path
from yaml import safe_load
from datetime import datetime,timezone
from mlflow.tracking import MlflowClient
from logger import create_log_path,CustomLogger
from mlflow.entities.model_registry import ModelVersion
from typing import Optional

TARGET_METRIC = 'roc_auc'

log_file_path = create_log_path("Promote_Model")
promote_model_logger = CustomLogger(
    logger_name="Model Promotion",
    log_filename=log_file_path
)

promote_model_logger.set_log_level(logging.INFO)
promote_model_logger.save_logs(f"Model Promotion Pipeline Started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}", log_level='info')
def load_config(config_path:Path)-> dict:
    try:
        with open(config_path) as f:
            config = safe_load(f)
    except FileNotFoundError:
        promote_model_logger.save_logs(f"Configuration file not found at {config_path}", log_level='error')
        raise
    except Exception as e:
        promote_model_logger.save_logs(f"Error loading configuration from {config_path}: {e}", log_level='error')
        raise
    else :
        promote_model_logger.save_logs(f"Configuration loaded successfully from {config_path}", log_level='info')
        return config

def get_production_version(client : MlflowClient , model_name : str) -> Optional[ModelVersion]:
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            if v.current_stage == "Production":
                return v
        promote_model_logger.save_logs(f"No production version found for model {model_name}", log_level='warning')
        return None
    except Exception as e:
        promote_model_logger.save_logs(f"Error retrieving production version for model {model_name}: {e}", log_level='error')
        raise

def get_metric_from_run(client : MlflowClient , run_id:str , metric_name:str) -> float:
    try :
        run = client.get_run(run_id)
        metrics = run.data.metrics
        if metric_name not in metrics:
            promote_model_logger.save_logs(f"Metric {metric_name} not found in run {run_id}", log_level='error')
            raise ValueError(f"Metric {metric_name} not found in run {run_id}")
        return metrics[metric_name]
    except Exception as e:
        promote_model_logger.save_logs(f"Error retrieving metric {metric_name} from run {run_id}: {e}", log_level='error')
        raise

def get_latest_version(client : MlflowClient ,model_name:str) -> ModelVersion:
    try :
        versions = client.get_latest_versions(model_name)
        if not versions:
            promote_model_logger.save_logs(f"No versions found for model {model_name}", log_level='error')
            raise ValueError(f"No versions found for model {model_name}")
        best_version = max(versions, key=lambda v: int(v.version))
        promote_model_logger.save_logs(f"Latest version retrieved successfully: Version {best_version.version}", log_level='info')
        return best_version
    except Exception as e:
        promote_model_logger.save_logs(f"Error retrieving latest version for model {model_name}: {e}", log_level='error')
        raise

def promote_model(client : MlflowClient , model_name : str , candidate_version : ModelVersion, production_version :  Optional[ModelVersion] = None) -> None:
    try : 
        if production_version :
            client.transition_model_version_stage(
                name = model_name ,
                version = production_version.version,
                stage="Archived"
            )
            
            promote_model_logger.save_logs(f"Archived previous production version: Version {production_version.version}", log_level='info')

        client.transition_model_version_stage(
        name=model_name,
        version=candidate_version.version,
        stage="Production"
        )
        promote_model_logger.save_logs(f"Promoted version {candidate_version.version} to Production successfully", log_level='info')
    except Exception as e:
        promote_model_logger.save_logs(f"Error promoting version {candidate_version.version} to Production: {e}", log_level='error')
        raise


def main():
    root_path = Path(__file__).parent.parent.parent
    config = load_config(root_path / "params.yaml")
    model_name = config.get("experiment_info",{}).get("model_name")
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()

    
    if len(sys.argv) > 1:
        version_number = sys.argv[1]
        candidate = client.get_model_version(model_name, version_number)
        promote_model_logger.save_logs(f"Using specified version: {candidate.version}", log_level='info')
    else:
        candidate = get_latest_version(client, model_name)
        promote_model_logger.save_logs(f"No version specified. Using latest version: {candidate.version}", log_level='info')
    
    production = get_production_version(client,model_name)
    candidate_auc = get_metric_from_run(client, candidate.run_id, TARGET_METRIC)

    promote_model_logger.save_logs(f"Candidate version {candidate.version} has {TARGET_METRIC}: {candidate_auc}", log_level='info')

    if production is None:
        promote_model_logger.save_logs(f"No Production model found. Promoting version {candidate.version} to Production.", log_level='info')
        promote_model(client, model_name, candidate)
        return
    
    production_auc = get_metric_from_run(client, production.run_id, TARGET_METRIC)

    promote_model_logger.save_logs(f"Current Production version {production.version} has {TARGET_METRIC}: {production_auc}", log_level='info')

    if candidate_auc > production_auc :
        promote_model_logger.save_logs(f"Candidate version {candidate.version} outperforms current Production version {production.version}. Promoting candidate to Production.", log_level='info')
        promote_model(client, model_name, candidate, production)
    else :
        promote_model_logger.save_logs(f"Candidate version {candidate.version} does not outperform current Production version {production.version}. No promotion performed.", log_level='info')

if __name__ == "__main__":
    main()
