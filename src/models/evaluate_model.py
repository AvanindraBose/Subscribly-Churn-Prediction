import mlflow
import sys
import logging
import pandas as pd
from logger import create_log_path , CustomLogger
from datetime import datetime , timezone
from sklearn.metrics import roc_auc_score
from mlflow.tracking import MlflowClient
from pathlib import Path
from yaml import safe_load
from mlflow.pyfunc import PyFuncModel
import mlflow.sklearn
from sklearn.metrics import confusion_matrix
# Steps to be followed : 
# 1. Remember since you have logged the entire pipeline of data preprocessing and the model so you have to pass
# the test dataset that is saved in data/processed/built/test.csv

TARGET = 'Churn'

log_file_path = create_log_path("Evaluate_Model")

evaluate_model_logger = CustomLogger(
    logger_name="Model Evaluation",
    log_filename=log_file_path
)

evaluate_model_logger.set_log_level(logging.INFO)

evaluate_model_logger.save_logs(f"Model Evaluation Pipeline started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}", log_level='info')

def load_df(path:Path , file_name : str) -> pd.DataFrame:
    try :
        df = pd.read_csv(path)
    except FileNotFoundError:
        evaluate_model_logger.save_logs(f"File Not found from path {path}: File Not Found" , log_level='error')
        raise
    except pd.errors.EmptyDataError:
        evaluate_model_logger.save_logs(f"Error loading dataset from {path}: Empty data", log_level='error')
        raise
    except Exception as e:
        evaluate_model_logger.save_logs(f"Unexpected error at loading {path} : {e}",log_level='error')
    else :
        evaluate_model_logger.save_logs(f"Successfully loaded {file_name} from path {path}",log_level='info')
        return df

def load_config(config_path : Path)-> dict:
    try :
        with open(config_path) as f :
            config = safe_load(f)
    except FileNotFoundError :
        evaluate_model_logger.save_logs(f"File not found at path {config_path}",log_level='error')
        raise
    except Exception as e :
        evaluate_model_logger.save_logs(f"Unexpected erro occured at {config_path}",log_level='error')
    else :
        evaluate_model_logger.save_logs(f"Successfully Loaded the file from {config_path}",log_level='info')
        return config

def load_model(model_name : str):
    try : 
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e :
        evaluate_model_logger.save_logs(f"Error Fetching the Models in {model_uri} : {e}",log_level='error')
        raise
    else :
        evaluate_model_logger.save_logs("Loading Production model from {model_uri}",log_level='info')
        return model

def main():
    if len(sys.argv) != 2 :
        evaluate_model_logger.save_logs("Incorrect Script Path has been Provided")
        raise ValueError("Usage: python evaluate_test.py <relative_path_to_test_csv>")
    
    # getting the path of current directory
    curr_path = Path(__file__)
    root_path = curr_path.parent.parent.parent

    #  fetching data path
    test_file_path = Path(sys.argv[1])
    file_name = test_file_path.parts[-1]

    # loading test_df
    test_df = load_df(test_file_path , file_name)

    if TARGET not in test_df.columns:
        evaluate_model_logger.save_logs(f"{TARGET} column not found in the dataset",log_level='error')
        raise ValueError(f"{TARGET} column not found in test dataset")

    # Preparing X and Y

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    # loading config file
    config_file_path = root_path / "params.yaml"
    config = load_config(config_file_path)

    #  Fetch model name and experiment name
    model_name = config.get("experiment_info",{}).get("model_name")
    exp_name = config.get("experiment_info",{}).get("experiment_name")

    # Load Production ML Model for Testing
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()

    model = load_model(model_name)
    # Getting Predictions
    
    y_prob = model.predict_proba(X_test)
    y_prob = y_prob[:,1]
    test_roc_auc = roc_auc_score(y_test, y_prob)
    evaluate_model_logger.save_logs(f"Test ROC-AUC: {test_roc_auc:.6f}",log_level='info')

    mlflow.set_experiment(f"{exp_name} v1")
    with mlflow.start_run(
        run_name= f"test_evaluation_{model_name}_{datetime.now(timezone.utc).strftime(('%Y%m%d_%H%M%S'))}"
    ):
        # Logging params to mlflow
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_param("evaluated_stage", "Production")
    
    evaluate_model_logger.save_logs(f"Successfully Ran the Evaluate Model Pipeline at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')} and logged the metrics and params to mlflow",
                                    log_level='info')

if __name__ == "__main__":
    main()
