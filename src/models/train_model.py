import logging
import pandas as pd
import sys
import mlflow
import mlflow.sklearn
import joblib
import subprocess
from sklearn.pipeline import Pipeline
from datetime import datetime , timezone
from logger import create_log_path,CustomLogger
from pathlib import Path
from xgboost import XGBClassifier
from yaml import safe_load
from sklearn.metrics import roc_auc_score

TARGET = 'Churn'
log_file_path = create_log_path("train_model")

train_model_logger = CustomLogger(
    logger_name="Model Training",
    log_filename=log_file_path
)

train_model_logger.set_log_level(logging.INFO)

def load_dataset(file_path:Path)-> pd.DataFrame:
    try :
        data = pd.read_csv(file_path)
        train_model_logger.save_logs(
            f"Dataset Loaded Successfully from {file_path} with shape {data.shape}",
            log_level='info'
        )
        return data
    except FileNotFoundError:
        train_model_logger.save_logs(f"Error loading dataset from {file_path}: File not found", log_level='error')
        raise
    except pd.errors.EmptyDataError:
        train_model_logger.save_logs(f"Error loading dataset from {file_path}: Empty data", log_level='error')
        raise
    except Exception as e:
        # To Catch unexpected errors
        train_model_logger.save_logs(
            f"Unexpected error loading {file_path}: {e}",
            log_level='error',
            exc_info=True 
        )
        raise

def get_git_commit()-> str :
    try :
        commit_hash = subprocess.check_output(
            ["git",
             "rev-parse",
             "HEAD"]
        ).decode("utf-8").strip()
        train_model_logger.save_logs(f"Git commit hash retrieved successfully: {commit_hash}", log_level='info')
        return commit_hash
    except Exception :
        train_model_logger.save_logs("Error retrieving git commit hash. Ensure this code is run within a git repository.", log_level='error', exc_info=True)
        return "Unknown"

def get_exp_info(input_file)-> dict:
    try :
        with open(input_file) as f :
            exp_info = safe_load(f)
    except FileNotFoundError:
        train_model_logger.save_logs(f"Experiment info file {input_file} not found. Using default experiment info.", log_level='error')
        default_exp_info = {
            'experiment_name': 'Default Experiment',
            'model_name' : 'XGBoost'
        }
        return default_exp_info
    except Exception as e :
        train_model_logger.save_logs(f"Unexpected error reading experiment info from {input_file}: {e}", log_level='error', exc_info=True)
        raise
    else :
        train_model_logger.save_logs(f"Experiment info loaded successfully from {input_file}", log_level='info')
        return exp_info["experiment_info"]

def get_params(input_file)-> dict :
    try :
        with open(input_file) as f :
            params = safe_load(f)
    except FileNotFoundError:
        train_model_logger.save_logs(f"Parameter file {input_file} not found. Using the default hyperparameters.", log_level='error')
        default_params = {
            'n_estimators':200,
            'max_depth':4,
            'larning_rate':0.1,
            'subsample':0.8,
            'colsample_bytree':0.8,
            'n_jobs':-1
        }

        return default_params
    
    except Exception as e :
        train_model_logger.save_logs(f"Unexpected error reading parameters from {input_file}: {e}", log_level='error', exc_info=True)
        raise
    else :
        train_model_logger.save_logs(f"Parameters loaded successfully from {input_file}", log_level='info')
        return params['model_params']

def load_preprocessor(preprocessor_path : Path)-> Pipeline:
    try :
        preprocessor = joblib.load(preprocessor_path)
    except FileNotFoundError:
        train_model_logger.save_logs(f"Preprocessor file {preprocessor_path} not found. Ensure the preprocessor is trained and saved correctly.", log_level='error')
        raise
    except Exception as e :
        train_model_logger.save_logs(f"Unexpected error loading preprocessor from {preprocessor_path}: {e}", log_level='error', exc_info=True)
        raise
    else :
        train_model_logger.save_logs(f"Preprocessor Loaded successfully from {preprocessor_path}", log_level='info')
        return preprocessor

def validate_schema(train_df:pd.DataFrame , val_df:pd.DataFrame):
    train_cols = set(train_df.drop(columns=TARGET).columns)
    val_cols = set(val_df.drop(columns=TARGET).columns)

    if train_cols != val_cols:
        train_model_logger.save_logs(
            "Schema mismatch between train and validation datasets.",
            log_level='error'
        )
        raise ValueError("Train and Validation feature columns do not match.")

    train_model_logger.save_logs(
        "Schema validation successful. Train and validation features match.",
        log_level='info'
    )

def get_model_instance(input_file)-> XGBClassifier:
    params = get_params(input_file)
    model = XGBClassifier(**params)
    train_model_logger.save_logs(f"XGBoost model instance created with parameters: {params}", log_level='info')
    return model

def evaluate_model(model: XGBClassifier , X_val : pd.DataFrame , y_val : pd.Series)-> float:
    try :
        y_pred_proba = model.predict_proba(X_val)[:,1]
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        train_model_logger.save_logs(f"Model evaluation completed. ROC AUC Score: {roc_auc:.4f}", log_level='info')
        return roc_auc
    except Exception as e :
        train_model_logger.save_logs(f"Error during model evaluation: {e}", log_level='error', exc_info=True)
        raise

def main():
        if len(sys.argv) != 3:
            raise ValueError("Usage: python train_model.py <train_path> <val_path>")
        
        root_path = Path(__file__).parent.parent
        train_file_path = Path(sys.argv[1])
        val_file_path = Path(sys.argv[2])
        preprocessor_path = root_path / "models"/ "transformers" / "preprocessor.joblib"
        train_df = load_dataset(train_file_path)
        val_df = load_dataset(val_file_path)

        if TARGET not in train_df.columns or TARGET not in val_df.columns:
            train_model_logger.save_logs(f"Target column '{TARGET}' missing in dataset. Please ensure both training and validation datasets contain the target column.", log_level='error')
            raise ValueError(f"Target column '{TARGET}' missing in dataset.")

        validate_schema(train_df, val_df)
        X_train = train_df.drop(columns=[TARGET])
        y_train = train_df[TARGET]

        X_val = val_df.drop(columns=[TARGET])
        y_val = val_df[TARGET]

        experiment_name , model_name = get_exp_info("params.yaml")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(experiment_name)
        git_commit_hash = get_git_commit()
        with mlflow.start_run(run_name=f"Train_{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{git_commit_hash}") as run:
            model = get_model_instance("params.yaml")
            model.fit(X_train,y_train)
            train_model_logger.save_logs("Model training completed successfully", log_level='info')
            #  Evaluate
            roc_auc = evaluate_model(model,X_val,y_val)

            print(f"Validation ROC AUC Score: {roc_auc:.4f}")

            #  Log Params 
            mlflow.log_params(model.get_params())

            #  Log Metric
            mlflow.log_metric("roc_auc", roc_auc)

            # Log extra metadata
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("feature_count", X_train.shape[1])
            mlflow.set_tag("training_time", datetime.now(timezone.utc).isoformat())
            mlflow.set_tag("git_commit", git_commit_hash)

            preprocessor = load_preprocessor(preprocessor_path)

            final_pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor), 
                    ("model", model)
                ]
            )

            mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model",
            registered_model_name=model_name
            )

            train_model_logger.save_logs(
            f"Model logged and registered under name: {model_name}",
            log_level='info'
            )

            print("\nModel successfully logged and registered.\n")


if __name__ == "__main__":
    main()