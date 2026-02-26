import logging
import pandas as pd
import sys
from logger import create_log_path,CustomLogger
from pathlib import Path
from xgboost import XGBClassifier
from yaml import safe_load

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

def get_model_instance(input_file)-> XGBClassifier:
    params = get_params(input_file)
    model = XGBClassifier(**params)
    train_model_logger.save_logs(f"XGBoost model instance created with parameters: {params}", log_level='info')
    return model

def main():
    input_file_path = Path(sys.argv[1])
    data = load_dataset(input_file_path)
    X = data.drop(columns=TARGET)
    y = data[TARGET]
    model = get_model_instance("params.yaml")
    model.fit(X,y)
    train_model_logger.save_logs("Model training completed successfully", log_level='info')

if __name__ == "__main__":
    main()