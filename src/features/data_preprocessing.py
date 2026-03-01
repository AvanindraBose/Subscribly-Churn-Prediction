import pandas as pd
import numpy as np
import sys
import joblib
import logging
from pathlib import Path
from logger import create_log_path , CustomLogger
from sklearn.preprocessing import StandardScaler , OneHotEncoder , OrdinalEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime,timezone
# Steps Followed:
# Numerical Columns : Perform Standard Scaling
# Categorical Columns Except Subscription Type : Perform OHE
# Subscription Type : Perform Ordinal Encoding
TARGET = "Churn"
log_path = create_log_path("Data_Preprocessing")

data_preprocessing_logger = CustomLogger(
    logger_name="Data_Preprocessing",
    log_filename=log_path
)

data_preprocessing_logger.set_log_level(logging.INFO)

data_preprocessing_logger.save_logs(f"Data Preprocessing Pipeline Started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}", log_level='info')

def fetch_data(data_path:Path)-> pd.DataFrame:
    if not isinstance(data_path, Path):
        raise TypeError("data_path must be a pathlib.Path object")

    try:
        df = pd.read_csv(data_path)
        data_preprocessing_logger.save_logs(
            f"Successfully loaded data from {data_path}",
            "info"
        )
        return df

    except FileNotFoundError as e:
        data_preprocessing_logger.save_logs(
            f"File not found at path: {data_path}",
            "error"
        )
        raise FileNotFoundError(f"File not found at path: {data_path}") from e

    except Exception as e:
        data_preprocessing_logger.save_logs(
            f"Unexpected error while reading file {data_path}: {str(e)}",
            "critical"
        )
        raise

def save_transformer(path:Path , obj):
    try:
        joblib.dump(obj, path)
        data_preprocessing_logger.save_logs(
            f"Saved transformer at: {path}",
            "info"
        )
    except Exception as e:
        data_preprocessing_logger.save_logs(
            f"Failed to save transformer at {path}: {str(e)}",
            "critical"
        )
        raise

def transform_cols(num_cols:list , ohe_cols:list , ord_cols:list)-> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(('num', StandardScaler(), num_cols))
        data_preprocessing_logger.save_logs(
            f"Added StandardScaler for numerical columns: {num_cols}",
            "info"
        )
    if ohe_cols:
        transformers.append(('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ohe_cols))
        data_preprocessing_logger.save_logs(
            f"Added OneHotEncoder for categorical columns: {ohe_cols}",
            "info"
        )
    if ord_cols:
        transformers.append(('ord', OrdinalEncoder(categories=[["Basic","Standard","Premium"]]), ord_cols))
        data_preprocessing_logger.save_logs(
            f"Added OrdinalEncoder for ordinal columns: {ord_cols}",
            "info"
        )
    ct = ColumnTransformer(transformers=transformers)
    data_preprocessing_logger.save_logs(
        f"Created ColumnTransformer with transformers: {transformers}",
        "info"
    )
    return ct

def build_and_fit_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("Training DataFrame is empty.")

    if "Subscription Type" not in X.columns:
        data_preprocessing_logger.save_logs(
            "'Subscription Type' column missing in training data.",
            "critical"
        )
        raise KeyError("'Subscription Type' column is required.")

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    ohe_cols = X.select_dtypes(include=["object", "category"]).drop(
        columns=["Subscription Type"], errors="ignore"
    ).columns.tolist()
    ord_cols = ["Subscription Type"]

    data_preprocessing_logger.save_logs(
        f"Identified columns | Num: {num_cols} | OHE: {ohe_cols} | Ord: {ord_cols}",
        "info"
    )

    preprocessor = transform_cols(num_cols, ohe_cols, ord_cols)
    preprocessor.set_output(transform="pandas")

    try:
        preprocessor.fit(X)
        data_preprocessing_logger.save_logs(
            "Preprocessor successfully fitted on training data.",
            "info"
        )
    except Exception as e:
        data_preprocessing_logger.save_logs(
            f"Error fitting preprocessor: {str(e)}",
            "critical"
        )
        raise

    return preprocessor

def save_to_csv(df:pd.DataFrame , path_to_save:Path):
    df.to_csv(path_to_save,index=False)
    data_preprocessing_logger.save_logs(
        f"Saved processed data to path: {path_to_save}",
        "info"
    )

def transform(transformer:ColumnTransformer,df:pd.DataFrame)-> pd.DataFrame:
    try:
        transformed_df = transformer.transform(df)
        data_preprocessing_logger.save_logs(
        f"Transformed data using the fitted transformer",
        "info"
        )
        return transformed_df
    except Exception as e:
        data_preprocessing_logger.save_logs(
            f"Error during transformation: {str(e)}",
            "critical"
        )
        raise

def load_transformer(path:Path)-> ColumnTransformer:
    if not path.exists():
        data_preprocessing_logger.save_logs(
            f"Transformer file not found at {path}",
            "critical"
        )
        raise FileNotFoundError(f"Transformer not found at {path}")

    try:
        transformer = joblib.load(path)
        data_preprocessing_logger.save_logs(
            f"Loaded transformer from {path}",
            "info"
        )
        return transformer
    except Exception as e:
        data_preprocessing_logger.save_logs(
            f"Error loading transformer from {path}: {str(e)}",
            "critical"
        )
        raise

def main(data_path:Path , file_name : str , transformers_path:Path)-> pd.DataFrame:
    data_preprocessing_logger.save_logs(
        f"Started processing file: {file_name}",
        "info"
    )
    df = fetch_data(data_path)
    if file_name == "train.csv":
        if TARGET not in df.columns:
            raise KeyError(f"Target column '{TARGET}' missing in training data.")
        
        data_preprocessing_logger.save_logs(
            f"Processing file: {file_name}",
            "info"
        )
        
        X = df.drop(columns=TARGET)
        y = df[TARGET]
        preprocessor = build_and_fit_preprocessor(X)
        save_transformer(transformers_path / "preprocessor.joblib", preprocessor)
        X_trans = transform(preprocessor,X)
        X_trans[TARGET] = y.values
        return X_trans
    
    elif file_name == "val.csv" :

        if TARGET not in df.columns:
            raise KeyError(f"Target column '{TARGET}' missing in validation data.")
        
        data_preprocessing_logger.save_logs(
            f"Processing file: {file_name}",
            "info"
        )
        X = df.drop(columns=TARGET)
        y = df[TARGET]
        preprocessor = load_transformer(transformers_path/"preprocessor.joblib")
        X_trans = transform(preprocessor,X)
        X_trans[TARGET] = y.values

        return X_trans
    
    else :

        data_preprocessing_logger.save_logs(
            f"Processing file: {file_name}",
            "info"
        )
        preprocessor = load_transformer(transformers_path/"preprocessor.joblib")
        df_transformed = transform(preprocessor,df)

        return df_transformed

if __name__ == "__main__":
    for input_file_path in sys.argv[1:]:
        curr_path = Path(__file__)
        root_path = curr_path.parent.parent.parent
        output_path = root_path/'data'/'processed'/'final'
        output_path.mkdir(parents=True,exist_ok=True)
        transformers_path = root_path/'models'/'transformers'
        transformers_path.mkdir(parents=True,exist_ok=True)
        data_path = root_path / input_file_path
        file_name = data_path.parts[-1]
        df_transformed = main(data_path , file_name , transformers_path)
        save_to_csv(df_transformed,output_path/file_name)



