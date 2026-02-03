import sys
import logging
import pandas as pd
from typing import List
import numpy as np
from pathlib import Path
from logger import create_log_path , CustomLogger
# Steps Followed Here : 
# 1. Remove Customer ID from all the extracts
# 2. Drop Null Values (Only 1 null row missing so far)
# 3. Drop Duplicate (Only 1 values seen so far)
# 4. Convert All the Numerical Columns to int except Total Spend
EXC_COLUMN = 'Total Spend'
COL_TO_BE_DELETED = 'CustomerID'
log_file_path = create_log_path("Modify_Features")
modify_logger = CustomLogger(
    logger_name='Modify_Features',
    log_filename=log_file_path
)

modify_logger.set_log_level(logging.INFO)

def remove_customer_id(df : pd.DataFrame):
    df.drop(columns=[COL_TO_BE_DELETED] , inplace = True)
    return df

def drop_nulls(df):
    df.dropna(inplace = True)
    return df

def remove_ts(cols : pd.DataFrame):
    cols.drop(columns = [EXC_COLUMN] , inplace =True)
    return cols

def convert(df : pd.DataFrame , num_cols : List) -> pd.DataFrame:
    for col in num_cols :
        df[col] = df[col].astype(int)
    
    return df

def get_num_cols(df:pd.DataFrame)-> List:
    numerical_columns = df.select_dtypes(include=np.number)
    final_numerical_columns = remove_ts(numerical_columns)
    final_numerical_columns = final_numerical_columns.columns.to_list()
    
    return final_numerical_columns


def input_modifications(df : pd.DataFrame):
    #  Removing Customer ID
    df = remove_customer_id(df)
    # Removing Null Values
    df = drop_nulls(df)
    # Converting float to int for all the columns except Total spend
    #  Get all the int cols
    num_cols = get_num_cols(df)

    df = convert(df , num_cols)

    return df
    
def fetch_data(data_path : Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df
def main(data_path:Path , file_name : str) -> pd.DataFrame:
    df = fetch_data(data_path)
    df_input_modified = input_modifications(df)

    return df_input_modified

def save_df(df : pd.DataFrame , output_path : Path):
    df.to_csv(output_path)

if __name__ == "__main__":
    for ind in range(1,4):
        input_file_path = sys.argv[ind]
        curr_path = Path(__file__)
        root_path = curr_path.parent.parent.parent
        output_path = root_path / 'data' / 'processed'
        data_path = root_path / input_file_path
        file_name = data_path.parts[-1]
        df_final = main(data_path , file_name)
        save_df(df_final , output_path / file_name)

        