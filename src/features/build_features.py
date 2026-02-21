import sys
import pandas as pd
from pathlib import Path
from logger import create_log_path, CustomLogger
from yaml import safe_load
# Steps Followed Here : 
# 1. Create Bucketed Age Column
# 2. Remove Age Column
# 3. Store the modified data in the processed / built folder for further transfromations
log_file_path = create_log_path("Build_Features")
build_logger = CustomLogger(
    logger_name='Build_Features',
    log_filename=log_file_path
)

def read_params(input_file:str)-> list:
    try:
        with open(input_file) as f:
            param_file = safe_load(f)
    except:
        build_logger.save_logs(msg = "Parameters File Not Found Switching To Default Values",
                                 log_level='error')
        default_bins = [0,18, 25, 35, 45, 55, 70, 120]
        return default_bins
    else:
        build_logger.save_logs(msg = "Parameters File Read Successfully",
                                 log_level='info')
        bins = param_file['build_features']['bins']
        return bins

def create_age_buckets(df : pd.DataFrame , param_file:str)-> pd.DataFrame:
    bins = read_params(param_file)

    df['age_bucket'] = pd.cut(
        df['Age'],
        bins=bins,
        right = False
    )

    return df

def remove_age_column(df:pd.DataFrame)-> pd.DataFrame:
    df.drop(columns=['Age'],inplace = True)
    return df

def repositioning(df:pd.DataFrame)->pd.DataFrame:
    col_to_add = df.pop('age_bucket')
    df.insert(10,'age_bucket',col_to_add)
    return df

def fetch_data(data_path : Path) -> pd.DataFrame:
    fetched_df = pd.read_csv(data_path)
    return fetched_df

def save_to_csv(df:pd.DataFrame , output_path : Path):
    df.to_csv(output_path,index = False)

def main(data_path : Path , file_name:str , param_file : str) -> pd.DataFrame:
    df = fetch_data(data_path)
    df_with_buckets = create_age_buckets(df , param_file)
    df_repositioned = repositioning(df_with_buckets)
    df_with_removed_age_column = remove_age_column(df_repositioned)
    
    return df_with_removed_age_column

if __name__ == "__main__":
    for input_file_path in sys.argv[1:]:
        curr_path = Path(__file__)
        root_path = curr_path.parent.parent.parent
        output_path = root_path / 'data' / 'processed' / 'built'
        output_path.mkdir(parents=True,exist_ok=True)
        param_file = 'params.yaml'
        data_path = root_path / input_file_path
        file_name = data_path.parts[-1]
        df_modified = main(data_path,file_name,param_file)
        save_to_csv(df_modified , output_path / file_name)


      


