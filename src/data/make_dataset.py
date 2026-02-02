import sys
import logging
import pandas as pd
from yaml import safe_load
from logger import CustomLogger,create_log_path
from sklearn.model_selection import train_test_split
from pathlib import Path

log_file_path = create_log_path("Make Dataset")
dataset_logger = CustomLogger(
    logger_name="Training",
    log_filename=log_file_path
)

dataset_logger.set_log_level(level=logging.INFO)


def load_raw_data(input_path:Path) -> pd.DataFrame:
    raw_data = pd.read_csv(input_path)
    rows,columns = raw_data.shape
    dataset_logger.save_logs(msg = f"{input_path.stem} data read having {rows} rows and {columns} columns",
                             log_level="info")
    
    return raw_data

def train_val_split(data:pd.DataFrame , test_size:float , random_state:int) -> tuple[pd.DataFrame , pd.DataFrame]:
    train_data,val_data = train_test_split(data ,
                                           test_size=test_size,
                                           random_state=random_state
                                )
    dataset_logger.save_logs(msg=f'Data is split into train split with shape {train_data.shape} and val split with shape {val_data.shape}',
                             log_level='info')
    dataset_logger.save_logs(msg=f'The parameter values are {test_size} for test_size and {random_state} for random_state',
                             log_level='info')
    
    return train_data,val_data

def read_params(input_file):
    try:
        with open(input_file) as f :
            params_file = safe_load(f)
    except :
        dataset_logger.save_logs(msg = "Parameters File Not Found Switching To Default Values",
                                 log_level='error')
        default_dict = {'test_size' : 0.2 ,
                        'random_state':42}
        
        test_size = default_dict['test_size']
        random_state = default_dict['random_state']

        return test_size , random_state
    else :
        dataset_logger.save_logs(msg = "Paramters File Read Successfully",
                                 log_level='info')
        test_size = params_file['make_dataset']['test_size']
        random_state = params_file['make_dataset']['random_state']

        return test_size , random_state

def save_data_path(input_df : pd.DataFrame , output_path:Path):
    input_df.to_csv(output_path , index = False)
    dataset_logger.save_logs(msg=f'{output_path.stem + output_path.suffix} data saved successfully to the output folder',
                             log_level='info')


def main():
    input_file_name = sys.argv[1]
    current_path = Path(__file__)

    root_path = current_path.parent.parent.parent

    interim_data_path = root_path/'data'/'interim'
    interim_data_path.mkdir(exist_ok=True)
    
    raw_df_path = root_path/'data'/'raw'/input_file_name
    test_size , random_state = read_params('params.yaml')
    raw_df = load_raw_data(input_path = raw_df_path)
    train_df , val_df = train_val_split(raw_df , test_size ,random_state)
    train_df_path = interim_data_path / 'train.csv'
    val_df_path = interim_data_path/'val.csv'

    save_data_path(train_df , train_df_path)
    save_data_path(val_df , val_df_path)



if __name__ == "__main__":
    main()
