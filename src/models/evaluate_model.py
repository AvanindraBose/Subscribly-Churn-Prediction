import mlflow
import logging
from logger import create_log_path , CustomLogger
from datetime import datetime , timezone

log_file_path = create_log_path("Evaluate_Model")

evaluate_model_logger = CustomLogger(
    logger_name="Model Evaluation",
    log_filename=log_file_path
)

evaluate_model_logger.set_log_level(logging.INFO)

evaluate_model_logger.save_logs(f"Model Evaluation Pipeline started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}", log_level='info')

def main():
    print("Hi Just Started")

if __name__ == "__main__":
    main()
