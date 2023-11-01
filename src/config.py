import logging
from datetime import datetime

# Get the current date to create a dynamic log filename
current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = f'lung_cancer_detection_{current_date}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
