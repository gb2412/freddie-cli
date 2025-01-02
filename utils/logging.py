import psutil
import logging
# Set logging configurations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datetime import datetime

# Define color class for pretty logging
class Colors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ITER = '\033[36m'
    ENDC = '\033[0m'

# Print memory usage
def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    logging.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

def log_info(message):
    logging.info(message)