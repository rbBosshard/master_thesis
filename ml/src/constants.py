import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(ROOT_DIR, '../config')
CONFIG_PATH = os.path.join(CONFIG_DIR_PATH, 'config_ml.yaml')

LOG_DIR_PATH = os.path.join(ROOT_DIR, '../logs')
DATA_DIR_PATH = os.path.join(ROOT_DIR, '../data')
EXPORT_DIR_PATH = os.path.join(DATA_DIR_PATH, 'export')
CSV_DIR_PATH = os.path.join(EXPORT_DIR_PATH, 'csv')
INPUT_DIR_PATH = os.path.join(DATA_DIR_PATH, 'input')
OUTPUT_DIR_PATH = os.path.join(DATA_DIR_PATH, 'output')

REMOTE_DIR_PATH = os.path.join(ROOT_DIR, '../../../pytcpl/data/metadata/')
