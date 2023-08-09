import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(ROOT_DIR, '../config')
EXPORT_DIR_PATH = os.path.join(ROOT_DIR, '../export')
CSV_DIR_PATH = os.path.join(EXPORT_DIR_PATH, 'csv')
LOG_DIR_PATH = os.path.join(EXPORT_DIR_PATH, 'logs')
CONFIG_PATH = os.path.join(CONFIG_DIR_PATH, 'config.yaml')