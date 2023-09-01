import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(ROOT_DIR, '../../config')
CONFIG_PATH = os.path.join(CONFIG_DIR_PATH, 'config.yaml')
CONFIG_CLASSIFIERS_PATH = os.path.join(CONFIG_DIR_PATH, 'config_classifiers.yaml')

LOG_DIR_PATH = os.path.join(ROOT_DIR, '../../logs')
DATA_DIR_PATH = os.path.join(ROOT_DIR, '../../data')
EXPORT_DIR_PATH = os.path.join(DATA_DIR_PATH, 'export')
CSV_DIR_PATH = os.path.join(EXPORT_DIR_PATH, 'csv')
INPUT_DIR_PATH = os.path.join(DATA_DIR_PATH, 'input')
OUTPUT_DIR_PATH = os.path.join(DATA_DIR_PATH, 'output')
METADATA_DIR_PATH = os.path.join(DATA_DIR_PATH, 'metadata')
METADATA_SUBSET_DIR_PATH = os.path.join(METADATA_DIR_PATH, 'subset')
MASS_BANK_DIR_PATH = os.path.join(METADATA_DIR_PATH, 'validation', 'massbank')


REMOTE_DATA_DIR_PATH = os.path.join(ROOT_DIR, '../../../../pytcpl/data/')
REMOTE_METADATA_DIR_PATH = os.path.join(REMOTE_DATA_DIR_PATH, 'metadata/')
REMOTE_METADATA_SUBSET_DIR_PATH = os.path.join(REMOTE_METADATA_DIR_PATH, 'subset/')

FILE_FORMAT = '.parquet.gzip'
