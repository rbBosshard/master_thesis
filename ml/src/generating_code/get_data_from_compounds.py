import os


from ml.src.ml_helper import load_config
from ml.src.constants import METADATA_DIR_PATH, REMOTE_DATA_DIR_PATH

CONFIG = load_config()[0]

MASS_BANK_DIR_PATH = os.path.join(METADATA_DIR_PATH, 'validation', 'massbank')

print(f'Reading...')
massbank_dtxsid_with_records_path = os.path.join(REMOTE_DATA_DIR_PATH, f"massbank_dtxsid_with_records.csv")
massbank_dtxsid_with_records_sirius_training_path = os.path.join(MASS_BANK_DIR_PATH, f"massbank_dtxsid_with_records_sirius_training.csv")
