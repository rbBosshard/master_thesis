from ml.src.utils.compute_compound_presence_matrix import compute_compound_presence_matrix
from ml.src.utils.get_concentrations import get_concentrations
from ml.src.utils.get_ml_data import extract_pytcpl_output
from ml.src.utils.get_val_set_coverage import get_val_set_coverage, prepare_validation_set
from ml.src.utils.helper import init_directories, csv_to_parquet_converter

# init_directories()
# csv_to_parquet_converter()
compute_compound_presence_matrix(ALL=1, SUBSET=1)
# extract_pytcpl_output()
# prepare_validation_set()
# get_val_set_coverage(COLLECT_STATS=1, COMPUTE_PRESENCE_MATRIX=1, COMPUTE_HIT_CALL_MATRIX=1)
# get_concentrations()
