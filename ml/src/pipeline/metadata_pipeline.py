from ml.src.utils.compute_assay_endpoint_compound_presence_matrix import compute_assay_endpoint_compound_presence_matrix
from ml.src.utils.calculate_statistics_on_concentration_response_series import calculate_statistics_on_tested_concentrations
from ml.src.utils.extract_ml_relevant_pytcpl_output import extract_ml_relevant_pytcpl_output
from ml.src.utils.compute_massbank_validation_set_coverage import compute_massbank_validation_set_coverage, prepare_validation_set
from ml.src.utils.helper import init_directories, csv_to_parquet_converter

init_directories()
csv_to_parquet_converter()
# compute_assay_endpoint_compound_presence_matrix(ALL=1, SUBSET=1)
extract_ml_relevant_pytcpl_output()
prepare_validation_set()
compute_massbank_validation_set_coverage(COLLECT_STATS=1, COMPUTE_PRESENCE_MATRIX=1, COMPUTE_HIT_CALL_MATRIX=1)
calculate_statistics_on_tested_concentrations()
