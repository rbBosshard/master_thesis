import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(parent_dir)

from ml.src.pipeline.ml_helper import load_config, get_assay_df, get_fingerprint_df, merge_assay_and_fingerprint_df, split_data, \
    partition_data, handle_oversampling, grid_search_cv, build_pipeline, predict_and_report, \
    get_label_counts, report_exception, save_model, load_model

from datetime import datetime
import traceback
from ml.src.utils.helper import get_subset_aeids


# Get ML configuration
CONFIG, CONFIG_CLASSIFIERS, START_TIME, LOGGER = load_config()

# Get fingerprint data
fps_df = get_fingerprint_df()

# Get assay endpoint ids from subset considered for ML
aeids = get_subset_aeids()['aeid']

for aeid in aeids[392:]:

    # Get assay data
    assay_df = get_assay_df(aeid)

    # Merge chemical ids in both dataframes
    df = merge_assay_and_fingerprint_df(assay_df, fps_df)

    # Partition data into X and y and respective massbank validation
    X, y, X_massbank_val_from_structure, X_massbank_val_from_sirius, y_massbank_val = partition_data(df)

    # Split ML data into train, validation and test set
    X_train, X_val, y_train, y_val, X_test, y_test = split_data(X, y)

    # Apply oversampling if configured
    X_train, y_train = handle_oversampling(X_train, y_train)

    # Get the label counts
    get_label_counts(y, y_train, y_val, y_test)

    # Build for each classifier a pipeline according to the configurations in the config file
    for classifier in CONFIG_CLASSIFIERS['classifiers']:
        try:
            start_time = datetime.now()

            # Build the pipeline for the current classifier with the specified parameter grid
            pipeline = build_pipeline(classifier)

            # Perform grid search (Note: CV on TRAINING set with RepeatedStratifiedKFold)
            grid_search = grid_search_cv(X_train, y_train, classifier, pipeline)

            # Save best estimator (estimator with best performing parameters from grid search)
            best_estimator = grid_search.best_estimator_
            save_model(grid_search)

            # Load model from path
            # best_estimator = load_model("best_estimator")

            # Predict on the validation set with the best estimator (X_val, y_val is unseen)
            predict_and_report(X_val, y_val, classifier, best_estimator, "validation")

            # Predict on the Massbank validation set with the best estimator (X_val, y_val is unseen)
            predict_and_report(X_massbank_val_from_structure, y_massbank_val, classifier, best_estimator, "massbank_validation_from_structure")
            # predict_and_report(X_massbank_val_from_sirius, y_massbank_val, classifier, best_estimator, "massbank_validation_from_sirius")

            elapsed = round((datetime.now() - start_time).total_seconds(), 2)
            LOGGER.info(f"Done {classifier['name']} >> {elapsed} seconds.\n{'_' * 75}\n\n")

        except Exception as e:
            traceback_info = traceback.format_exc()
            report_exception(e, traceback_info, classifier)

    # Calculate the elapsed time
    elapsed_seconds = round((datetime.now() - START_TIME).total_seconds(), 2)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    elapsed_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    LOGGER.info(f"Finished: Total time >> {elapsed_formatted}\n")

