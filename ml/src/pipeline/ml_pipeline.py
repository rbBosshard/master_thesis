import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(parent_dir)

from ml.src.pipeline.ml_helper import load_config, get_assay_df, get_fingerprint_df, merge_assay_and_fingerprint_df, \
    split_data, \
    partition_data, handle_oversampling, grid_search_cv, build_pipeline, predict_and_report, \
    get_label_counts, report_exception, save_model, build_preprocessing_pipeline, preprocess_all_sets, \
    predict_and_report_regression

from datetime import datetime
import traceback
from ml.src.utils.helper import get_subset_aeids

import numpy as np

if __name__ == '__main__':
    # Get ML configuration
    CONFIG, CONFIG_CLASSIFIERS, START_TIME, LOGGER = load_config()

    # Get fingerprint data
    fps_df = get_fingerprint_df()

    # Get assay endpoint ids from subset considered for ML
    aeids_target_assays = get_subset_aeids()['aeid']
    # Iterate through aeids_target_assays and launch each iteration in a separate process
    for aeid in [2363]: #aeids_target_assays[:]:  # [97]: #
        # Get assay data
        assay_df = get_assay_df(aeid)

        # Merge chemical ids in both dataframes
        df = merge_assay_and_fingerprint_df(assay_df, fps_df)

        # Partition data into X and y and respective massbank validation
        X, y, X_massbank_val_from_structure, X_massbank_val_from_sirius, y_massbank_val = partition_data(df)

        # Split ML data into train, validation and test set
        X_train, y_train, X_test, y_test = split_data(X, y)

        preprocessing_pipeline_steps = build_preprocessing_pipeline()

        X_train, y_train, X_test, y_test, X_massbank_val_from_structure, y_massbank_val = preprocess_all_sets(preprocessing_pipeline_steps, X_train, y_train, X_test, y_test, X_massbank_val_from_structure, y_massbank_val)

        # Apply oversampling if configured
        X_train, y_train = handle_oversampling(X_train, y_train)

        # Get the label counts
        get_label_counts(y, y_train, y_test)

        # Build for each classifier a pipeline according to the configurations in the config file
        for classifier in CONFIG_CLASSIFIERS['classifiers']:
            try:
                start_time = datetime.now()

                # Build the pipeline for the current classifier with the specified parameter grid
                pipeline = build_pipeline(classifier)

                # Perform grid search (Note: CV on TRAINING set with RepeatedStratifiedKFold)
                grid_search = grid_search_cv(X_train, y_train, X_test, y_test, classifier, pipeline)

                # Save best estimator (estimator with best performing parameters from grid search)
                best_estimator = grid_search.best_estimator_

                # Predict on the test set with the best estimator (X_test, y_test is unseen)
                predict_and_report_regression(X_test, y_test, best_estimator, "validation")
                exit()
                predict_and_report(X_test, y_test, classifier, best_estimator, "validation")

                # Concatenate training and test data
                X_combined = np.vstack((X_train, X_test))
                y_combined = np.concatenate((y_train, y_test))

                # Retrain the best estimator from the grid search with train+test data for Massbank validation (unseen)
                best_estimator.fit(X_combined, y_combined)

                # Predict on the true Massbank validation set with the best estimator
                predict_and_report(X_massbank_val_from_structure, y_massbank_val, classifier, best_estimator,
                                   "massbank_validation_from_structure")

                # Predict on the SIRIUS predicted Massbank validation set with the best estimator
                # predict_and_report(X_massbank_val_from_sirius, y_massbank_val, classifier, best_estimator, "massbank_validation_from_sirius")

                # Concatenate combined data (train + test) and true Massbank validation data
                X_all = np.vstack((X_combined, X_massbank_val_from_structure))
                y_all = np.concatenate((y_combined, y_massbank_val))

                # Retrain the best estimator from the grid search with all data available for later inference
                best_estimator.fit(X_all, y_all)

                # Save the best estimator model, trained on all data available
                save_model(best_estimator)

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



