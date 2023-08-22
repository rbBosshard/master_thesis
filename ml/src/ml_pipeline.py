from ml.src.ml_helper import load_config, get_assay_df, get_fingerprint_df, merge_assay_and_fingerprint_df, split_data, \
    partition_data, handle_oversampling, grid_search_cv, build_pipeline, predict_and_report, \
    get_label_counts, report_exception

from datetime import datetime


# Get ML configuration
CONFIG, LOGGER = load_config()

# Get assay data
assay_df = get_assay_df()

# Get fingerprint data
fps_df = get_fingerprint_df()

# Merge chemical ids in both dataframes
df = merge_assay_and_fingerprint_df(assay_df, fps_df)

# Partition data into X and y
X, y = partition_data(df)

# Split ML data into train, validation and test set
X_train, X_val, y_train, y_val, X_test, y_test = split_data(X, y)

# Apply oversampling if configured
X_train, y_train = handle_oversampling(X_train, y_train)

# Get the label counts
get_label_counts(y, y_train, y_val, y_test)

# Build for each classifier a pipeline according to the configurations in the config file
total_time_start = datetime.now()

for classifier in CONFIG['classifiers']:
    try:
        start_time = datetime.now()

        # Build the pipeline for the current classifier with the specified parameter grid
        pipeline = build_pipeline(classifier['steps'])

        # Perform grid search (Note: CV on TRAINING set with RepeatedStratifiedKFold)
        grid_search = grid_search_cv(X_train, y_train, classifier, pipeline)

        # Predict on the validation set with the best estimator (X_val, y_val is unseen)
        predict_and_report(X_val, y_val, classifier, grid_search.best_estimator_)

        elapsed = round((datetime.now() - start_time).total_seconds(), 2)
        LOGGER.info(f"Done {classifier['name']} >> {elapsed} seconds.\n{'_' * 75}\n\n")

    except Exception as e:
        report_exception(e, classifier)

elapsed = round((datetime.now() - total_time_start).total_seconds(), 2)
LOGGER.info(f"Finished: Total time >> {elapsed} seconds.\n")

