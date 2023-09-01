import logging
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import make_scorer, fbeta_score, roc_curve, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from ml.src.pipeline.constants import ROOT_DIR, CONFIG_PATH, LOG_DIR_PATH, CONFIG_CLASSIFIERS_PATH, METADATA_DIR_PATH

CONFIG = {}
CONFIG_CLASSIFIERS = {}
START_TIME = datetime.now()
LOGGER = logging.getLogger(__name__)
LOG_PATH = ""
AEID = 0
CLASSIFIER_NAME = ""


def load_config(only_load=0):
    global CONFIG, CONFIG_CLASSIFIERS, START_TIME, AEID, LOG_PATH

    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
        if config["ignore_warnings"]:
            import warnings
            warnings.filterwarnings("ignore")

    if only_load:
        return config

    with open(CONFIG_CLASSIFIERS_PATH, 'r') as file:
        config_classifiers = yaml.safe_load(file)

    CONFIG = config
    CONFIG_CLASSIFIERS = config_classifiers
    START_TIME = datetime.now()
    AEID = CONFIG['aeid']
    LOGGER = init_logger()

    log_config_path = os.path.join(LOG_PATH, "config.yaml")
    with open(log_config_path, 'w') as file:
        yaml.dump(CONFIG, file)
    log_config_classifiers_path = os.path.join(LOG_PATH, "config_classifiers.yaml")
    with open(log_config_classifiers_path, 'w') as file:
        yaml.dump(CONFIG_CLASSIFIERS, file)
    LOGGER.info(f"Config files dumped to '{LOG_PATH}'\n")

    return CONFIG, CONFIG_CLASSIFIERS, START_TIME, AEID, LOGGER


def get_assay_df():
    LOGGER.info(f"Running ML pipeline for assay ID: {CONFIG['aeid']}\n")
    assay_file_path = os.path.join(ROOT_DIR, CONFIG['remote_data_dir'], f"{CONFIG['aeid']}{CONFIG['file_format']}")
    assay_df = pd.read_parquet(assay_file_path)
    assay_df = assay_df[['dsstox_substance_id', 'hitcall']]
    LOGGER.info(f"Assay dataframe: {assay_df.shape[0]} chemical/hitcall datapoints")
    return assay_df


def get_fingerprint_df():
    fps_file_path = os.path.join(METADATA_DIR_PATH, 'fps', f"{CONFIG['fingerprint_file']}{CONFIG['file_format']}")
    fps_df = pd.read_parquet(fps_file_path)
    LOGGER.info(f"Fingerprint dataframe: {fps_df.shape[0]} chemicals, {fps_df.iloc[:, 1:].shape[1]} binary features")
    return fps_df


def merge_assay_and_fingerprint_df(assay_df, fps_df):
    # Get intersection and merge the assay and fingerprint dataframes
    df = pd.merge(assay_df, fps_df, on="dsstox_substance_id").reset_index(drop=True)
    assert df.shape[0] == df['dsstox_substance_id'].nunique()
    LOGGER.info(f"\nMerged dataframe for this ML pipeline: {df.shape[0]} datapoints (chemical fingerprint/hitcall)")
    return df


def split_data(X, y):
    # Split the data into train and test sets before oversampling to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=CONFIG['train_test_split_ratio'],
                                                        random_state=CONFIG['random_state'],
                                                        shuffle=True,  # shuffle the data before splitting (default)
                                                        stratify=y)  # stratify to ensure the same class distribution in the train and test sets

    # Split the train set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=CONFIG['train_test_split_ratio'],
                                                      random_state=CONFIG['random_state'],
                                                      shuffle=True,  # shuffle the data before splitting (default)
                                                      stratify=y_train)  # stratify to ensure the same class distribution in the train and test sets

    return X_train, X_val, y_train, y_val, X_test, y_test


def partition_data(df):
    # Partition the data into features (X) and labels (y)
    # Select all columns as fingerprint features, starting from the third column (skipping dtxsid and hitc)
    X = df.iloc[:, 2:]
    # Select the hitcall as the label based on the activity threshold
    t = CONFIG['activity_threshold']
    LOGGER.info(f"Activity threshold: (hitcall >= {t} is active)\n")
    y = (df['hitcall'] >= t).astype(int)
    return X, y


def print_label_count(y, title):
    counts = y.value_counts().values
    LOGGER.info(f"Label Count {title}: {len(y)} datapoints\n"
          f" with {counts[0]} inactive, {counts[1]} active "
          f"({counts[1]/sum(counts)*100:.2f}%)\n")


def handle_oversampling(X, y):
    # If smote configured: Oversample the minority class in the training set
    if CONFIG['apply']['smote']:
        oversampler = SMOTE(random_state=CONFIG['random_state'])
        X, y = oversampler.fit_resample(X, y)
        print_label_count(y, "TRAIN (after oversampling)")
    return X, y


def build_pipeline(classifier):
    global CLASSIFIER_NAME
    CLASSIFIER_NAME = classifier['name']
    os.makedirs(os.path.join(LOG_PATH, CLASSIFIER_NAME))
    pipeline_steps = []
    for step in classifier['steps']:
        step_name = step['name']
        step_args = step.get('args', {})  # get the hyperparameters for the step, if any
        step_instance = globals()[step_name](**step_args)  # dynmically create an instance of the step
        pipeline_steps.append((step_name, step_instance))
    return Pipeline(pipeline_steps)


def build_param_grid(classifier_steps):
    param_grid = {}
    for step in classifier_steps:
        step_name = step['name']
        step_args = step.get('args', {})
        param_grid.update({f'{step_name}__{key}': value for key, value in step_args.items() if isinstance(value, list)})
    return param_grid


def grid_search_cv(X, y, classifier, pipeline):
    scoring = CONFIG['grid_search_cv']['scoring']
    # Define the scoring function using F-beta score if specified in the config file
    scorer = scoring if scoring != 'f_beta' else make_scorer(fbeta_score, beta=CONFIG['grid_search_cv']['beta'])

    grid_search = GridSearchCV(pipeline,
                               param_grid=build_param_grid(classifier['steps']),
                               # outer grid: cross-validation, repeated stratified k-fold
                               cv=RepeatedStratifiedKFold(n_splits=CONFIG['grid_search_cv']['n_splits'],
                                                          n_repeats=CONFIG['grid_search_cv']['n_repeats'],
                                                          random_state=CONFIG['random_state']),
                               scoring=scorer,
                               n_jobs=CONFIG["grid_search_cv"]["n_jobs"],
                               verbose=CONFIG["grid_search_cv"]["verbose"],
                               ).fit(X, y)

    LOGGER.info(f"{classifier['name']}: GridSearchCV Results:")
    best_params = grid_search.best_params_ if grid_search.best_params_ else "default"
    LOGGER.info(f"Best params:\n{best_params} with mean cross-validated {scorer} score: {grid_search.best_score_}\n")

    return grid_search


def find_optimal_threshold(X, y, best_estimator):
    # Predict the probabilities (using validation set)
    y_pred_proba = best_estimator.predict_proba(X)[:, 1]

    # Tune the decission threshold for the classifier, used to map probabilities  to class labels
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

    # Plot the ROC curve
    df_fpr_tpr = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': thresholds})

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))

    # Create scatter plot
    plt.scatter(df_fpr_tpr['FPR'], df_fpr_tpr['TPR'], s=10, label='ROC Points')
    plt.plot(df_fpr_tpr['FPR'], df_fpr_tpr['TPR'], linestyle='-', label='ROC Curve')

    # Highlight optimal point
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, marker='x', label='Optimal Threshold')
    plt.text(fpr[optimal_idx], tpr[optimal_idx], f'Optimal Threshold: {optimal_threshold:.3f}',
             color='red', fontsize=10, ha='right', va='bottom')

    # Add labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Add FPR and TPR formulas as text annotations
    plt.annotate('FPR = FP / (FP + TN)', xy=(0.5, 0.2), xytext=(0.5, 0.2), textcoords='axes fraction',
                 ha='center', va='bottom', fontsize=8, color='black')
    plt.annotate('TPR = TP / (TP + FN)', xy=(0.5, 0.1), xytext=(0.5, 0.1), textcoords='axes fraction',
                 ha='center', va='bottom', fontsize=8, color='black')

    plt.grid()
    plt.tight_layout()

    path = os.path.join(LOG_PATH, CLASSIFIER_NAME, f"roc_curve.png")
    plt.savefig(path, dpi=300)
    LOGGER.info(f"Optimal threshold saved as png")
    return optimal_threshold


def predict_and_report(X, y, classifier, best_estimator):
    LOGGER.info(f"Predict..")

    if not CONFIG['threshold_moving']:
        # ROC curve for finding the optimal threshold, we want to minimize the false negatives
        y_pred = best_estimator.predict(X)
    else:
        # Adjust predictions based on the optimal threshold
        optimal_threshold = find_optimal_threshold(X, y, best_estimator)
        LOGGER.info(f"Optimal threshold: {optimal_threshold}")
        y_pred_proba = best_estimator.predict_proba(X)[:, 1]
        y_pred = np.where(y_pred_proba > optimal_threshold, 1, 0)

    labels = [True, False]
    LOGGER.info(f"Classification Report {classifier['name']}:")
    LOGGER.info(classification_report(y, y_pred, labels=labels))

    cm = confusion_matrix(y, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()  # Extract values from confusion matrix
    LOGGER.info(f"Total: {len(y)} datapoints")
    LOGGER.info(f"Ground truth: {tn + fp} positive, {tp + fn} negative")
    LOGGER.info(f"Prediction: {tn + fn} positive, {tp + fp} negative")

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive", "Negative"])
    cm_display.plot()

    plt.title(f"Confusion Matrix for {classifier['name']}")
    path = os.path.join(LOG_PATH, CLASSIFIER_NAME, f"confusion_matrix.png")
    plt.savefig(path)
    plt.close()


def get_label_counts(y, y_train, y_val, y_test):
    print_label_count(y, "TOTAL")
    print_label_count(y_train, "TRAIN")
    print_label_count(y_val, "VALIDATION")
    print_label_count(y_test, "TEST")


class ElapsedTimeFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        self.start_time = START_TIME

    def format(self, record):
        delta = datetime.now() - self.start_time
        hours, remainder = divmod(delta.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        hundredths = int((delta.microseconds / 10000) % 100)
        elapsed_time_formatted = f"[{int(hours):02}:{int(minutes):02}:{int(seconds):02}:{hundredths:02}]"
        return f"{elapsed_time_formatted} {super().format(record)}"


def create_empty_log_file(filename):
    with open(filename, 'w', encoding='utf-8'):
        pass


def init_logger():
    global LOGGER, LOG_PATH
    LOG_PATH = os.path.join(LOG_DIR_PATH, f"{AEID}", f"{get_timestamp(START_TIME)}")
    os.makedirs(LOG_PATH, )
    log_filename = os.path.join(LOG_PATH, "ml_pipeline.log")
    error_filename = os.path.join(LOG_PATH, "ml_pipeline.error")
    create_empty_log_file(log_filename)
    create_empty_log_file(error_filename)

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    console_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = ElapsedTimeFormatter('%(message)s')
    LOGGER.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)

    return LOGGER


def get_timestamp(time_point):
    return time_point.strftime('%Y-%m-%d_%H-%M-%S')


def report_exception(exception, classifier):
    error_file_path = os.path.join(LOG_PATH, f"error.error")
    with open(error_file_path, "a") as f:
        err_msg = f"{classifier} failed: {exception}"
        LOGGER.error(err_msg)
        print(err_msg, file=f)


def load_model(ok):
    classifier_log_folder = os.path.join(LOG_PATH, CLASSIFIER_NAME)
    best_estimator_path = os.path.join(classifier_log_folder, f"best_estimator.joblib")
    model = joblib.load(best_estimator_path)
    LOGGER.info(f"Loaded model from {best_estimator_path}")
    return model


def save_model(grid_search):
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_

    classifier_log_folder = os.path.join(LOG_PATH, CLASSIFIER_NAME)
    best_estimator_path = os.path.join(classifier_log_folder, f"best_estimator.joblib")
    best_params_path = os.path.join(classifier_log_folder, f"best_params.joblib")

    joblib.dump(best_estimator, best_estimator_path, compress=3)
    joblib.dump(best_params,best_params_path, compress=3)

    LOGGER.info(f"Saved model in {classifier_log_folder}")
