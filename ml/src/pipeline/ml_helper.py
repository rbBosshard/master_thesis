import logging
import os
import sys
from datetime import datetime
import traceback

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
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import xgboost as xgb
from xgboost import plot_importance
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
# import gaussainNB
from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import NMF

import matplotlib
matplotlib.use('Agg')
import warnings

# Filter out ConvergenceWarnings
# warnings.filterwarnings("ignore")

from ml.src.pipeline.constants import ROOT_DIR, CONFIG_PATH, LOG_DIR_PATH, CONFIG_CLASSIFIERS_PATH, METADATA_DIR_PATH, \
    INPUT_FINGERPRINTS_DIR_PATH, FINGERPRINT_FILE, FILE_FORMAT, REMOTE_DATA_DIR_PATH, MASSBANK_DIR_PATH

CONFIG = {}
CONFIG_CLASSIFIERS = {}
START_TIME = datetime.now()
LOGGER = logging.getLogger(__name__)
LOG_PATH = ""
AEID = 0
CLASSIFIER_NAME = ""


def load_config():
    global CONFIG, CONFIG_CLASSIFIERS, START_TIME, LOG_PATH

    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
        if config["ignore_warnings"]:
            import warnings
            warnings.filterwarnings("ignore")

    with open(CONFIG_CLASSIFIERS_PATH, 'r') as file:
        config_classifiers = yaml.safe_load(file)

    CONFIG = config
    CONFIG_CLASSIFIERS = config_classifiers
    START_TIME = datetime.now()
    LOGGER = init_logger(CONFIG)

    log_config_path = os.path.join(LOG_PATH, '.log', "config.yaml")
    with open(log_config_path, 'w') as file:
        yaml.dump(CONFIG, file)
    log_config_classifiers_path = os.path.join(LOG_PATH, '.log', "config_classifiers.yaml")
    with open(log_config_classifiers_path, 'w') as file:
        yaml.dump(CONFIG_CLASSIFIERS, file)
    LOGGER.info(f"Config files dumped to '{os.path.join(LOG_PATH, '.log')}'")

    return CONFIG, CONFIG_CLASSIFIERS, START_TIME, LOGGER


def set_aeid(aeid):
    global AEID
    AEID = aeid


def get_assay_df(aeid):
    set_aeid(aeid)
    LOGGER.info(f"Start ML pipeline for assay ID: {AEID}\n")
    assay_file_path = os.path.join(REMOTE_DATA_DIR_PATH, "output", f"{AEID}{FILE_FORMAT}")
    assay_df = pd.read_parquet(assay_file_path)
    # omit_compound_mask = assay_df['omit_flag'] == "PASS"
    # assay_df = assay_df[omit_compound_mask]
    # LOGGER.info(f"Number of compounds omitted through: ICE OMIT_FLAG filter: {len(omit_compound_mask)}")
    assay_df = assay_df[['dsstox_substance_id', 'hitcall_c', 'ac50']]
    LOGGER.info(f"Assay dataframe: {assay_df.shape[0]} chemical/hitcall datapoints")
    return assay_df


def get_fingerprint_df():
    fps_file_path = os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"{CONFIG['fingerprint_file']}{FILE_FORMAT}")
    fps_df = pd.read_parquet(fps_file_path)
    LOGGER.info(f"Fingerprint dataframe: {fps_df.shape[0]} chemicals, {fps_df.iloc[:, 1:].shape[1]} binary features")
    LOGGER.info("#" * 80)
    return fps_df


def merge_assay_and_fingerprint_df(assay_df, fps_df):
    # Get intersection and merge the assay and fingerprint dataframes
    df = pd.merge(assay_df, fps_df, on="dsstox_substance_id").reset_index(drop=True)
    assert df.shape[0] == df['dsstox_substance_id'].nunique()
    LOGGER.info(f"Merged aeid output and fps: {df.shape[0]} datapoints (chemical fingerprint/hitcall)")
    return df


def split_data(X, y):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=CONFIG['train_test_split_ratio'],
                                                        random_state=CONFIG['random_state'],
                                                        shuffle=True,  # shuffle the data before splitting (default)
                                                        stratify=y)  # stratify to ensure the same class distribution in the train and test sets

    return X_train, y_train, X_test, y_test


def partition_data(df):
    # Select the hitcall as the label based on the activity threshold

    # Split off the massbank validation set
    validation_compounds_path = os.path.join(MASSBANK_DIR_PATH, f"validation_compounds_safe{FILE_FORMAT}")
    validation_compounds = pd.read_parquet(validation_compounds_path)["dsstox_substance_id"]
    validation_filter_condition = df['dsstox_substance_id'].isin(validation_compounds)
    training_df, validation_df = df[~validation_filter_condition], df[validation_filter_condition]

    # Partition the data into features (X) and labels (y)
    # Select all columns as fingerprint features, starting from the third column (skipping dtxsid and hitcall and ac50)
    X = training_df.iloc[:, 3:].astype(np.uint8)
    X_massbank_val_from_structure = validation_df.iloc[:, 3:].astype(np.uint8)
    X_massbank_val_from_sirius = validation_df.iloc[:, 3:].astype(np.uint8)  # Todo: replace with predicted fingerprints

    if CONFIG['ml_algorithm'] == 'hitcall_regression':
        y = training_df['hitcall_c']
        y_massbank_val = validation_df['hitcall_c']
    elif CONFIG['ml_algorithm'] == 'ac50_regression':
        y = training_df['ac50']
        y_massbank_val = validation_df['ac50']
    else:  # CONFIG['ml_algorithm'] == 'binary_classification'
        t = CONFIG['activity_threshold']
        LOGGER.info(f"Activity threshold: (hitcall_c >= {t} is active)\n")
        y = (training_df['hitcall_c'] >= t).astype(np.uint8)
        y_massbank_val = (validation_df['hitcall_c'] >= t).astype(np.uint8)

    return X, y, X_massbank_val_from_structure, X_massbank_val_from_sirius, y_massbank_val


def print_label_count(y, title):
    counts = y.value_counts().values
    LOGGER.info(f"Label Count {title}: {len(y)} datapoints\n"
                f" with {counts[0]} inactive, {counts[1]} active "
                f"({counts[1] / sum(counts) * 100:.2f}%)\n")


def handle_oversampling(X, y):
    # If smote configured: Oversample the minority class in the training set
    if CONFIG['apply']['smote']:
        oversampler = SMOTE(random_state=CONFIG['random_state'])
        X, y = oversampler.fit_resample(X, y)
        print_label_count(y, "TRAIN (after oversampling)")
    return X, y


def build_preprocessing_pipeline():
    preprocessing_pipeline_steps = []
    if CONFIG['apply']['feature_selection']:
        if CONFIG['apply']['variance_threshold']:
            # VarianceThreshold is a feature selector that removes all low-variance features. -> Did not improve results significantly
            feature_selection_variance_threshold = VarianceThreshold(0.01)
            preprocessing_pipeline_steps.append(('feature_selection_variance_threshold', feature_selection_variance_threshold))

        if CONFIG['apply']['non_negative_matrix_factorization']:
            # Non-Negative Matrix Factorization (NMF) -> Takes very long and did not improve results significantly (tested: n_components = [100, 200, 500]
            feature_selection_nmf = NMF(n_components=100)
            preprocessing_pipeline_steps.append(('feature_selection_nmf', feature_selection_nmf))

        feature_selection_model = RandomForestClassifier(n_estimators=200, max_depth=7)  # RandomForestClassifier()  #XGBClassifier(tree_method='gpu_hist') #RandomForestClassifier()
        feature_selection_from_model = SelectFromModel(estimator=feature_selection_model, threshold='mean')   # max_features=CONFIG['feature_selection']['max_features']
        preprocessing_pipeline_steps.append(('feature_selection_from_model', feature_selection_from_model))

    return Pipeline(preprocessing_pipeline_steps)


def build_pipeline(classifier):
    global CLASSIFIER_NAME
    CLASSIFIER_NAME = classifier['name']
    os.makedirs(os.path.join(LOG_PATH, f"{AEID}", CLASSIFIER_NAME), exist_ok=True)
    pipeline_steps = []  # build_preprocessing_pipeline()
    for i, step in enumerate(classifier['steps']):
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


def grid_search_cv(X_train, y_train, X_test, y_test, classifier, pipeline):
    scorer = None
    if CONFIG['apply']['custom_scorer']:
        scoring = CONFIG['grid_search_cv']['scoring']
        scorer = scoring if scoring != 'f_beta' else make_scorer(fbeta_score, beta=CONFIG['grid_search_cv']['beta'])

    grid_search_cv = GridSearchCV(
        pipeline,
        param_grid=build_param_grid(classifier['steps']),
        cv=RepeatedStratifiedKFold(
            n_splits=CONFIG['grid_search_cv']['n_splits'],
            n_repeats=CONFIG['grid_search_cv']['n_repeats'],
            random_state=CONFIG['random_state']),
        scoring=scorer,
        n_jobs=CONFIG["grid_search_cv"]["n_jobs"],
        verbose=CONFIG["grid_search_cv"]["verbose"],
    )

    # Use train set as input to Grid Search Cross Validation (kfold validation sets drawn internally from train set)
    grid_search_cv_fitted = grid_search_cv.fit(X_train, y_train)

    LOGGER.info(f"{classifier['name']}: GridSearchCV Results:")
    best_params = grid_search_cv_fitted.best_params_ if grid_search_cv_fitted.best_params_ else "default"
    LOGGER.info(f"Best params:\n{best_params} with mean cross-validated {scorer} score: {grid_search_cv_fitted.best_score_}\n")

    return grid_search_cv_fitted


def find_optimal_threshold(X, y, best_estimator, input_set):
    # Predict the probabilities (using validation set)
    y_pred_proba = best_estimator.predict_proba(X)[:, 1]

    # Tune the decision threshold for the classifier, used to map probabilities to class labels
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

    # Plot the ROC curve
    df_fpr_tpr = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': thresholds})

    # Calculate the complement of the weight_tpr parameter
    weight_tpr = CONFIG['threshold_moving']['weight_tpr']
    weight_fpr = 1 - weight_tpr

    # Find the optimal threshold based on a cost function, weighting true positive rate and false positive rate
    costs = (1 - weight_tpr * tpr) + weight_fpr * fpr  # A higher weight_tpr value gives more weight to minimizing false negatives.
    optimal_idx = np.argmin(costs)
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

    path = os.path.join(LOG_PATH, f"{AEID}", CLASSIFIER_NAME, input_set, f"roc_curve.png")
    plt.savefig(path, dpi=300)

    if input_set == 'validation':
        path = os.path.join(LOG_PATH, f"{AEID}", CLASSIFIER_NAME, f"optimal_treshold.joblib")
        joblib.dump(optimal_threshold, path)

    LOGGER.info(f"Optimal threshold saved.")
    return optimal_threshold


def predict_and_report(X, y, classifier, best_estimator, input_set):
    os.makedirs(os.path.join(LOG_PATH, f"{AEID}", CLASSIFIER_NAME, input_set), exist_ok=True)
    LOGGER.info(f"Predict ({input_set})")

    if not CONFIG['apply']['threshold_moving']:
        # ROC curve for finding the optimal threshold, we want to minimize the false negatives
        y_pred = best_estimator.predict(X)
    else:
        # Adjust predictions based on the optimal threshold
        optimal_threshold = find_optimal_threshold(X, y, best_estimator, input_set)
        LOGGER.info(f"Optimal threshold: {optimal_threshold}")
        y_pred_proba = best_estimator.predict_proba(X)[:, 1]
        y_pred = np.where(y_pred_proba > optimal_threshold, 1, 0)

    labels = [True, False]
    LOGGER.info(f"Classification Report {classifier['name']}:")
    report = classification_report(y, y_pred, labels=labels, output_dict=True)
    path = os.path.join(LOG_PATH, f"{AEID}", CLASSIFIER_NAME, input_set, f"report.csv")
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(path)
    LOGGER.info(report)

    try:
        cm = confusion_matrix(y, y_pred, labels=labels)
        tn, fp, fn, tp = cm.ravel()  # Extract values from confusion matrix
        LOGGER.info(f"Total: {len(y)} datapoints")
        LOGGER.info(f"Ground truth: {tn + fp} positive, {tp + fn} negative")
        LOGGER.info(f"Prediction: {tn + fn} positive, {tp + fp} negative")

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive", "Negative"])
        cm_display.plot()

        plt.title(f"Confusion Matrix for {classifier['name']}")
        path = os.path.join(LOG_PATH, f"{AEID}", CLASSIFIER_NAME, input_set, f"confusion_matrix.png")
        plt.savefig(path, format='png')
        plt.close()

    except Exception as e:
        traceback_info = traceback.format_exc()
        report_exception(e, traceback_info, classifier)


def get_label_counts(y, y_train, y_test):
    print_label_count(y, "TOTAL")
    print_label_count(y_train, "TRAIN")
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


def init_logger(CONFIG):
    global LOGGER, LOG_PATH
    LOG_PATH = os.path.join(LOG_DIR_PATH, f"runs_{CONFIG['ml_algorithm']}", f"{get_timestamp(START_TIME)}")
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(os.path.join(LOG_PATH, '.log'), exist_ok=True)
    log_filename = os.path.join(LOG_PATH, '.log', "ml_pipeline.log")
    error_filename = os.path.join(LOG_PATH, '.log', "ml_pipeline.error")
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


def report_exception(exception, traceback_info, classifier):
    error_file_path = os.path.join(LOG_PATH, '.log', f"error.error")
    with open(error_file_path, "a") as f:
        err_msg = f"{classifier} failed: {exception}"
        LOGGER.error(err_msg)
        LOGGER.error(traceback_info)
        print(err_msg, file=f)
        print(traceback_info, file=f)


def load_model(path):
    classifier_log_folder = os.path.join(LOG_PATH, f"{AEID}", CLASSIFIER_NAME)
    best_estimator_path = os.path.join(classifier_log_folder, f"best_estimator.joblib")
    model = joblib.load(best_estimator_path)
    LOGGER.info(f"Loaded model from {best_estimator_path}")
    return model


def save_model(best_estimator):
    classifier_log_folder = os.path.join(LOG_PATH, f"{AEID}", CLASSIFIER_NAME)
    best_estimator_path = os.path.join(classifier_log_folder, f"best_estimator.joblib")
    best_params_path = os.path.join(classifier_log_folder, f"best_params.joblib")

    joblib.dump(best_estimator, best_estimator_path, compress=3)
    joblib.dump(best_estimator.get_params(), best_params_path, compress=3)

    LOGGER.info(f"Saved model in {classifier_log_folder}")


def preprocess_all_sets(preprocessing_pipeline, X_train, y_train, X_test, y_test, X_massbank_val_from_structure, y_massbank_val):
    if preprocessing_pipeline.steps:
        X_train = preprocessing_pipeline.fit_transform(X_train, y_train)
        X_test = preprocessing_pipeline.transform(X_test)
        X_massbank_val_from_structure = preprocessing_pipeline.transform(X_massbank_val_from_structure)
        print(f"Number of selected features: {X_train.shape[1]}")
        if X_train.shape[1] != X_test.shape[1]:
            raise RuntimeError("Error in feature selection")

        # Save preprocessing_model
        preprocessing_model_log_folder = os.path.join(LOG_PATH, f"{AEID}")
        os.makedirs(preprocessing_model_log_folder, exist_ok=True)
        preprocessing_model_path = os.path.join(preprocessing_model_log_folder, f"preprocessing_model.joblib")
        joblib.dump(preprocessing_pipeline, preprocessing_model_path, compress=3)
        LOGGER.info(f"Saved preprocessing model in {preprocessing_model_log_folder}")

    return X_train, y_train, X_test, y_test, X_massbank_val_from_structure, y_massbank_val
