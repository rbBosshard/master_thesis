import logging
import os
import sys
from datetime import datetime
import traceback

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import make_scorer, fbeta_score, roc_curve, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.decomposition import NMF
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from sklearn.decomposition import NMF

import matplotlib

matplotlib.use('Agg')
import warnings

# Filter out ConvergenceWarnings
# warnings.filterwarnings("ignore")

from ml.src.pipeline.constants import CONFIG_PATH, LOG_DIR_PATH, CONFIG_CLASSIFIERS_PATH, \
    INPUT_FINGERPRINTS_DIR_PATH, FILE_FORMAT, REMOTE_DATA_DIR_PATH, MASSBANK_DIR_PATH, \
    CONFIG_REGRESSORS_PATH

CONFIG = {}
CONFIG_ESTIMATORS = {}
START_TIME = datetime.now()
LOGGER = logging.getLogger(__name__)
LOG_PATH = ""
AEID = 0
ESTIMATOR_NAME = ""
TARGET_RUN_FOLDER = ""


def load_config():
    global CONFIG, CONFIG_ESTIMATORS, START_TIME, LOG_PATH, TARGET_RUN_FOLDER

    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
        if config["ignore_warnings"]:
            import warnings
            warnings.filterwarnings("ignore")

    config_estimators_path = CONFIG_REGRESSORS_PATH if 'reg' in config['ml_algorithm'] else CONFIG_CLASSIFIERS_PATH
    with open(config_estimators_path, 'r') as file:
        config_estimators = yaml.safe_load(file)

    CONFIG = config
    CONFIG_ESTIMATORS = config_estimators
    START_TIME = datetime.now()
    LOGGER = init_logger(CONFIG)

    # Get model folder of old run (if needed and specified in config: only_predict: 1)
    get_load_from_model_folder(rank=config['load_from_model']['rank'])

    log_config_path = os.path.join(LOG_PATH, '.log', "config.yaml")
    with open(log_config_path, 'w') as file:
        yaml.dump(CONFIG, file)
    log_config_estimators_path = os.path.join(LOG_PATH, '.log', "config_estimators.yaml")
    with open(log_config_estimators_path, 'w') as file:
        yaml.dump(CONFIG_ESTIMATORS, file)
    
    LOGGER.info(f"Config files dumped to '{os.path.join(LOG_PATH, '.log')}'")

    return CONFIG, CONFIG_ESTIMATORS, START_TIME, LOGGER, TARGET_RUN_FOLDER


def set_aeid(aeid):
    global AEID
    AEID = aeid


def get_assay_df():
    LOGGER.info(f"Start ML pipeline for assay ID: {AEID}\n")
    assay_file_path = os.path.join(REMOTE_DATA_DIR_PATH, "output", f"{AEID}{FILE_FORMAT}")
    assay_df = pd.read_parquet(assay_file_path)
    # omit_compound_mask = assay_df['omit_flag'] == "PASS"
    # assay_df = assay_df[omit_compound_mask]
    # LOGGER.info(f"Number of compounds omitted through: ICE OMIT_FLAG filter: {len(omit_compound_mask)}")
    hitcall = 'hitcall'
    if CONFIG['apply']['cytotoxicity_corrected_hitcalls']:
        hitcall = 'hitcall_c'
    assay_df = assay_df[['dsstox_substance_id', hitcall, 'ac50']]
    LOGGER.info(f"Assay dataframe: {assay_df.shape[0]} chemical/hitcall datapoints")
    return assay_df


def get_fingerprint_df():
    fps_file_path = os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"{CONFIG['fingerprint_file']}{FILE_FORMAT}")
    fps_df = pd.read_parquet(fps_file_path)
    LOGGER.info(f"Fingerprint dataframe: {fps_df.shape[0]} chemicals, {fps_df.iloc[:, 1:].shape[1]} binary features")
    LOGGER.info("%" * 60 + "\n")
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
                                                        # stratify=y # stratify to ensure the same class distribution in the train and test sets
                                                        )

    return X_train, y_train, X_test, y_test


def partition_data(df):
    # Split off the massbank validation set
    # Load safe-to-use massbank compounds
    validation_compounds_path = os.path.join(MASSBANK_DIR_PATH, f"validation_compounds_safe{FILE_FORMAT}")
    validation_compounds = pd.read_parquet(validation_compounds_path)["dsstox_substance_id"]

    # Load the massbank validation set (unsafe + safe compounds)
    massbank_val_pred_path = os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"sirius_massbank_fingerprints{FILE_FORMAT}")
    massbank_val_pred_df = pd.read_parquet(massbank_val_pred_path).drop(columns=['449']) # Why does SIRIUS Fingerprint has an additional column '449'?

    # Identify common identifiers
    massbank_val_pred_df = massbank_val_pred_df[massbank_val_pred_df['dsstox_substance_id'].isin(validation_compounds)]
    massbank_val_true_df = df[df['dsstox_substance_id'].isin(validation_compounds)]

    common_ids = set(massbank_val_pred_df['dsstox_substance_id']).intersection(massbank_val_true_df['dsstox_substance_id'])

    # Filter rows in based on common identifiers
    massbank_val_pred_df = massbank_val_pred_df[massbank_val_pred_df['dsstox_substance_id'].isin(common_ids)]
    massbank_val_true_df = massbank_val_true_df[massbank_val_true_df['dsstox_substance_id'].isin(common_ids)]

    # Arrange to match the order in massbank_val_true_df
    massbank_val_true_df = massbank_val_true_df.sort_values(by='dsstox_substance_id')
    massbank_val_pred_df = massbank_val_pred_df.sort_values(by='dsstox_substance_id')

    validation_compounds = massbank_val_pred_df['dsstox_substance_id']

    validation_filter_condition = df['dsstox_substance_id'].isin(validation_compounds)
    training_df = df[~validation_filter_condition]  # From this in a later step, another internal validation set is split off

    # Safety check that the compounds intersection with the exteranl validation set is empty
    is_distinct = len(set(training_df['dsstox_substance_id']).intersection(massbank_val_pred_df['dsstox_substance_id'])) == 0
    assert is_distinct

    # Partition the data into features (X) and labels (y)
    # Select all columns as fingerprint features, starting from the third column (skipping dtxsid and hitcall and ac50)
    feature_names = training_df.columns[3:]

    X = training_df.iloc[:, 3:]
    X_massbank_val_from_structure = massbank_val_true_df.iloc[:, 3:]
    X_massbank_val_from_sirius = massbank_val_pred_df.iloc[:, 1:]

    if CONFIG['apply']['cytotoxicity_corrected_hitcalls']:
        hitcall = 'hitcall_c'
    else:
        hitcall = 'hitcall'

    # Distinguish between regression and binary classification
    if 'reg' in CONFIG['ml_algorithm']:
        y = training_df[hitcall]
        y_massbank_val = massbank_val_true_df[hitcall]
    else:  # binary classification
        t = CONFIG['activity_threshold']
        LOGGER.info(f"Activity threshold: ({hitcall} >= {t} is active)\n")
        y = (training_df[hitcall] >= t).astype(np.uint8)
        y_massbank_val = (massbank_val_true_df[hitcall] >= t).astype(np.uint8)

    return feature_names, X, y, X_massbank_val_from_structure, X_massbank_val_from_sirius, y_massbank_val


def assess_similarity(ground_truth, predicted):
    predicted = predicted.astype(int)
    ground_truth = ground_truth.astype(int)

    # Flatten both matrices to 1D arrays
    y_true = ground_truth.to_numpy().flatten()
    y_pred = predicted.to_numpy().flatten()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # print("Accuracy:", accuracy)
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1 Score:", f1)

    # Create a figure with two subplots (heatmap and line plot)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(50, 20)) 

    # Calculate dissimilarity

    dissimilarity_matrix = (predicted.values - ground_truth.values).astype(int)
    cmap = sns.color_palette(['blue', 'white', 'red'])  # Blue for -1, White for 0, Red for 1
    sns.heatmap(dissimilarity_matrix,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                cmap='viridis',
                ax=ax1,
                )
    ax1.set_title(f"(aeid={AEID}) Dissimilarity in Massbank Validation Set with Shape {predicted.shape}: "
                  f"Predicted - True Fingerprints. "
                  f"Purple=-1, Green=0, Yellow=1",
                  fontsize=50)
    ax1.set_xticklabels([])
    ax1.set_xlabel('Fingerprint features', fontsize=40)
    ax1.set_ylabel('Compounds', fontsize=40)
    # Display the metrics
    legend_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
    ax1.annotate(legend_text, xy=(1, 1), xytext=(0.9, 0.61), fontsize=40,
                 xycoords='axes fraction', textcoords='axes fraction',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.4))

    sum_of_columns = ground_truth.sum(axis=0)
    ax2.plot(sum_of_columns, marker='.', linestyle='-', label='True Fingerprints')
    sum_of_columns = predicted.sum(axis=0)
    ax2.plot(sum_of_columns, marker='.', linestyle='--', label='Predicted by SIRIUS')
    ax2.set_xlabel('Fingerprint Features', fontsize=40)
    ax2.set_ylabel('Sum of Present Features', fontsize=40)
    ax2.set_title('Sum of Bits in Fingerprint Features', fontsize=50)
    ax2.set_xticklabels([])
    ax2.tick_params(axis='y', labelsize=35)
    ax2.set_xticks([])
    ax2.set_xlim(0, len(sum_of_columns))
    legend = ax2.legend(prop={'size': 40})

    # Adjust the layout of the subplots
    plt.subplots_adjust(wspace=0, hspace=20)
    plt.tight_layout()

    path = os.path.join(LOG_PATH, f"{AEID}", "dissimilarity.png")
    plt.savefig(path, format='png')
    plt.close()

    predicted = predicted.astype(np.uint8)
    ground_truth = ground_truth.astype(np.uint8)


def print_binarized_label_count(y, title):
    counts = (y >= CONFIG['activity_threshold']).value_counts().values
    LOGGER.info(f"Binarized Label Count {title}: {len(y)} datapoints"
                f" with {counts[0]} inactive, {counts[1]} active "
                f"({counts[1] / sum(counts) * 100:.2f}%)")


def handle_oversampling(X, y):
    # If smote configured: Oversample the minority class in the training set
    if CONFIG['apply']['smote']:
        oversampler = SMOTE(random_state=CONFIG['random_state'])
        X, y = oversampler.fit_resample(X, y)
        print_binarized_label_count(y, "TRAIN (after oversampling)")
    return X, y


def build_preprocessing_pipeline():
    preprocessing_pipeline_steps = []
    if CONFIG['apply']['feature_selection']:
        if CONFIG['apply']['variance_threshold']:
            # VarianceThreshold is a feature selector that removes all low-variance features. -> Did not improve results significantly
            feature_selection_variance_threshold = VarianceThreshold(CONFIG['feature_selection']['variance_threshold'])
            preprocessing_pipeline_steps.append(('feature_selection_variance_threshold', feature_selection_variance_threshold))

        if CONFIG['apply']['non_negative_matrix_factorization']:
            # Non-Negative Matrix Factorization (NMF) -> Takes very long and did not improve results significantly (tested: n_components = [100, 200, 500]
            feature_selection_nmf = NMF(n_components=100, random_state=CONFIG['random_state'])  # n_components=50, 100, 200
            preprocessing_pipeline_steps.append(('feature_selection_nmf', feature_selection_nmf))

        if 'reg' in CONFIG['ml_algorithm']:
            feature_selection_model = XGBRegressor(random_state=CONFIG['random_state']) \
                if CONFIG['feature_selection']['select_from_model'] == 'XGB' \
                else RandomForestRegressor(random_state=CONFIG['random_state'])
        else:
            feature_selection_model = XGBClassifier(random_state=CONFIG['random_state']) \
                if CONFIG['feature_selection']['select_from_model'] == 'XGB' \
                else RandomForestClassifier(random_state=CONFIG['random_state'])

        feature_selection_from_model = SelectFromModel(estimator=feature_selection_model, threshold='mean')   # max_features=CONFIG['feature_selection']['max_features']
        preprocessing_pipeline_steps.append(('feature_selection_from_model', feature_selection_from_model))

    pipeline = Pipeline(preprocessing_pipeline_steps)
    LOGGER.info(f"Built Preprocessing pipeline (feature selection)")
    return pipeline


def build_pipeline(estimator):
    pipeline_steps = []
    for i, step in enumerate(estimator['steps']):
        step_name = step['name']
        step_args = step.get('args', {})  # get the hyperparameters for the step, if any
        step_args.update({'random_state': CONFIG['random_state']})
        step_instance = globals()[step_name](**step_args)  # dynmically create an instance of the step
        pipeline_steps.append((step_name, step_instance))


    pipeline = Pipeline(pipeline_steps)
    LOGGER.info(f"Built Pipeline for {ESTIMATOR_NAME}")
    return pipeline


def init_aeid(aeid):
    set_aeid(aeid)
    os.makedirs(os.path.join(LOG_PATH, f"{AEID}"), exist_ok=True)


def init_estimator(estimator):
    global ESTIMATOR_NAME
    ESTIMATOR_NAME = estimator['name']
    os.makedirs(os.path.join(LOG_PATH, f"{AEID}", ESTIMATOR_NAME), exist_ok=True)


def build_param_grid(estimator_steps):
    param_grid = {}
    for step in estimator_steps:
        step_name = step['name']
        step_args = step.get('args', {})
        param_grid.update({f'{step_name}__{key}': value for key, value in step_args.items() if isinstance(value, list)})
    return param_grid


def grid_search_cv(X_train, y_train, estimator, pipeline):
    scorer = None

    if 'reg' in CONFIG['ml_algorithm']:
        cv = KFold(n_splits=CONFIG['grid_search_cv']['n_splits'], shuffle=True, random_state=CONFIG['random_state'])
        if CONFIG['apply']['custom_scorer']:
            def custom_scorer(y_true, y_pred):
                errors = np.abs(y_true - y_pred)
                weight = 2.0  # weight for errors when the true value is closer to 1
                custom_score = np.mean(np.where(y_true >= 0.5, weight * errors, errors))
                return custom_score

            scorer = make_scorer(custom_scorer, greater_is_better=False)
    else:
        cv = RepeatedStratifiedKFold(
            n_splits=CONFIG['grid_search_cv']['n_splits'],
            n_repeats=CONFIG['grid_search_cv']['n_repeats'],
            random_state=CONFIG['random_state']
        )
        if CONFIG['apply']['custom_scorer']:
            scoring = CONFIG['grid_search_cv']['scoring']
            scorer = scoring if scoring != 'f_beta' else make_scorer(fbeta_score, beta=CONFIG['grid_search_cv']['beta'])

    grid_search_cv = GridSearchCV(
        pipeline,
        param_grid=build_param_grid(estimator['steps']),
        cv=cv,
        scoring=scorer,
        n_jobs=CONFIG["grid_search_cv"]["n_jobs"],
        verbose=CONFIG["grid_search_cv"]["verbose"],
    )

    # Use train set as input to Grid Search Cross Validation (kfold validation sets drawn internally from train set)
    grid_search_cv_fitted = grid_search_cv.fit(X_train, y_train)

    LOGGER.info(f"{estimator['name']}: GridSearchCV Results:")
    best_params = grid_search_cv_fitted.best_params_ if grid_search_cv_fitted.best_params_ else "default"
    LOGGER.info(f"Best params: {best_params} with mean cross-validated {scorer} score: {grid_search_cv_fitted.best_score_}\n")

    return grid_search_cv_fitted


def find_optimal_threshold(y, y_pred_proba, input_set, target_tpr, target_tnr, default_threshold):
    # Tune the classification threshold for the classifier, used to map probabilities to class labels
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    tnr = 1 - fpr

    # Plot the ROC curve
    df_fpr_tpr = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': thresholds})

    # Find the index of the threshold that is closest to the default threshold
    idx_default = np.argmin(np.abs(thresholds - default_threshold))

    # Fixed Threshold Evaluation (for model comparison), find the closest threshold that achieves the desired TPR, TNR
    idx_tpr = np.argmax(tpr >= target_tpr)
    fixed_threshold_tpr = thresholds[idx_tpr]

    idx_tnr = np.abs(tnr - target_tnr).argmin()
    fixed_threshold_tnr = thresholds[idx_tnr]

    # Threshold Moving
    def custom_cost_to_minimize(cost_tpr, tpr, cost_fpr, fpr):
        # The goal is to minimize this cost function by penalizing the false negatives / false positives
        # Linear combination of costs, higher tpr is better, lower fpr is better,
        return cost_tpr * (1 - tpr) + cost_fpr * fpr

    # Find the optimal threshold based on a cost function, weighting true positive rate and false positive rate
    cost_tpr = CONFIG['threshold_moving']['cost_tpr']
    cost_fpr = CONFIG['threshold_moving']['cost_fpr']
    costs = custom_cost_to_minimize(cost_tpr, tpr, cost_fpr, fpr)
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]

    # Plot the ROC curve
    plt.figure(figsize=(8, 8))
    fontsize = 12

    # Classifier
    plt.scatter(df_fpr_tpr['FPR'], df_fpr_tpr['TPR'], s=5, alpha=0.8, color='black', zorder=2)
    plt.plot(df_fpr_tpr['FPR'], df_fpr_tpr['TPR'], linestyle='-', alpha=0.8, color='black', zorder=2,
             label=f'ROC Curve: {ESTIMATOR_NAME}')

    # No-Skill classifier
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.8,
             label='ROC Curve: No-Skill Classifier')

    # Highlight thresholds point
    plt.scatter(fpr[idx_default], tpr[idx_default], alpha=0.8, color='blue', s=120, marker='o',
                label=f'"Default" Threshold: {default_threshold}')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], alpha=0.8, color='green', s=120, marker='o',
                label=f'"Optimal" Threshold: {optimal_threshold:.3f}')
    plt.scatter(fpr[idx_tpr], tpr[idx_tpr], alpha=0.8, color='orange', s=120, marker='o',
                label=f'"TPR≈{target_tpr}" Threshold: {fixed_threshold_tpr:.3f}')
    plt.scatter(fpr[idx_tnr], tpr[idx_tnr], alpha=0.8, color='red', s=120, marker='o',
                label=f'"TNR≈{target_tnr}" Threshold: {fixed_threshold_tnr:.3f}')

    # plt.axhline(target_tnr, color='orange', linestyle=':', alpha=0.8)
    # plt.axvline(target_tnr, color='red', linestyle=':', alpha=0.8)

    # Info Legend
    info_text = f'Info:\n' \
                f'        TPR = TP / (TP + FN)\n' \
                f'        FPR = FP / (FP + TN)\n' \
                f'        TNR = 1 - FPR\n' \
                f'        Optimal Cost(TPR, FPR) =\n' \
                f'        {cost_tpr} * (1 - TPR) + {cost_fpr} * FPR'
    plt.plot([1, 1], [1, 1], linestyle='', alpha=0.0, label=f'{info_text}')

    plt.legend(fontsize=fontsize, loc='lower right', framealpha=0.6)

    plt.xlim(-0.01, 1)
    plt.ylim(0, 1.01)
    plt.xlabel('False Positive Rate (FPR)', fontsize=fontsize+2)
    plt.ylabel('True Positive Rate (TPR)', fontsize=fontsize+2)
    plt.xticks(fontsize=fontsize-1)
    plt.yticks(fontsize=fontsize-1)
    plt.suptitle(f"ROC Curve & Classification Thresholds", fontsize=fontsize+4)
    plt.title(f"aeid: {AEID}, {ESTIMATOR_NAME}", fontsize=fontsize+3)

    plt.grid()
    plt.tight_layout()

    path = os.path.join(LOG_PATH, f"{AEID}", ESTIMATOR_NAME, input_set, f"roc_curve.png")
    plt.savefig(path)
    plt.close()

    path = os.path.join(LOG_PATH, f"{AEID}", ESTIMATOR_NAME, f"fixed_threshold_tpr_{input_set}.joblib")
    joblib.dump(fixed_threshold_tpr, path)

    path = os.path.join(LOG_PATH, f"{AEID}", ESTIMATOR_NAME, f"fixed_threshold_tnr_{input_set}.joblib")
    joblib.dump(fixed_threshold_tnr, path)

    path = os.path.join(LOG_PATH, f"{AEID}", ESTIMATOR_NAME, f"optimal_threshold_{input_set}.joblib")
    joblib.dump(optimal_threshold, path)

    LOGGER.info(f"Optimal and fixed threshold saved.")
    return optimal_threshold, fixed_threshold_tpr, fixed_threshold_tnr


def predict_and_report_classification(X, y, best_estimator, input_set):
    os.makedirs(os.path.join(LOG_PATH, f"{AEID}", ESTIMATOR_NAME, input_set), exist_ok=True)
    LOGGER.info(f"\n")
    LOGGER.info(f"Predict ({input_set}), {ESTIMATOR_NAME}")

    # Predict the probabilities (using validation set)
    y_pred_proba = best_estimator.predict_proba(X)[:, 1]
    default_threshold = CONFIG['threshold_moving']['default_threshold']
    LOGGER.info(f"Default threshold: {default_threshold}")
    y_pred_default_threshold = np.where(y_pred_proba >= default_threshold, 1, 0)
    data = {'Actual': y, 'Predicted': y_pred_default_threshold}
    y_preds = [y_pred_default_threshold]
    y_preds_names = ['using_default_threshold']
    y_preds_descs = [f'Classification Threshold default={default_threshold}']

    # Adjust predictions based on classification threshold
    if CONFIG['apply']['threshold_moving']:
        target_tpr = CONFIG['threshold_moving']['target_tpr']
        target_tnr = CONFIG['threshold_moving']['target_tnr']

        optimal_threshold, fixed_threshold_tpr, fixed_threshold_tnr = \
            find_optimal_threshold(y, y_pred_proba, input_set, target_tpr=target_tpr, target_tnr=target_tnr,
                                   default_threshold=default_threshold)

        LOGGER.info(f"Optimal threshold: {optimal_threshold}")
        LOGGER.info(f"Fixed threshold TPR≈{target_tpr}: {fixed_threshold_tpr}")
        LOGGER.info(f"Fixed threshold, TNR≈{target_tnr}: {fixed_threshold_tnr}")

        y_pred_optimal_threshold = np.where(y_pred_proba >= optimal_threshold, 1, 0)
        y_pred_fixed_threshold_tpr = np.where(y_pred_proba >= fixed_threshold_tpr, 1, 0)
        y_pred_fixed_threshold_tnr = np.where(y_pred_proba >= fixed_threshold_tnr, 1, 0)

        y_preds += [y_pred_optimal_threshold, y_pred_fixed_threshold_tpr, y_pred_fixed_threshold_tnr]
        y_preds_names += ['using_optimal_threshold', 'using_fixed_threshold_tpr', 'using_fixed_threshold_tnr']
        y_preds_descs += ['Classification Threshold by cost(TPR, TNR)',
                          f'Classification Threshold by TPR≈{target_tpr}',
                          f'Classification Threshold by TNR≈{target_tnr}']

        new_data = {
            'Predicted_with_optimal_threshold': y_pred_optimal_threshold,
            'Predicted_with_fixed_threshold_tpr': y_pred_fixed_threshold_tpr,
            'Predicted_with_fixed_threshold_tnr': y_pred_fixed_threshold_tnr
        }
        data.update(new_data)
        data = pd.DataFrame(data)

    folder = os.path.join(LOG_PATH, f"{AEID}", ESTIMATOR_NAME, input_set)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(folder, f"estimator_results.csv"), index=False)

    labels = [True, False]

    for i, y_pred in enumerate(y_preds):
        LOGGER.info("." * 50)
        LOGGER.info(f"Predict {y_preds_names[i]}, {ESTIMATOR_NAME}")
        name = y_preds_names[i]
        desc = y_preds_descs[i]
        report = classification_report(y, y_pred, labels=labels, output_dict=True)
        path = os.path.join(folder, f"report_{name}.csv")
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(path)

        try:
            cmap = plt.get_cmap(CONFIG['cmap'])
            cm = confusion_matrix(y, y_pred, labels=labels)
            tn, fp, fn, tp = cm.ravel()  # Extract values from confusion matrix
            LOGGER.info(f"Total: {len(y)} datapoints")
            LOGGER.info(f"Ground truth: {tn + fp} positive, {tp + fn} negative")
            LOGGER.info(f"Prediction: {tn + fn} positive, {tp + fp} negative")

            display_labels = {
                'Positive': {'fontsize': 30},
                'Negative': {'fontsize': 30},
            }

            plt.figure(figsize=(8, 8))

            # cm_df = pd.DataFrame(cm, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])
            # sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            cm_display.plot(cmap=cmap, colorbar=False)

            pos_count = cm[0, 0] + cm[0, 1]
            neg_count = cm[1, 0] + cm[1, 1]

            plt.suptitle(f"Confusion Matrix: {desc} ", fontsize=10)
            plt.title(f"aeid: {AEID}, {ESTIMATOR_NAME}, Count: {len(y)} (P:{pos_count}, N:{neg_count})", fontsize=10)
            path = os.path.join(folder, f"confusion_matrix_{name}.png")
            plt.savefig(path, format='png')
            plt.close("all")

        except Exception as e:
            traceback_info = traceback.format_exc()
            report_exception(e, traceback_info, ESTIMATOR_NAME)
            plt.close()


def predict_and_report_regression(X, y, best_estimator, input_set):
    os.makedirs(os.path.join(LOG_PATH, f"{AEID}", ESTIMATOR_NAME, input_set), exist_ok=True)
    LOGGER.info("\n")
    LOGGER.info(f"Predict ({input_set}), {ESTIMATOR_NAME}")
    y_pred = best_estimator.predict(X)

    folder = os.path.join(LOG_PATH, f"{AEID}", ESTIMATOR_NAME, input_set)

    data = {
        'Actual': y,
        'Predicted': y_pred
    }

    df = pd.DataFrame(data)

    df.to_csv(os.path.join(folder, f"reg_results.csv"), index=False)

    mse_val = mean_squared_error(y, y_pred)
    r2_val = r2_score(y, y_pred)

    data = {
        'Metric': ['mse', 'r2'],
        'Value': [mse_val, r2_val]
    }

    report_df = pd.DataFrame(data)

    # Save the report to csv
    report_df.to_csv(os.path.join(folder, f"report.csv"), index=False)

    plt.scatter(y, y_pred, alpha=0.2, s=5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Validation Set - Actual vs. Predicted Values")
    plt.savefig(os.path.join(folder, "results.png"))
    plt.close()

    heatmap, xedges, yedges = np.histogram2d(y, y_pred, bins=5, range=[[0, 1], [0, 1]])
    fig, ax = plt.subplots()
    from matplotlib.colors import LogNorm
    cax = ax.imshow(heatmap.T, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', norm=LogNorm()) # , norm=LogNorm()
    cbar = fig.colorbar(cax)
    cbar.set_label('Frequency')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Validation Set - Actual vs. Predicted Values")
    plt.savefig(os.path.join(folder, "results_heatmap.png"))
    plt.close()


def get_label_counts(y, y_train, y_test, y_massbank_val):
    print_binarized_label_count(y, "TOTAL")
    print_binarized_label_count(y_train, "TRAIN")
    print_binarized_label_count(y_test, "TEST")
    print_binarized_label_count(y_massbank_val, "MassBank VALIDATION")


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


def report_exception(exception, traceback_info, entitiy):
    error_file_path = os.path.join(LOG_PATH, '.log', f"error.error")
    with open(error_file_path, "a") as f:
        err_msg = f"{entitiy} failed: {exception}"
        LOGGER.error(err_msg)
        LOGGER.error(traceback_info)
        print(err_msg, file=f)
        print(traceback_info, file=f)


def load_model(path, pipeline):
    model = joblib.load(path)
    LOGGER.info(f"Loaded {pipeline} model from {path}")
    return model


def save_model(best_estimator, fit_set):
    estimator_log_folder = os.path.join(LOG_PATH, f"{AEID}", ESTIMATOR_NAME)
    best_estimator_path = os.path.join(estimator_log_folder, f"best_estimator_{fit_set}.joblib")
    joblib.dump(best_estimator, best_estimator_path, compress=3)
    # best_params_path = os.path.join(estimator_log_folder, f"best_params_{fit_set}.joblib")
    # joblib.dump(best_estimator.get_params(), best_params_path, compress=3)


def preprocess_all_sets(preprocessing_pipeline, feature_names, X_train, y_train, X_test, y_test,
                        X_massbank_val_from_structure, X_massbank_val_from_sirius, y_massbank_val):
    # Feature selection fitted on train set. Transform all sets with the same feature selection
    if CONFIG['apply']['only_predict']:
        folder = os.path.join(TARGET_RUN_FOLDER, f"{AEID}")
        preprocessing_model_path = os.path.join(folder, f"preprocessing_model.joblib")
        preprocessing_pipeline = load_model(preprocessing_model_path, "preprocessing")

    if preprocessing_pipeline.steps:
        X_train = preprocessing_pipeline.fit_transform(X_train, y_train)

        num_features = X_train.shape[1]
        LOGGER.info(f"Number of selected features: {num_features}")

        # Get the selected feature indices and then names
        selected_feature_indices = preprocessing_pipeline[-1].get_support()
        feature_names = [feature_names[i] for i, selected in enumerate(selected_feature_indices) if selected]

        # Transform other sets (e.g. subselect feature columns that were selected by the feature selection model)
        X_test = preprocessing_pipeline.transform(X_test)
        X_massbank_val_from_structure = preprocessing_pipeline.transform(X_massbank_val_from_structure)
        X_massbank_val_from_sirius = preprocessing_pipeline.transform(X_massbank_val_from_sirius)

        # Assert that all sets have the same features
        if num_features != X_test.shape[1] or num_features != X_massbank_val_from_structure.shape[1] or num_features != X_massbank_val_from_sirius.shape[1]:
            LOGGER.error(f"Number of features in train, test, massbank_val_from_structure, massbank_val_from_sirius do not match: "
                         f"{num_features}, {X_test.shape[1]}, {X_massbank_val_from_structure.shape[1]}, {X_massbank_val_from_sirius.shape[1]}")
            raise RuntimeError("Error in feature selection")

    # Save preprocessing_model
    preprocessing_model_log_folder = os.path.join(LOG_PATH, f"{AEID}")
    os.makedirs(preprocessing_model_log_folder, exist_ok=True)
    preprocessing_model_path = os.path.join(preprocessing_model_log_folder, f"preprocessing_model.joblib")
    joblib.dump(preprocessing_pipeline, preprocessing_model_path, compress=3)
    LOGGER.info(f"Saved preprocessing model in {preprocessing_model_log_folder}")

    return feature_names, X_train, y_train, X_test, y_test, X_massbank_val_from_structure, X_massbank_val_from_sirius, y_massbank_val


def folder_name_to_datetime(folder_name):
    return datetime.strptime(folder_name, '%Y-%m-%d_%H-%M-%S')


def get_load_from_model_folder(rank=1):
    global TARGET_RUN_FOLDER
    logs_folder = os.path.join(LOG_DIR_PATH, f"runs_{CONFIG['ml_algorithm']}")
    subfolders = [f for f in os.listdir(logs_folder)]
    sorted_subfolders = sorted(subfolders, key=folder_name_to_datetime, reverse=True)
    target_run_folder = sorted_subfolders[rank] if CONFIG['load_from_model']['use_last_run'] else CONFIG['load_from_model']['target_run']
    TARGET_RUN_FOLDER = os.path.join(logs_folder, target_run_folder)


def get_feature_importance_if_applicable(best_estimator, feature_names):
    best_estimator_model = best_estimator[-1]  # best_estimator is the pipeline and the last step is the model itself
    try:
        feature_importances = best_estimator_model.feature_importances_
        importance_df = pd.DataFrame({'feature': feature_names, 'importances': feature_importances})
        importance_df = importance_df.sort_values(by='importances', ascending=False)

        # Select the top n or less features
        top_n = min(CONFIG['feature_selection']['top_n_feature_importances_to_plot'], importance_df.shape[0])
        top_importance_df = importance_df.head(top_n)

        plt.figure(figsize=(12, 6))
        sns.barplot(x='importances', y='feature', data=top_importance_df, palette='viridis')
        plt.title(f'Feature Importance: {ESTIMATOR_NAME}')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        folder = os.path.join(LOG_PATH, f"{AEID}", ESTIMATOR_NAME)
        path = os.path.join(folder, f'sorted_feature_importances.csv')
        importance_df.to_csv(path, index=False)
        path = os.path.join(folder, f'{top_n}_feature_importances.svg')
        plt.savefig(path, format='svg', bbox_inches='tight')
        plt.close()
    except Exception as e:
        feature_importances = None
        LOGGER.error(f"Could not get feature importances for {ESTIMATOR_NAME}, {e}")
    return feature_importances
