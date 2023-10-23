import streamlit as st
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
import joblib
import json
import plotly.express as px
from ml.src.pipeline.constants import METADATA_SUBSET_DIR_PATH, OUTPUT_DIR_PATH


st.set_page_config(
    layout="wide",
)



ML_ALGORITHM = "binary_classification"
TARGET_RUN = "2023-10-18_22-36-10" #"2023-10-14_16-38-47_all_post_ml_pipeline"


def blank_to_underscore(x):
    return x.replace(' ', '_')

rename = {
    'hitcall': 'hitcall',
    'hitcall_c': 'hitcall (c)',
    'True': 'positive',
    'False': 'negative',
    'macro avg': 'macro avg',
    'weighted avg': 'weighted avg',
    'val': 'Internal validation',
    'mb_val_structure': 'MB validation from structure',
    'mb_val_sirius': 'MB validation SIRIUS-predicted',
    'Internal validation': 'Internal',
    'MB validation from structure': 'MB structure',
    'MB validation SIRIUS-predicted': 'MB SIRIUS',
    'default': 'default=0.5',
    'tpr': 'TPR≈0.5',
    'tnr': 'TNR≈0.5',
    'optimal': 'cost(TPR,TNR)',
    'XGBClassifier': 'XGBoost',
    'XGBoost': 'XGB',
    'RF': 'RF',
    'RBF SVM': 'SVM',
    'MLP': 'MLP',
    'LR': 'LR',
    'RandomForestClassifier': 'RF',
    'LogisticRegression': 'LR',
    'SVC': 'RBF SVM',
    'MLPClassifier': 'MLP',
    'accuracy': 'accuracy',
}

subset_assay_info_columns = ["MechanisticTarget",
                             "ToxicityEndpoint",
                             "biological_process_target",
                             "intended_target_family",
                             "intended_target_family_sub",
                             "intended_target_type",
                             "intended_target_type_sub",
                             "burst_assay",
                             "assay_function_type",
                             "signal_direction",
                             "aeid",
                             "assay_component_endpoint_name",
                             ]


logs_folder = os.path.join(OUTPUT_DIR_PATH)
run_folder = st.sidebar.selectbox('Select Run', [run_folder for run_folder in os.listdir(logs_folder)][::-1])
folder = os.path.join(logs_folder, run_folder)

# Load the JSON data from the output files
with open(os.path.join(folder, 'aeid_paths.json'), 'r') as fp:
    aeid_paths = json.load(fp)

with open(os.path.join(folder, 'model_paths.json'), 'r') as fp:
    model_paths = json.load(fp)

with open(os.path.join(folder, 'feature_importances_paths.json'), 'r') as fp:
    feature_importances_paths = json.load(fp)

with open(os.path.join(folder, 'validation_results.json'), 'r') as fp:
    validation_results = json.load(fp)

with open(os.path.join(folder, 'validation_results_scores.json'), 'r') as fp:
    validation_results_scores = json.load(fp)



if "predictions_df" not in st.session_state:
    st.session_state.predictions_df = None

if "compounds" not in st.session_state:
    st.session_state.compounds = None

if "selected_compound" not in st.session_state:
    st.session_state.selected_compound = None

if "assay_info" not in st.session_state:
    st.session_state.assay_info = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, 'assay_info.parquet.gzip'))
    st.session_state.assay_info['aeid'] = st.session_state.assay_info['aeid'].astype(str)
    st.session_state.assay_info = st.session_state.assay_info[subset_assay_info_columns]


col1, col2 = st.columns(2)

with col1:
    col1.header("Predict on Environmental Sample Data")
    uploaded_file = st.file_uploader(
        "Input Environmental Sample Data: SIRIUS' Predicted Chemical Compound Fingerprints")
    test_data = None
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        if test_data.shape[1] != 2363:
            st.error("Invalid file format. Please upload a CSV file with 2364 columns.")
            test_data = None
        elif test_data.columns[0] != "dsstox_substance_id":
            st.error("Invalid file format. Please upload a CSV file with the first column named 'dsstox_substance_id'.")
            test_data = None
        with st.expander("Preview"):
            st.dataframe(test_data, hide_index=True)
        st.success("File loaded successfully")
    else:
        st.info("Choose a CSV file.")

    model_paths = model_paths["hitcall"]["classification"]
    max_workers = -1
    predictions_df = None
    if test_data is not None:
        
        if st.button("Predict!"):
            with st.spinner('Wait for it...'):
                classifiers = {}
                for index, row in model_paths.items():
                    estimator = row['Feature_Selection_XGBClassifier']['XGBClassifier']
                    folder = os.path.dirname(estimator)
                    estimator_path = os.path.join(folder, 'mb_val_sirius', 'best_estimator_full_data.joblib')
                    feature_selection_model_path = folder[:-14]
                    feature_selection_model_path = os.path.join(feature_selection_model_path, 'preprocessing_model.joblib')
                    classifiers[index] = (joblib.load(feature_selection_model_path), joblib.load(estimator_path))


                def predict_for_endpoint(endpoint, clf_data, features):
                    feature_selection_model = clf_data[0]
                    features = feature_selection_model.transform(features)
                    clf = clf_data[1]
                    prediction = clf.predict(features)
                    return endpoint, prediction

                st.session_state.compounds = test_data['dsstox_substance_id'].values
                tasks = [(endpoint, clf_data, test_data.iloc[:, 1:]) for endpoint, clf_data in classifiers.items()]
                results = Parallel(n_jobs=max_workers)(delayed(predict_for_endpoint)(*task) for task in tasks)
                predictions = {endpoint: prediction for endpoint, prediction in results}
                predictions_df = pd.DataFrame(predictions)
                predictions_df.insert(0, 'dsstox_substance_id', test_data['dsstox_substance_id'])

                pivot_prediction = predictions_df.melt(id_vars=['dsstox_substance_id'], var_name='aeid',
                                                    value_name='prediction')
                predictions_df.set_index('dsstox_substance_id', inplace=True)
                predictions_df = predictions_df.transpose()
                st.session_state.predictions_df = predictions_df
                st.success("Prediction done!")

        # Insert a dropdown menu to choose the compound to display the prediction for
        if st.session_state.compounds is not None:
            st.session_state.selected_compound = st.selectbox("Select a compound to display the prediction for",
                                                              st.session_state.compounds)
            if st.session_state.selected_compound is not None:
                with st.expander("Prediction for selected compound"):
                    st.dataframe(st.session_state.predictions_df[st.session_state.selected_compound], hide_index=True)

with col2:
    if st.session_state.predictions_df is not None:
        col2.header("Interactive Assay Grouping")

        df = st.session_state.assay_info.merge(st.session_state.predictions_df, left_on='aeid', right_index=True)
        if 'Select' not in df.columns:
            df.insert(0, 'Select', True)
        group_column = st.sidebar.selectbox("Select Column to Group On", subset_assay_info_columns)

        grouped_df = df.groupby(group_column)

        selected_group_names = []
        grouped_new = {}

        for group_name, group_data in grouped_df:
            st.write(f"#### {group_name} Group:")

            checkbox_config = st.data_editor(
                group_data,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Select?",
                        help="Select assay endpoint for inclusion in predictive model",
                        default=True,
                    )
                },
                disabled=subset_assay_info_columns,
                hide_index=True,
                use_container_width=True,
            )
            df_group = pd.DataFrame(checkbox_config)
            grouped_new[group_name] = df_group

        grouped_selected = {}
        for group_name, group_data in grouped_new.items():
            grouped_selected[group_name] = group_data[group_data['Select'] == True]

        counts = [(group_name, len(group_data)) for group_name, group_data in grouped_selected.items()]
        total_count = sum(count for _, count in counts)
        st.sidebar.write(f"Total assay endpoints selected: {total_count}")

        # aggregate example
        agg_hitcall = [(group_name, np.mean(group_data[st.session_state.selected_compound])) for group_name, group_data
                       in grouped_selected.items()]

        agg_hitcall_df = pd.DataFrame(agg_hitcall, columns=['Group', 'Aggregated Hitcall'])
        agg_hitcall_df_sorted = agg_hitcall_df.sort_values(by='Aggregated Hitcall', ascending=False)

        fig = px.bar(agg_hitcall_df_sorted, x='Group', y='Aggregated Hitcall', color='Group',
                     title=f'{st.session_state.selected_compound}: Aggregated Hitcall per Group')
        fig.update_traces(showlegend=False)

        with col1:
            with st.expander("Plot"):
                st.plotly_chart(fig, use_container_width=True)
                # st.title("Counts")
                st.dataframe(pd.DataFrame(counts, columns=['Group', 'Count']), hide_index=True)
