import streamlit as st
import pandas as pd
import numpy as np
import os

from joblib import Parallel, delayed

import joblib


import plotly.express as px

from ml.src.pipeline.ml_helper import get_fingerprint_df

from ml.src.pipeline.constants import METADATA_SUBSET_DIR_PATH, FILE_FORMAT


subset_assay_info_columns = ["aeid",
                            "assay_component_endpoint_name",
                            "assay_function_type",
                            "signal_direction",
                            "MechanisticTarget",
                            "ToxicityEndpoint",
                            "biological_process_target",
                            "intended_target_family",
                            "intended_target_family_sub",
                            "intended_target_type",
                            "intended_target_type_sub",
                            "burst_assay",
                            ]

algo = "binary_classification"

if "assay_info" not in st.session_state:
    st.session_state.assay_info = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, 'assay_info.parquet.gzip'))
    st.session_state.assay_info['aeid'] = st.session_state.assay_info['aeid'].astype(str)
    st.session_state.assay_info = st.session_state.assay_info[subset_assay_info_columns]

st.set_page_config(
    page_title="Predict",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This is a demo app for the ML pipeline. It is currently under development"
    }
)


col1, col2 = st.columns(2)

with col1:
    col1.header("Predict on Environmental Sample Data")
    uploaded_file = st.file_uploader("Input Environmental Sample Data: SIRIUS' Predicted Chemical Compound Fingerprints")
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
        st.success("File loaded!")
    else:
        st.info("Choose a CSV file.")
    
    path = os.path.join(METADATA_SUBSET_DIR_PATH, f"model_paths{algo}{FILE_FORMAT}")
    model_paths = pd.read_parquet(path)[:16]

    max_workers = -1
    predictions_df = None
    if test_data is not None and st.button("Run"):
        classifiers = {}
        for index, row in model_paths.iterrows():
            classifiers[row['aeid']] = joblib.load(row['model_path'])

        def predict_for_endpoint(endpoint, clf, features):
            prediction = clf.predict(features)
            return endpoint, prediction

        compounds = test_data['dsstox_substance_id'].values
        tasks = [(endpoint, clf, test_data.iloc[:, 1:]) for endpoint, clf in classifiers.items()]
        results = Parallel(n_jobs=max_workers)(delayed(predict_for_endpoint)(*task) for task in tasks)
        predictions = {endpoint: prediction for endpoint, prediction in results}
        predictions_df = pd.DataFrame(predictions)
        predictions_df.insert(0, 'dsstox_substance_id', test_data['dsstox_substance_id'])
        
        pivot_prediction = predictions_df.melt(id_vars=['dsstox_substance_id'], var_name='aeid', value_name='prediction')
        predictions_df.set_index('dsstox_substance_id', inplace=True)
        predictions_df = predictions_df.transpose()
        st.success("Prediction done!")
        # Insert a dropdown menu to choose the compound to display the prediction for
        selected_compound = st.selectbox("Select a compound to display the prediction for", compounds)
        if selected_compound is not None:
            with st.expander("Prediction for selected compound"):
                st.dataframe(predictions_df[selected_compound], hide_index=True)


with col2:
    if predictions_df is not None:
        col2.header("Interactive Assay Grouping")

        df = st.session_state.assay_info.merge(predictions_df, left_on='aeid', right_index=True)
        if 'Select' is not df.columns[0]:
            st.session_state.assay_info.insert(0, 'Select', True)
        group_column = st.sidebar.selectbox("Select Column to Group On", df.columns[3:])

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

        # aggregate example
        avg_aeid = [(group_name, np.mean(group_data[selected_compound])) for group_name, group_data in grouped_selected.items()]

        avg_age_df = pd.DataFrame(avg_aeid, columns=['Group', 'Average Age'])

        fig = px.line_polar(avg_age_df, r='Average Age', theta='Group', line_close=True)

        # Set the layout for the polar plot
        fig.update_traces(fill='toself')
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                ),
            ),
        )

        st.title("Polar Plot")

        st.plotly_chart(fig)

        total_row_count = sum(count for _, count in counts)

        st.sidebar.write(f"Assay endpoint counts per groups: {total_row_count}")
        st.sidebar.dataframe(pd.DataFrame(counts, columns=['Group', 'Count']), hide_index=True)
        st.sidebar.dataframe(avg_aeid)

        st.sidebar.write(f"Total Number of Rows in Groups: {total_row_count}")