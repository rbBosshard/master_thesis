import os
import sys
import time

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events
import plotly.figure_factory as ff
from plotly import graph_objs as go

sys.path.append(r"C:\Users\bossh\Documents\GitHub")
from pytcpl.src.pipeline.pipeline_helper import load_config, init_config, init_aeid, query_db

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(parent_dir)

from ml.src.constants import METADATA_DIR_PATH, METADATA_SUBSET_DIR_PATH, REMOTE_DATA_DIR_PATH


config, _ = load_config()
init_config(config)
init_aeid(0)

from ml.src.ml_helper import load_config
from ml.src.constants import MASS_BANK_DIR_PATH, FILE_FORMAT, \
    METADATA_SUBSET_DIR_PATH

CONFIG = load_config(only_load=1)

VALIDATION_COVERAGE_PLOTS_DIR_PATH = os.path.join(MASS_BANK_DIR_PATH, 'validation_coverage_plots')
 


def validation_set_coverage():
    st.header("Massbank validation set coverage")
    start_time = time.time()
    with st.spinner(f"Loading.."):
        subset_ids_list_names = ["compounds_safe_and_unsafe_for_validation", "compounds_safe_for_validation", "compounds_unsafe_for_validation"]
        dfs_wrapped = []
        for subset_ids_list_name in subset_ids_list_names:
            path = os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, subset_ids_list_name, f"coverage_info{FILE_FORMAT}")
            df_coverage_info = pd.read_parquet(path)
            path = os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, subset_ids_list_name, f"presence_matrix{FILE_FORMAT}")
            df_presence_matrix = pd.read_parquet(path).T
            dfs_wrapped.append((subset_ids_list_name, df_coverage_info, df_presence_matrix))
        print(f"Total time taken: {(time.time() - start_time):.2f} seconds")
    
        tabs = st.tabs([df_wrapped[0] for df_wrapped in dfs_wrapped])

        colorscale= [[0, 'white'], [0.5, 'lightgrey'], [1, 'azure']]
        value_labels = {0: "Absent", 1: "Present in assay endpoint", 2: ".. and in subset"}
        tickvals = list(value_labels.keys())
        ticktext = list(value_labels.values())

        for i, (df_name, df_coverage_info, df_presence_matrix) in enumerate(dfs_wrapped):
            print(i, df_name)
            with tabs[i]:
                fig = go.Figure(data=go.Heatmap(z=df_presence_matrix, colorscale=colorscale,
                                                colorbar = dict(tickvals=tickvals, 
                                                                ticktext=ticktext),
                                                #text=df_presence_matrix,
                                                hoverinfo ='text', showscale=True))
                
                fig.update_layout(
                    title=f"Presence matrix {df_name}",
                    xaxis_title="Assay Endpoint Index",
                    yaxis_title="Compound Index"
                )
                
                st.plotly_chart(fig)

                df_coverage_info = df_coverage_info.sort_values(by=['relative_coverage'], ascending=False)
                fig = px.scatter(df_coverage_info, x=df_coverage_info.index, y=df_coverage_info['relative_coverage'], labels={'x': 'Assay endpoints', 'y': 'Relative coverage'})
                fig.update_layout(title_text=f'{df_name} relative coverage (100% = all compounds in validation subset tested in assay endpoints)')
                fig.update_xaxes(type='category')
                fig.update_traces(marker=dict(size=3))
                fig.update_traces(marker_symbol='circle-open')
                fig.update_xaxes(title_text=f'Assay endpoints (truncated, total: {df_coverage_info.shape[0]})')
                fig.update_yaxes(title_text=f'Relative coverage')
                fig.update_yaxes(range=[0, 1])
                fig.update_layout(yaxis_tickformat='.0%')
                # fig.update_xaxes(showticklabels=False)
                st.plotly_chart(fig)
                # selected_points = plotly_events(fig)
                # st.write(selected_points)

                df_coverage_info = df_coverage_info.sort_values(by=['overlap'], ascending=False)
                fig = px.scatter(df_coverage_info, x=df_coverage_info.index, y=df_coverage_info['overlap'], labels={'x': 'Assay endpoints', 'y': 'Overlap'})
                fig.update_layout(title_text=f'{df_name} overlap')
                fig.update_xaxes(type='category')
                fig.update_traces(marker=dict(size=3))
                fig.update_traces(marker_symbol='circle-open')
                fig.update_xaxes(title_text=f'Assay endpoints (truncated, total: {df_coverage_info.shape[0]})')
                fig.update_yaxes(title_text=f"Overlap (max: {int(df_coverage_info['compounds'].iloc[0])})")
                fig.update_yaxes(range=[0, df_coverage_info['overlap'].max()])
                # fig.update_layout(yaxis_tickformat='.0%')
                # fig.update_xaxes(showticklabels=False)
                st.plotly_chart(fig)
    

# Create a function for the about page
def assay_endpoint_coverage_page():
    st.header('Assay endpoint coverage by compounds')
    start_time = time.time()
    with st.spinner(f"Loading.."):
        aeid_chid_presence_matrix_path = os.path.join(METADATA_DIR_PATH, f"aeid_compound_presence_matrix{FILE_FORMAT}")
        df_all = pd.read_parquet(aeid_chid_presence_matrix_path)

        aeid_chid_presence_matrix_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeid_compound_presence_matrix{FILE_FORMAT}")
        df_subset = pd.read_parquet(aeid_chid_presence_matrix_path)
        print(f"Total time taken: {(time.time() - start_time):.2f} seconds")

        dfs_wrapped = [("All assay endpoints", df_all), ("Subset of all assay endpoints", df_subset)]
        tabs = st.tabs([df_wrapped[0] for df_wrapped in dfs_wrapped])

        print(f"Plotting..")
        for i, (df_name, df) in enumerate(dfs_wrapped):
            print(i, df_name)
            with tabs[i]:
                df = df.T
                num_compounds = df.shape[0]
                num_assay_endpoints = df.shape[1]
                df = df.sum(axis=1).sort_values(ascending=False) / num_assay_endpoints
                fig = px.scatter(df, x=df.index, y=df.values, labels={'x': 'Compounds', 'y': 'Assay endpoints coverage'})
                fig.update_layout(title_text=f'{df_name} coverage (100% = {num_assay_endpoints} assay endpoints)')
                fig.update_xaxes(type='category')
                fig.update_traces(marker=dict(size=3))
                fig.update_traces(marker_symbol='circle-open')
                fig.update_xaxes(title_text=f'Compounds (truncated, total: {num_compounds})')
                fig.update_yaxes(title_text=f'Assay endpoint coverage')
                fig.update_yaxes(range=[0, 1])
                fig.update_layout(yaxis_tickformat='.0%')       
                # fig.update_xaxes(showticklabels=False)
                st.plotly_chart(fig)
                # selected_points = plotly_events(fig)
                # st.write(selected_points)
      
# Page setup
st.set_page_config(page_title="Data inspector", layout="wide")

# Create navigation
nav_options = ["Assay endpoint coverage by compounds","Massbank validation set coverage"]
page = st.sidebar.radio("", nav_options)

# Display selected page
if page == nav_options[0]:
    assay_endpoint_coverage_page()
elif page == nav_options[1]:
    validation_set_coverage()



