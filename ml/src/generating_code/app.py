import os
import sys
import time

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from ml.src.ml_helper import load_config
from ml.src.constants import CSV_DIR_PATH

CONFIG, CONFIG_CLASSIFIERS, START_TIME, AEID, LOGGER = load_config()

start_time = time.time()
print(f"Start analyzing aeid chid matrix..")

print(f"Loading input..")
csv_input = os.path.join(CSV_DIR_PATH,
                         f"small_subset_sample_matrix{CONFIG['file_format']}")  # presence_matrix_aeid_chid, small_subset_sample_matrix
df = pd.read_parquet(csv_input)  # .set_index('aeid').rename_axis("chid", axis="columns")

print(f"Plotting..")
df = df.sum().sort_values(ascending=False)
fig = px.bar(x=df.index, y=df.values, title='Number of Assay Endpoints Tested')  # Assay Endpoints Coverage
fig.update_xaxes(type='category')

st.set_page_config(layout="wide")
st.title('Interactive Sorted Bar Plot')
selected_points = plotly_events(fig)
st.write(selected_points)

print(f"Writing output..")

print(f"Total time taken: {(time.time() - start_time):.2f} seconds")
