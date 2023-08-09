import os
import time

import plotly.express as px
import streamlit as st
import pandas as pd
import seaborn as sns
from streamlit_plotly_events import plotly_events


from helper import CSV_DIR_PATH

start_time = time.time()
print(f"Start analyzing aeid chid matrix..")

print(f"Loading input..")
suffix = ".parquet.gzip"
csv_input = os.path.join(CSV_DIR_PATH, f"small_subset_sample_matrix{suffix}")  # presence_matrix_aeid_chid, small_subset_sample_matrix
df = pd.read_parquet(csv_input) #.set_index('aeid').rename_axis("chid", axis="columns")

print(f"Plotting..")

# Print overall presence matrix
heatmap = sns.heatmap(df, vmin=0, vmax=1, cbar=False, cmap="winter")
# plt.show()

# # Sort the data
df = df.sum().sort_values(ascending=False)
fig = px.bar(x = df.index, y = df.values, title='Number of Assay Endpoints Tested') #  Assay Endpoints Coverage
fig.update_xaxes(type='category')

st.set_page_config(layout="wide")
st.title('Interactive Sorted Bar Plot')
selected_points = plotly_events(fig)
st.write(selected_points)

print(f"Writing output..")

print(f"Total time taken: {(time.time() - start_time):.2f} seconds")
