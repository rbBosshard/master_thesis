import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ml.src.pipeline.constants import REMOTE_DATA_DIR_PATH, FILE_FORMAT, METADATA_SUBSET_DIR_PATH


def calculate_mean_metrics(df, group_by, metrics):
    aggregated_metrics = df.groupby(group_by)[metrics].mean().reset_index()
    aggregated_metrics = aggregated_metrics.rename(columns=metric_rename_dict )
    return aggregated_metrics


print(f'Reading...')
dest_folder_path = os.path.join(METADATA_SUBSET_DIR_PATH, 'conc')
os.makedirs(dest_folder_path, exist_ok=True)
dest_path = os.path.join(dest_folder_path, f"{0}{FILE_FORMAT}")

metrics_to_plot = ['num_points', 'num_groups', 'num_replicates', 'range_min', 'range_max']
metric_rename_dict = {
    'num_points': '# Datapoints',
    'num_groups': '# Concentration Groups',
    'num_replicates': '# Replicates (log10)',
    'range_min': 'Lowest (log10) Concentration',
    'range_max': 'Highest (log10) Concentration',
}

if not os.path.exists(dest_path):
    src_path = os.path.join(REMOTE_DATA_DIR_PATH, 'merged', 'output', f"0{FILE_FORMAT}")
    df = pd.read_parquet(src_path)
    df = df[['aeid', 'dsstox_substance_id', 'conc']]
    df['conc'] = df['conc'].apply(json.loads)

    def calculate_metrics(conc):
        num_groups = len(set(conc))
        num_replicates = len(conc) // num_groups
        num_points = len(conc)
        min_val = np.min(conc)
        max_val = np.max(conc)
        return pd.Series([num_points, num_groups, num_replicates, min_val, max_val], index=metrics_to_plot)


    df[metrics_to_plot] = df['conc'].apply(calculate_metrics)
    df = df.drop(columns=['conc'])
    df.to_parquet(dest_path, compression='gzip')

else:
    df = pd.read_parquet(dest_path)


print(f"Shape: {df.shape}")
aeid_metrics = calculate_mean_metrics(df, group_by='aeid', metrics=metrics_to_plot)
substance_metrics = calculate_mean_metrics(df, group_by='dsstox_substance_id', metrics=metrics_to_plot)

subplot_titles = [f'{metric}' for metric in metric_rename_dict.values()]
subplot_titles = [item for item in subplot_titles for _ in range(2)]

column_titles = ("Mean-aggregated metrics on the concentrations grouped by 'Compounds'",
                 "Mean-aggregated metrics on the concentrations grouped by 'Assay Endpoints'")

fig = make_subplots(rows=5, cols=2,
                         column_titles=column_titles,
                         shared_xaxes=True,
                    )

colors = px.colors.qualitative.Plotly
template = 'simple_white'  # plotly, simple_white


def plot_metrics(metrics, metric, group_by_name, row, col):
    metrics_sorted = metrics.sort_values(by=metric, ascending=False)

    if row > 2:
        y_data = np.log10(metrics_sorted[metric])
        # y_axis_label = f"{metric} (log-scale)"
    else:
        y_data = metrics_sorted[metric]
    y_axis_label = f"{metric}"

    color = colors[row % len(colors)]

    x_data = list(range(1, len(metrics_sorted) + 1))  # Convert range to list
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name=metric, marker=dict(symbol='circle-open', color=color, size=4)), row=row, col=col, )

    fig.update_xaxes(title_text=f'{group_by_name} Index', row=row, col=col)
    fig.update_yaxes(title_text=y_axis_label, row=row, col=col)

    y_min = np.min(y_data)
    y_max = np.max(y_data)
    y_max += (y_max - y_min) * 0.1

    if y_min < 0:
        fig.update_yaxes(range=[y_min, y_max], row=row, col=col)
    else:
        fig.update_yaxes(range=[0, y_max], row=row, col=col)


for i, metric in enumerate(metric_rename_dict.values()):
    plot_metrics(substance_metrics, metric, "Compound", i + 1, 1)

for i, metric in enumerate(metric_rename_dict.values()):
    plot_metrics(aeid_metrics, metric, "Assay Endpoint", i + 1, 2)

fig.update_layout(
    # title_text="Average Concentration Metrics Aggregated by 'Assay Endpoint' and 'Compound'",
    showlegend=False,
    template=template
)

st.plotly_chart(fig, use_container_width=True)



# dest_folder_path = os.path.join(METADATA_SUBSET_DIR_PATH, 'conc')
# os.makedirs(dest_folder_path)
# dest_path = os.path.join(dest_folder_path, f"{0}{FILE_FORMAT}")
#
# df.to_parquet(dest_path, compression='gzip')
#
# aeids = get_subset_aeids()['aeid']
# for aeid in aeids:
#     df_aeid = df[df['aeid'] == aeid]
#     df_aeid = df_aeid[['dsstox_substance_id', 'conc']]
#     dest_path = os.path.join(dest_folder_path, f"{aeid}{FILE_FORMAT}")
#     df_aeid.to_parquet(dest_path, compression='gzip')
