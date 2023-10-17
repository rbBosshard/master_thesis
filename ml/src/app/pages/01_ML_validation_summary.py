import pandas as pd
import os
import json
import numpy as np

from ml.src.pipeline.constants import OUTPUT_DIR_PATH
from ml.src.utils.helper import render_svg

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


MOST_RECENT = 0
TARGET_RUN = "2023-10-14_16-38-47_all_post_ml_pipeline"

st.set_page_config(
    layout="wide",
)

rename = {
    'True': 'positive',
    'False': 'negative',
    'macro avg': 'macro avg',
    'weighted avg': 'weighted avg',
    'val': 'Internal validation',
    'mb_val_structure': 'MB validation from structure',
    'mb_val_sirius': 'MB validation SIRIUS-predicted',
    'default': 'default=0.5',
    'tpr': 'TPR≈0.5',
    'tnr': 'TNR≈0.5',
    'optimal': 'cost(TPR,TNR)',
    'XGBClassifier': 'XGBoost',
    'RandomForestClassifier': 'RandomForest',
    'LogisticRegression': 'LogisticRegression',
    'SVC': 'RBF SVM',
    'MLPClassifier': 'MLP',
    'accuracy': 'accuracy',
}

reverse_rename = {v: k for k, v in rename.items()}


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

selected_target_variable = st.sidebar.selectbox('Select Target Variable', list(validation_results.keys()))
selected_ml_algorithm = st.sidebar.selectbox('Select ML Algorithm', list(validation_results[selected_target_variable].keys()))
dummy_aeid = str(list(aeid_paths[selected_target_variable][selected_ml_algorithm].keys())[0])
selected_preprocessing_model = st.sidebar.selectbox('Select Feature Selection Model', list(validation_results[selected_target_variable][selected_ml_algorithm][dummy_aeid].keys())[::-1])
selected_validation_type = st.sidebar.selectbox('Select Validation Set', ['Internal validation', 'MB validation from structure', 'MB validation SIRIUS-predicted'])
selected_threshold = st.sidebar.selectbox('Select Classification Threshold', ['default=0.5', 'cost(TPR,TNR)', 'TPR≈0.5', 'TNR≈0.5'])
class_metrics = ['macro avg', 'weighted avg', 'positive', 'negative',  'accuracy']
support_class_metrics = ['macro avg', 'positive', 'negative']
selected_class_metric = st.sidebar.selectbox('Select Class Slice', class_metrics)
selected_marker_size = st.sidebar.selectbox('Select Marker Size based on', ['Support', 'Support_Positive', 'Support_Negative', 'more balanced=larger', 'more imbalanced=larger', None])
st.sidebar.divider()
trendline_type = st.sidebar.selectbox('Select Trendline Type', [None, 'ols', 'lowess'])
marginal_type = st.sidebar.selectbox('Select Marginal Type', ['box', 'histogram', 'violin', 'rug', None])

color_palette_mapping = {
    'Light24': px.colors.qualitative.Light24,
    'Plotly': px.colors.qualitative.Plotly,
    'Bold': px.colors.qualitative.Bold,
    'D3': px.colors.qualitative.D3,
    'G10': px.colors.qualitative.G10,
    'Safe': px.colors.qualitative.Safe,
}

with st.sidebar.expander('Settings', expanded=False):
    selected_palette = st.sidebar.selectbox('Select Color Palette', list(color_palette_mapping.keys()))
    selected_palette = color_palette_mapping.get(selected_palette)
    show_summary = st.sidebar.checkbox("Show Summary", value=True)
    clip_axis = st.sidebar.checkbox("Clip Axis", value=True)
    save_figure = st.sidebar.checkbox("Save Figure", value=True)
    generate_latex = st.sidebar.checkbox("Generate LaTeX", value=False)


reports = {}
threshold_names = []
for estimator_model in validation_results[selected_target_variable][selected_ml_algorithm][dummy_aeid][selected_preprocessing_model].keys():
    reports[estimator_model] = {}
    for aeid in validation_results[selected_target_variable][selected_ml_algorithm].keys():
        reports[estimator_model][aeid] = {}
        for metric in validation_results[selected_target_variable][selected_ml_algorithm][aeid][selected_preprocessing_model][estimator_model][reverse_rename[selected_validation_type]][reverse_rename[selected_threshold]][reverse_rename[selected_class_metric]].keys():
            value = validation_results[selected_target_variable][selected_ml_algorithm][aeid][selected_preprocessing_model][estimator_model][reverse_rename[selected_validation_type]][reverse_rename[selected_threshold]][reverse_rename[selected_class_metric]][metric]
            reports[estimator_model][aeid][metric] = value
        for metric_class in support_class_metrics:
            new_metric_class = f"{metric_class}_support"
            value = validation_results[selected_target_variable][selected_ml_algorithm][aeid][selected_preprocessing_model][estimator_model][reverse_rename[selected_validation_type]][reverse_rename[selected_threshold]][reverse_rename[metric_class]]['support']
            reports[estimator_model][aeid][new_metric_class] = value
        
        accuracy = validation_results[selected_target_variable][selected_ml_algorithm][aeid][selected_preprocessing_model][estimator_model][reverse_rename[selected_validation_type]][reverse_rename[selected_threshold]][reverse_rename['accuracy']]['precision'] # for all the same, just take the first one
        reports[estimator_model][aeid]['accuracy'] = accuracy



df = pd.DataFrame(reports)
df = df.stack().reset_index()
df = df.rename(columns={'level_0': 'aeid', 'level_1': 'Estimator', 'level_2': 'Metric', 0: 'Value'})

df['Estimator'] = df['Estimator'].apply(lambda x: rename[x])
df['Support'] = df['Value'].apply(lambda x: x['macro avg_support'])
df['Support_Positive'] = df['Value'].apply(lambda x: x['positive_support'])
df['Support_Negative'] = df['Value'].apply(lambda x: x['negative_support'])
df['Imbalance'] = df['Support_Negative'] - df['Support_Positive']
max_imbalance = df['Imbalance'].max()
df['Imbalance_Score'] = df['Imbalance'] / max_imbalance
df['Balance_Score'] = 1 - df['Imbalance_Score']


df['Accuracy'] = df['Value'].apply(lambda x: x['accuracy'])
df['Precision'] = df['Value'].apply(lambda x: x['precision'])
df['Recall'] = df['Value'].apply(lambda x: x['recall'])
df['F1'] = df['Value'].apply(lambda x: x['f1-score'])  # You mentioned wanting F1 in the hover data
df['Support'] = df['Value'].apply(lambda x: x['support'])

if selected_marker_size == 'Support':
    df['Marker_Size'] = (df['Support'] ** 0.3)
elif selected_marker_size == 'Support_Positive':
    df['Marker_Size'] = (df['Support_Positive'] ** 0.3)
elif selected_marker_size == 'Support_Negative':
    df['Marker_Size'] = (df['Support_Negative'] ** 0.3)
elif selected_marker_size == 'more balanced=larger':
    max_marker_size = 20
    min_marker_size = 2
    def scaler(x):
        return min_marker_size + (max_marker_size - min_marker_size) * x

    df['Marker_Size'] = scaler(df['Balance_Score'])

    # Calculate the ratio between x and y for each data point
    imbalance = df['Support_Negative'] - df['Support_Positive']
    sorted_imbalance = sorted(imbalance)
    st.line_chart(sorted_imbalance)
    # sort df['Support'] with the same sort key 



    ratio = df['Support_Positive'] ** 0.2 / df['Support_Negative'] ** 0.2
    # Normalize the ratio values to control marker size
    min_ratio = min(ratio)
    max_ratio = max(ratio)
    normalized_ratio = (ratio - min_ratio) / (max_ratio - min_ratio)
    # sort 
    # sorted_normaliezd_ratio = sorted(normalized_ratio)
    # st.line_chart(sorted_normaliezd_ratio)
    # Define a marker size range (you can adjust this as needed)
    min_marker_size = 2
    max_marker_size = 20

    # make new figure for marker size distribution, simple line plot with px

    # Calculate marker sizes based on the normalized ratio
    marker_sizes = min_marker_size + normalized_ratio * (max_marker_size - min_marker_size)


else:
    raise ValueError(f"Unknown marker size {selected_marker_size}")

df = df.drop(columns=['Value'])

        
# Add one more dummy datapoint for every estimator to make the legend work (otherwise the legend shows very small markers)
dummy_data = []
for estimator in df['Estimator'].unique():
    dummy_row = {
        'aeid': '-1',
        'Estimator': estimator,
        'Precision': -1.0,
        'Recall': -1.0,
        'F1': -1.0,
        'Accuracy': -1.0,
        'Support': -1,
        'Support_Positive': -1,
        'Support_Negative': -1,
        'Imbalance_Score': -1,
        'Balance_Score': -1,
        'Marker_Size': 2000,
    }
    dummy_data.append(dummy_row)
dummy_df = pd.DataFrame(dummy_data)
df = pd.concat([df, dummy_df], ignore_index=True)


trendline_args = {'trendline': trendline_type, 'trendline_color_override':'black'} if trendline_type is not None else {}
hisogram_args = {'marginal_x': marginal_type, 'marginal_y': marginal_type} if marginal_type is not None else {}
args = {'opacity': 0.8,
        'hover_data': ['aeid', 'F1', 'Accuracy', 'Support', 'Support_Positive', 'Support_Negative', 'Imbalance_Score', 'Balance_Score', 'Marker_Size'],
        'color_discrete_sequence': selected_palette,
        'custom_data': ['Marker_Size'], # Add customdata to match marker size
        } 

fig = px.scatter(df, x='Recall', y='Precision', color='Estimator', **args, **trendline_args, **hisogram_args)

for j, trace in enumerate(fig.data):
    if 'scatter' in trace.type:
        # Get the marker size from the customdata
        marker_size = list(fig.data[j].customdata[:, 0])
        fig.data[j].update(marker=dict(size=marker_size, symbol='circle-dot',  line=dict(color='black', width=0.7)))

# drop the dummy values again
df = df[df['aeid'] != '-1']
offset = 0.01
min_x = np.min(df['Recall']) - offset
max_x = np.max(df['Recall']) + offset
min_y = np.min(df['Precision']) - offset
max_y = np.max(df['Precision']) + offset

# fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1))
# fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1)) 

if clip_axis:
    fig.update_layout(xaxis=dict(range=[min_x, max_x]))
    fig.update_layout(yaxis=dict(range=[min_y, max_y]))
else:
    fig.update_layout(xaxis=dict(range=[0.0, 1.0]))
    fig.update_layout(yaxis=dict(range=[0.0, 1.0]))


title = f"{selected_validation_type}, P vs. R with {selected_threshold} threshold on {selected_class_metric}"

margin = None # 0
axis_font_size = 18
fig.update_layout(width=600, height=600, title=title, title_font=dict(size=14, color='black'), xaxis_title_font=dict(size=axis_font_size, color="black"), yaxis_title_font=dict(size=axis_font_size, color="black"),  margin=dict(t=margin))
fig.update_layout(yaxis=dict(showgrid=True, zeroline=True, gridcolor='lightgray'), xaxis=dict(showgrid=True, zeroline=True, gridcolor='lightgray'))
# axis tick font size   
fig.update_xaxes(tickfont=dict(size=axis_font_size-2, color="black"))
fig.update_yaxes(tickfont=dict(size=axis_font_size-2, color="black"))

# uniform grid 
# fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1))
# fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1)) 
# fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
# fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
# fig.update_xaxes(showspikes=True, nticks=10, tickfont=dict(size=20, color="black"))  
# fig.update_yaxes(showspikes=True, nticks=10, tickfont=dict(size=20, color="black")) 


fig.update_layout(legend_title_text='')
fig.update_layout(legend_traceorder="reversed")
fig.update_layout(legend=dict(orientation='v', yanchor='top',  xanchor='left',
                                y=1.01, 
                                x=0.72, 
                                font=dict(size=13.5, color='black')))
fig.update_layout(legend=dict(bgcolor='rgba(0, 0, 0, 0.05)'))


# fig.update_layout(legend_title_text='')
# fig.update_layout(legend_traceorder="reversed")
# fig.update_layout(legend=dict(orientation='v', yanchor='top', xanchor='left',
#                              x=0.72, y=1.01, font=dict(size=13.5, color='black'), itemmode='expand'))
# fig.update_layout(legend_title=dict(text='', font=dict(size=15, color='blue')))
# fig.update_layout(legend=dict(bordercolor='black', borderwidth=2, bgcolor='rgba(255, 255, 255, 0.8)'))



col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig)
    # save the figure as png
    parent_folder = os.path.dirname(os.path.abspath(__file__))
    full_name = f"{selected_target_variable}_{selected_ml_algorithm}_{selected_preprocessing_model}_{selected_validation_type}_{selected_threshold}_{selected_class_metric}"
    if save_figure:
        file = f"{full_name}.png"
        dest_path = os.path.join(parent_folder, 'generated_results', file)
        fig.write_image(dest_path, format='png', engine='kaleido', width=550, height=550, scale=2)

# summary table of average metrics, precision, recall, f1 grouped by Estimator
if show_summary:
    with col1:
        grouped = df[['Estimator', 'Precision', 'Recall', 'F1', 'Support']].groupby(['Estimator']).mean().reset_index()
        grouped['Precision'] = grouped['Precision'].apply(lambda x: f'{x:.2f}')
        grouped['Recall'] = grouped['Recall'].apply(lambda x: f'{x:.2f}')
        grouped['F1'] = grouped['F1'].apply(lambda x: f'{x:.2f}')
        grouped['Support'] = grouped['Support'].apply(lambda x: f'{x:.0f}')
        summary = grouped[['Estimator', 'Recall', 'Precision', 'F1']]
        st.dataframe(summary)
        
        if save_figure:
            file = f"{full_name}.tex"
            dest_path = os.path.join(parent_folder, 'generated_results', file)
            
            def pandas_df_to_latex(data):
                latex = data.to_latex(index=False, escape=False, column_format='l' + 'c' * len(df.columns))
                return latex
            
            latex_table = pandas_df_to_latex(summary)

            with open(dest_path, 'w') as file:
                file.write(latex_table)
            
            if generate_latex:
                with st.expander('LaTeX table string', expanded=True):
                    st.latex(latex_table)





      



            
    
    


