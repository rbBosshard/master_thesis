import pandas as pd
import os
import json
import numpy as np

from ml.src.pipeline.constants import OUTPUT_DIR_PATH
from ml.src.utils.helper import render_svg

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import io
import kaleido
import plotly
print(plotly.__version__, kaleido.__version__)

MOST_RECENT = 0
TARGET_RUN = "2023-10-14_16-38-47_all"

st.set_page_config(
    layout="wide",
)

rename_dict = {
    'True': 'Positive',
    'False': 'Negative',
    'macro avg': 'macro avg',
    'weighted avg': 'weighted avg',
    'val': 'Internal validation',
    'mb_val_structure': 'MB validation from structure',
    'mb_val_sirius': 'MB validation SIRIUS-predicted',
    'default': 'Default = 0.5',
    'tpr': 'TPR ≈ 0.5',
    'tnr': 'TNR ≈ 0.5',
    'optimal': 'cost(TPR, TNR)',
    'XGBClassifier': 'XGBoost',
    'RandomForestClassifier': 'RandomForest',
    'LogisticRegression': 'LogisticRegr',
    'SVC': 'RBF SVM',
    'MLPClassifier': 'MLP',
    'accuracy': 'accuracy',
}


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
dummy_estimator_model = list(validation_results[selected_target_variable][selected_ml_algorithm][dummy_aeid][selected_preprocessing_model].keys())[0]
selected_validation_type = st.sidebar.selectbox('Select Validation Set', list(validation_results[selected_target_variable][selected_ml_algorithm][dummy_aeid][selected_preprocessing_model][dummy_estimator_model].keys())[::-1])
selected_class_metric = st.sidebar.selectbox('Select Class Slice', ['macro avg', 'True', 'False', 'weighted avg', 'accuracy'])
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
selected_palette = st.sidebar.selectbox('Select Color Palette', list(color_palette_mapping.keys()))
selected_palette = color_palette_mapping.get(selected_palette)

show_summary = st.sidebar.checkbox("Show Summary", value=True)
show_legend = st.sidebar.checkbox("Show Legend", value=True)
clip_axis = st.sidebar.checkbox("Clip Axis", value=True)
save_figure = st.sidebar.checkbox("Save Figure", value=True)
generate_latex = st.sidebar.checkbox("Generate LaTeX", value=True)

reports = {}
threshold_names = []
for estimator_model in validation_results[selected_target_variable][selected_ml_algorithm][dummy_aeid][selected_preprocessing_model].keys():
    reports[estimator_model] = {}
    for aeid in validation_results[selected_target_variable][selected_ml_algorithm].keys():
        reports[estimator_model][aeid] = {}
        for classification_threshold in validation_results[selected_target_variable][selected_ml_algorithm][aeid][selected_preprocessing_model][estimator_model][selected_validation_type].keys():
            threshold_names.append(classification_threshold)
            # suffix = os.path.join(selected_preprocessing_model, estimator_model, selected_validation_type)
            # report_path = os.path.join(aeid_paths[selected_target_variable][selected_ml_algorithm][aeid], suffix, f'report_{classification_threshold}.csv')
            report = validation_results[selected_target_variable][selected_ml_algorithm][aeid][selected_preprocessing_model][estimator_model][selected_validation_type][classification_threshold]
            reports[estimator_model][aeid][classification_threshold] = pd.DataFrame(report)
    
selected_threshold = st.sidebar.selectbox('Select Classification Threshold', threshold_names)


threshold = selected_threshold
# st.markdown(f"#### Precision vs. recall using the '{threshold.upper()}' classification threshold and metric: {rename_dict[selected_class_metric]}")

filtered_reports = {}
for estimator_model, aeid_dict in reports.items():
    filtered_reports[estimator_model] = {}
    for aeid, report in aeid_dict.items():
        if threshold in report:
            filtered_reports[estimator_model][aeid] = report[threshold]

# lists to store data for plotting
estimators = []
aeids = []
accuracy = []
precision = []
recall = []
f1 = []
support = []

for j, (estimator, aeid_data) in enumerate(filtered_reports.items()):
    for aeid, metrics in aeid_data.items():
        estimators.append(rename_dict[estimator] + '  ')
        aeids.append(aeid)
        metric_accuracy = next(item for item in metrics if item['class'] == 'accuracy')
        accuracy.append(list(metric_accuracy.values())[1])
        metric = next(item for item in metrics if item['class'] == selected_class_metric)
        precision.append(metric['precision'])
        recall.append(metric['recall'])
        f1.append(metric['f1-score'])
        support.append(metric['support'])

# Add one more dummy datapoint for every estimator to make the legend work (otherwise the legend shows very small markers)
for estimator in filtered_reports.keys():
    estimators.append(rename_dict[estimator] + '  ')
    aeids.append('-1')
    accuracy.append(-1)
    precision.append(-1)
    recall.append(-1)
    f1.append(-1)
    support.append(1000000000)
    
df = pd.DataFrame({'Estimator': estimators, 'aeid': aeids, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Support': support})  

trendline_args = {'trendline': trendline_type, 'trendline_color_override':'black'} if trendline_type is not None else {}
hisogram_args = {'marginal_x': marginal_type, 'marginal_y': marginal_type} if marginal_type is not None else {}
args = {}

# Define the marker size as support-encoded, size depends on the validation type
scale = 4 if selected_validation_type == 'val' else 2
# marker_size = df['Support'] / scale
if selected_class_metric == 'accuracy':
    marker_size = None
    opacity = 0.3
else:
    marker_size = df['Support'].apply(lambda x: np.sqrt(x) / scale)
    opacity = 0.8    

fig = px.scatter(df,
    x='Recall', y='Precision', color='Estimator',
    hover_data=['aeid', 'Accuracy', 'F1', 'Support'], opacity=opacity,
    color_discrete_sequence=selected_palette,
        **args, **trendline_args, **hisogram_args)

for j, trace in enumerate(fig.data):
    if 'scatter' in trace.type:
        fig.data[j].update(marker=dict(size=marker_size, symbol="circle-dot", line=dict(color='black', width=0.7)))

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


title = f"{rename_dict[selected_validation_type]}, P vs. R with {rename_dict[threshold]} threshold on {rename_dict[selected_class_metric]}"

margin = None # 0
axis_font_size = 18
fig.update_layout(width=550, height=550, title=title, title_font=dict(size=14, color='black'), xaxis_title_font=dict(size=axis_font_size, color="black"), yaxis_title_font=dict(size=axis_font_size, color="black"),  margin=dict(t=margin))
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
                                y=1.03, 
                                x=0.74, 
                                font=dict(size=13.5, color='black')))
fig.update_layout(legend=dict(bgcolor='rgba(255, 255, 255, 0.6)'))


col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig)
    # save the figure as png
    parent_folder = os.path.dirname(os.path.abspath(__file__))
    full_name = f"{selected_target_variable}_{selected_ml_algorithm}_{selected_preprocessing_model}_{selected_validation_type}_{threshold}_{selected_class_metric}"
    if save_figure:
        file = f"{full_name}.png"
        dest_path = os.path.join(parent_folder, 'generated_results', file)
        fig.write_image(dest_path, format='png', engine='kaleido', width=550, height=550, scale=2)

# summary table of average metrics, accuracy, precision, recall, f1 grouped by Estimator
if show_summary:
    with col2:
        grouped = df[['Estimator', 'Accuracy', 'Recall', 'Precision', 'F1', 'Support']].groupby(['Estimator']).mean().reset_index()
        grouped['Accuracy'] = grouped['Accuracy'].apply(lambda x: f'{x:.2f}')
        grouped['Precision'] = grouped['Precision'].apply(lambda x: f'{x:.2f}')
        grouped['Recall'] = grouped['Recall'].apply(lambda x: f'{x:.2f}')
        grouped['F1'] = grouped['F1'].apply(lambda x: f'{x:.2f}')
        grouped['Support'] = grouped['Support'].apply(lambda x: f'{x:.0f}')
        summary = grouped[['Estimator', 'Accuracy', 'Recall', 'Precision', 'F1']]
        st.dataframe(summary)
        if save_figure:
            file = f"{full_name}.tex"
            dest_path = os.path.join(parent_folder, 'generated_results', file)
            def pandas_df_to_latex(df):
                latex = df.to_latex(index=False, escape=False, column_format='l' + 'c' * len(df.columns))
                return latex
            latex_table = pandas_df_to_latex(df)

            with open(dest_path) as file:
                file.write(latex_table)
            
            if generate_latex:
                st.latex(latex_table)





      



            
    
    


