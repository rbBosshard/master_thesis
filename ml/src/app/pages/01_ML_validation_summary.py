import pandas as pd
import os
import json
import numpy as np
from ml.src.pipeline.constants import OUTPUT_DIR_PATH
import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go  # Import Plotly graph objects
from ml.src.utils.helper import render_svg



MOST_RECENT = 0
TARGET_RUN = "2023-10-14_16-38-47_all_post_ml_pipeline"

st.set_page_config(
    layout="wide",
)

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
    'default': 'default=0.5',
    'tpr': 'TPR≈0.5',
    'tnr': 'TNR≈0.5',
    'optimal': 'cost(TPR,TNR)',
    'XGBClassifier': 'XGBoost',
    'RandomForestClassifier': 'RF',
    'LogisticRegression': 'LR',
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
st.sidebar.divider()
trendline_type = st.sidebar.selectbox('Select Trendline Type', [None, 'ols', 'lowess'])
marginal_type = st.sidebar.selectbox('Select Marginal Type', ['box', 'histogram', 'violin', 'rug', None])
selected_marker_size = st.sidebar.selectbox('Select Marker Size based on', ['Higher Imbalance = LARGER', 'Total Support', 'Support Positive', 'Support Negative', None])

color_palette_mapping = {
    'Light24': px.colors.qualitative.Light24,
    'Plotly': px.colors.qualitative.Plotly,
    'Safe': px.colors.qualitative.Safe,
    'Bold': px.colors.qualitative.Bold,
    'D3': px.colors.qualitative.D3,
    'G10': px.colors.qualitative.G10,
}

with st.sidebar.expander('Settings', expanded=False):
    marker_symbol = st.sidebar.select_slider('Select marker symbol', 
                                              options=['circle-dot','circle-open-dot',
                                                       'x-thin', 'x', 
                                                       'circle-x', 'circle-x-open',
                                                        'cross-thin', 'asterisk'])
    opacity = st.sidebar.slider('Opacity', min_value=0.0, max_value=1.0, value=1.0, step=0.1)
    marker_line_width = st.sidebar.slider('Marker Line Width', min_value=0.3, max_value=2.0, value=0.4, step=0.1)
    marker_line_color = st.sidebar.checkbox('Dark Marker Line', value='#000000')

    selected_palette = st.sidebar.selectbox('Select Color Palette', list(color_palette_mapping.keys()))
    selected_palette = color_palette_mapping.get(selected_palette)
    show_summary = st.sidebar.checkbox("Show Summary", value=True)
    clip_axis = st.sidebar.checkbox("Clip Axis", value=True)
    autoscale = st.sidebar.checkbox("Autoscale", value=True)
    save_figure = st.sidebar.checkbox("Save Figure", value=True)
    generate_latex = st.sidebar.checkbox("Generate LaTeX", value=False)
    hover_event = st.sidebar.checkbox("Enable Hover Event", value=False)


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

df['Precision'] = df['Value'].apply(lambda x: x['precision'])
df['Recall'] = df['Value'].apply(lambda x: x['recall'])
df['F1'] = df['Value'].apply(lambda x: x['f1-score'])  
df['Accuracy'] = df['Value'].apply(lambda x: x['accuracy'])

df['Total Support'] = df['Value'].apply(lambda x: x['macro avg_support'])
df['Support Positive'] = df['Value'].apply(lambda x: x['positive_support'])
df['Support Negative'] = df['Value'].apply(lambda x: x['negative_support'])

# df['True Positives'] = df['Support Positive'] * df['Recall']
# df['True Negatives'] = df['Support Negative'] * (1 - df['Recall'])
# df['False Positives'] = df['Support Negative'] - df['True Negatives']
# df['False Negatives'] = df['Support Positive'] - df['True Positives']

df['Imbalance'] = df['Support Negative'] - df['Support Positive']
df['Imbalance Score'] = (df['Support Positive'] - df['Support Negative']) / df['Total Support']
df['Imbalance Score'] = df['Imbalance Score'] / df['Imbalance Score'].abs().max()
df['Imbalance Score'] = df['Imbalance Score'].round(3)

df = df.drop(columns=['Value'])

if selected_marker_size == 'Total Support':
    df['Marker Size'] = (df['Total Support'] ** 0.42)
elif selected_marker_size == 'Support Positive':
    df['Marker Size'] = (df['Support Positive'] ** 0.55)
elif selected_marker_size == 'Support Negative':
    df['Marker Size'] = (df['Support Negative'] ** 0.45)
elif selected_marker_size == 'Higher Imbalance = LARGER':
    df['Marker Size'] = np.abs(df['Imbalance Score']) * 15
    # sort df['Support'] with the same sort key

    ratio = df['Support Positive'] ** 0.2 / df['Support Negative'] ** 0.2
    # Normalize the ratio values to control marker size
    min_ratio = min(ratio)
    max_ratio = max(ratio)
    normalized_ratio = (ratio - min_ratio) / (max_ratio - min_ratio)
    # sort 
    # sorted_normaliezd_ratio = sorted(normalized_ratio)
    # st.line_chart(sorted_normaliezd_ratio)
    # Define a marker size range (you can adjust this as needed)
    min_marker_size = 4
    max_marker_size = 15

    # make new figure for marker size distribution, simple line plot with px

    # Calculate marker sizes based on the normalized ratio
    marker_sizes = min_marker_size + normalized_ratio * (max_marker_size - min_marker_size)
else:
    df['Marker Size'] = 10

df['Marker Size'] = df['Marker Size'].round(3)

trendline_args = {'trendline': trendline_type} if trendline_type is not None else {}
hisogram_args = {'marginal_x': marginal_type, 'marginal_y': marginal_type} if marginal_type is not None else {}

args = {'opacity': opacity,
        'hover_data': ['aeid', 'F1', 'Accuracy',
                    #    'True Positives', 'True Negatives', 'False Positives', 'False Negatives',
                       'Total Support', 'Support Positive', 'Support Negative', 
                       'Imbalance', 'Imbalance Score',
                       'Marker Size'],
        'color_discrete_sequence': selected_palette,
        'custom_data': ['Marker Size'], # Add customdata to match marker size
        } 

# Scatter plot
fig = px.scatter(df, x='Recall', y='Precision', color='Estimator', **args, **trendline_args, **hisogram_args)


# Update the marker size based on the customdata
if trendline_type is None:
    for j, trace in enumerate(fig.data):
        if 'scatter' in trace.type:
            # Get the marker size from the customdata
            marker_size = list(fig.data[j].customdata[:, 0])
            marker_line_color_ = 'black' if marker_line_color else None
            marker_line_width_ = marker_line_width if marker_line_color else None
            fig.data[j].update(marker=dict(
                symbol=marker_symbol, #'circle-open-dot', 
                size=marker_size, 
                line=dict(color=marker_line_color_, width=marker_line_width_)
                )
            )

# drop the dummy values again
# df = df[df['aeid'] != '-1']
offset = 0.01
min_x = np.min(df['Recall']) - offset
max_x = np.max(df['Recall']) + offset
min_y = np.min(df['Precision']) - offset
max_y = np.max(df['Precision']) + offset

if autoscale:
    fig.update_layout(autosize=True)
else:
    if clip_axis:
        fig.update_layout(xaxis=dict(range=[min_x, max_x]))
        fig.update_layout(yaxis=dict(range=[min_y, max_y]))
    else:
        fig.update_layout(xaxis=dict(range=[0.0, 1.0]))
        fig.update_layout(yaxis=dict(range=[0.0, 1.0]))



title = f"y={rename[selected_target_variable]} | FS={rename[selected_preprocessing_model.split('_')[2]]} | {selected_validation_type} | {selected_threshold} | {selected_class_metric}"
margin = None # 0
axis_font_size = 24
fig.update_layout(width=1000, height=1000, title=title, title_font=dict(size=21, color='black'), xaxis_title_font=dict(size=axis_font_size, color="black"), yaxis_title_font=dict(size=axis_font_size, color="black"),  margin=dict(t=margin))
fig.update_layout(yaxis=dict(showgrid=True, zeroline=True, gridcolor='lightgray'), xaxis=dict(showgrid=True, zeroline=True, gridcolor='lightgray'))
fig.update_xaxes(tickfont=dict(size=axis_font_size-2, color="black"))
fig.update_yaxes(tickfont=dict(size=axis_font_size-2, color="black"))


# legend
fig.update_layout(legend_title_text='', legend_traceorder="reversed")
fig.update_layout(legend=dict(orientation='v', yanchor='top',  xanchor='left',
                                y=1.01, 
                                x=0.743, 
                                bgcolor='rgba(255, 255, 255, 0.6)',
                                font=dict(size=31, color='black')))


# st.plotly_chart(fig)

selected_points = plotly_events(fig, click_event=True, hover_event=hover_event, key="plotly_event") # click_event=True, hover_event=True
if selected_points:
    info = selected_points[0]
    curve_number = info['curveNumber']
    point_index = info['pointIndex']
    estimator_clicked = fig.data[curve_number].legendgroup
    aeid_clicked = fig.data[curve_number].customdata[point_index, 1]
    
    # Load custom data
    aeid_path = aeid_paths[selected_target_variable][selected_ml_algorithm][aeid_clicked]
    cm_path = os.path.join(aeid_path, selected_preprocessing_model, reverse_rename[estimator_clicked], reverse_rename[selected_validation_type], f'cm_{reverse_rename[selected_threshold]}.svg')
    
    
    # Load feature importances:
    if estimator_clicked in ['XGBoost', 'RF']:
        feature_importances_path = feature_importances_paths[selected_target_variable][selected_ml_algorithm][aeid_clicked][selected_preprocessing_model][reverse_rename[estimator_clicked]]
        feature_importances = pd.read_csv(feature_importances_path)
        feature_importances = feature_importances.sort_values(by=['feature'], ascending=True).reset_index(drop=True)
        feature_importances = feature_importances.rename(columns={'feature': 'Abs. Feature Index', 'importances': 'Feature Importance'}).reset_index().rename(columns={'index': 'Rel. Feature Index'})
        fig_feature_importances = px.bar(feature_importances, x='Rel. Feature Index', y='Feature Importance', hover_data=['Abs. Feature Index'])

        fig_feature_importances.update_layout(title=f"Feature Importances on selected_preprocessing_model (independent of validation set, threshold and metric)", title_font=dict(size=14, color='black'))

        # More fancy plot of the same:
        fig_feature_importances2 = px.bar_polar(feature_importances, r='Feature Importance', theta='Rel. Feature Index', color='Feature Importance', hover_data=['Abs. Feature Index'])
        
    
    with st.expander(f'Infos for {estimator_clicked} with {aeid_clicked}', expanded=True):
        st.divider()
        render_svg(open(cm_path).read())
        st.divider()
        if estimator_clicked in ['XGBoost', 'RF']:
            st.plotly_chart(fig_feature_importances)
            st.divider()
            st.plotly_chart(fig_feature_importances2)
        else:
            st.write(f"Feature importances not available for {estimator_clicked}, only for XGBoost and RF")


parent_folder = os.path.dirname(os.path.abspath(__file__))
full_name = f"{selected_target_variable}_" \
            f"{selected_ml_algorithm}_" \
            f"{selected_preprocessing_model}_" \
            f"{reverse_rename[selected_validation_type]}_" \
            f"{reverse_rename[selected_threshold]}_" \
            f"{blank_to_underscore(reverse_rename[selected_class_metric])}"
if save_figure:
    file = f"{full_name}.png"
    dest_path = os.path.join(parent_folder, 'generated_results', file)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    fig.write_image(dest_path, format='png', engine='kaleido', scale=2)



if show_summary:
    
    grouped = df[['Estimator', 'Accuracy', 'Precision', 'Recall', 'F1', 'Total Support']].groupby(['Estimator']).median().reset_index()
    grouped['Accuracy'] = grouped['Accuracy'].apply(lambda x: f'{x:.3f}')
    grouped['Precision'] = grouped['Precision'].apply(lambda x: f'{x:.3f}')
    grouped['Recall'] = grouped['Recall'].apply(lambda x: f'{x:.3f}')
    grouped['F1'] = grouped['F1'].apply(lambda x: f'{x:.3f}')
    grouped['Total Support'] = grouped['Total Support'].apply(lambda x: f'{x:.0f}')
    summary = grouped[['Estimator', 'Accuracy', 'Recall', 'Precision', 'F1', 'Total Support']]
    summary = summary.rename(columns={'Estimator': 'Estimator', 'Accuracy': 'Accuracy', 'Recall': 'Recall', 'Precision': 'Precision', 'F1': 'F1', 'Total Support': 'Support'})

    st.dataframe(summary, use_container_width=True)
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


# Calculate the ratio between x and y for each data point
single_estimator = df[df['Estimator'] == 'XGBoost'].reset_index(drop=True)
single_estimator = single_estimator.sort_values(by=['Imbalance'])
index = [i for i in range(len(single_estimator))]



# plot IMbalance and support
fig = go.Figure()

# Add the bar plots
fig.add_trace(go.Bar(x=index, y=single_estimator['Support Positive'], name='Positive Support', marker_color='blue'))
fig.add_trace(go.Bar(x=index, y=-single_estimator['Support Negative'], name='Negative Support', marker_color='red'))

# Add the line plots
fig.add_trace(go.Scatter(x=index, y=single_estimator['Imbalance'], mode='lines', name='Imbalance', line=dict(color='black', width=2, dash='dash')))
fig.add_trace(go.Scatter(x=index, y=single_estimator['Total Support'], mode='lines', name='Total Support', line=dict(color='green', width=2)))

fig.update_layout(title=f'Sorted Imbalance and Support Across Target Assay Endpoints: {selected_validation_type}, y={selected_target_variable}', barmode='group')  # Set barmode to 'group' for grouped bar charts
# update layout title size, axis label size and color to black
# set x axis title

fig.update_layout(title_font=dict(size=21, color='black'))
fig.update_layout(xaxis_title_font=dict(size=axis_font_size, color="black"))
fig.update_layout(yaxis_title_font=dict(size=axis_font_size, color="black"))
fig.update_layout(legend_title_text='', legend_traceorder="reversed")


st.plotly_chart(fig, use_container_width=True)


# # limit range of y-axis
# # fig.update_yaxes(range=[0.0, max(single_estimator['Total Support']) * 1.1])


# # Display the figure in Streamlit
# st.plotly_chart(fig)
      



            
    
    


