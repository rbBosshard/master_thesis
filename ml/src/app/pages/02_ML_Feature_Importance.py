import pandas as pd
import numpy as np
import seaborn as sns
import os
import json
from ml.src.pipeline.constants import OUTPUT_DIR_PATH
from sklearn.metrics import jaccard_score
import plotly.express as px

import matplotlib.pyplot as plt


TARGET_RUN = "2023-10-18_22-36-10"


logs_folder = os.path.join(OUTPUT_DIR_PATH)
folder = os.path.join(logs_folder, TARGET_RUN)

with open(os.path.join(folder, 'aeid_paths.json'), 'r') as fp:
    aeid_paths = json.load(fp)





def get_figure(aeid_paths, top_n, name):
    all_features = pd.DataFrame()
    unique_features = set()



    for aeid in aeid_paths[target_variable][ml_algorithm].keys():
        aeid_path = aeid_paths[target_variable][ml_algorithm][aeid]
        sorted_feature_importances_path = os.path.join(aeid_path, preprocessing_model, estimator_model, 'mb_val_sirius', 'sorted_feature_importances.csv')
        sorted_feature_importances = pd.read_csv(sorted_feature_importances_path).reset_index(drop=True).head(top_n)
    
    # Collect unique feature indexes
        unique_features.update(sorted_feature_importances['feature'])
        

        sorted_feature_importances['aeid'] = aeid
        all_features = pd.concat([all_features, sorted_feature_importances], ignore_index=True)

# Create a mapping of old feature indexes to new linear indexes
    feature_index_mapping = {feature_index: linear_index for linear_index, feature_index in enumerate(unique_features)}

# Replace the original feature indexes with the new linear indexes
    all_features['linearized_feature_index'] = all_features['feature'].map(feature_index_mapping)

# log transorm importances
    all_features['importances'] = all_features['importances'] ** 0.001


    nbinsx = all_features['aeid'].unique().shape[0]
    nbinsy = all_features['feature'].unique().shape[0]


    fig = px.density_heatmap(all_features,
                        x='aeid',
                        y="linearized_feature_index",
                        hover_data=["feature", "importances"],
                        nbinsx=nbinsx,
                        nbinsy=nbinsy,
                        marginal_y="histogram",
                        z="importances",
                        color_continuous_scale=[[0.0, "white"], [1.0, "black"]]
                        )

    fig.update_layout(
    autosize=False,
    width=1000,
    height=1000,
    margin=dict(
        l=50,
        r=50,
        b=50,
        t=50,
        pad=4
    ),
    xaxis_title="Assay Index",
    yaxis_title="Feature index",
    font=dict(
        size=18,
        )
    )

# dont sho x lables
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

# hide color bar
    fig.update_layout(coloraxis_showscale=False)

    dest_path = os.path.join(name, 'feature_importance.png')
    fig.write_image(dest_path, format='png', engine='kaleido', scale=2)

    return fig, all_features


# set main def

if __name__ == '__main__':
    # set params
    heatmap = 1
    top_n = 10

    target_variable = 'hitcall'
    ml_algorithm = 'classification'
    preprocessing_model = 'Feature_Selection_XGBClassifier'
    estimator_model = 'XGBClassifier'

    fig, all_features1 = get_figure(aeid_paths, top_n)

    target_variable = 'hitcall'
    ml_algorithm = 'classification'
    preprocessing_model = 'Feature_Selection_RandomForestClassifier'
    estimator_model = 'RandomForestClassifier'
    fig, all_features2 = get_figure(aeid_paths, top_n, f"{preprocessing_model}__{preprocessing_model}")

    # get jaccard score
    jaccard_score = jaccard_score(all_features1['feature'].values, all_features2['feature'].values)
    print(jaccard_score)




