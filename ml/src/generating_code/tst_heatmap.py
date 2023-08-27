import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a sample presence matrix (replace this with your actual data)
presence_matrix = np.array([
    [0, 1, 2, 0],
    [0, 2, 0, 1],
    [1, 0, 0, 0],
    [2, 1, 0, 2]
])

# Define a custom color map
custom_cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

# Create a heatmap plot with the custom color map
plt.figure(figsize=(10, 6))
sns.heatmap(presence_matrix, annot=True, fmt="d", cmap=custom_cmap,
            cbar_kws={'label': 'Presence'}, vmin=0, vmax=2)
plt.xlabel("Compound IDs")
plt.ylabel("Dataframe Index")
plt.title("Compound Presence Heatmap")
plt.show()