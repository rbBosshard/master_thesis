from diagrams import Diagram
from diagrams.custom import Custom

# Define custom icons for assay endpoints and compounds
assay_icon = Custom("Assay Endpoint", "figures\ml_dataset_visualization.png")
compound_icon = Custom("Compound", "figures\presence_matrix_aeid_chid.png")

# Create a diagram
with Diagram("Assay Endpoint Compounds", show=False):
    # Define assay endpoints and compounds
    assay1 = assay_icon
    assay2 = assay_icon
    compound_set1 = compound_icon
    compound_set2 = compound_icon
    compound_set_shared = compound_icon

    # Create connections between assay endpoints and compounds
    assay1 - compound_set1
    assay2 - compound_set2
    assay1 - compound_set_shared
    assay2 - compound_set_shared

from diagrams import Diagram, Cluster
from diagrams.custom import Custom
from diagrams.aws.general import Generic
from diagrams.generic import Node

with Diagram("Main Block", show=False):
    with Cluster("Main Frame"):
        # Add list items as nodes
        Node("Item 1")
        Node("Item 2")
        Node("Item 3")
        # ... Add more items as needed

