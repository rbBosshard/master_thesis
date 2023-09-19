from diagrams import Diagram, Cluster
from diagrams.custom import Custom


# with Diagram("Custom with local icons\n Can be downloaded here: \nhttps://creativecommons.org/about/downloads/", show=False, filename="custom_local", direction="LR"):
#   cc_heart = Custom("Creative Commons", "./diagram_resources/ok.png")
#   cc_attribution = Custom("Credit must be given to the creator", "./diagram_resources/ok.png")

#   cc_sa = Custom("Adaptations must be shared\n under the same terms", "./diagram_resources/ok.png")
#   cc_nd = Custom("No derivatives or adaptations\n of the work are permitted", "./diagram_resources/ok.png")
#   cc_zero = Custom("Public Domain Dedication", "./diagram_resources/ok.png")

#   with Cluster("Non Commercial"):
#     non_commercial = [Custom("Y", "./diagram_resources/ok.png") - Custom("E", "./diagram_resources/ok.png") - Custom("S", "./diagram_resources/ok.png")]

#   cc_heart >> cc_attribution
#   cc_heart >> non_commercial
#   cc_heart >> cc_sa
#   cc_heart >> cc_nd
#   cc_heart >> cc_zero

# from diagrams import Diagram
# from diagrams.custom import Custom


# with Diagram("Custom with local icons\n Can be downloaded here: \nhttps://creativecommons.org/about/downloads/", show=False, filename="compounds", direction="LR"):

#   # Define custom icons for assay endpoints and compounds
#   assay_icon = Custom("Assay Endpoint", "./diagram_resources/ok.png")
#   compound_icon = Custom("Compound", "./diagram_resources/no.png")

#   # Set up the global context and create a diagram
#   with Diagram("Assay Endpoint Compounds", show=False):
#       # Define assay endpoints and compounds
#       assay1 = assay_icon
#       assay2 = assay_icon
#       compound_set1 = compound_icon
#       compound_set2 = compound_icon
#       compound_set_shared = compound_icon

#       # Create connections between assay endpoints and compounds
#       assay1 - compound_set1
#       assay2 - compound_set2
#       assay1 - compound_set_shared
#       assay2 - compound_set_shared

from diagrams import Diagram, Cluster
from diagrams.custom import Custom
from diagrams.generic.compute import Rack
from diagrams.generic import Node

with Diagram("Main Block", show=False, filename="compound", direction="LR"):
    with Cluster("Main Block"):
        sub_blocks = []
        for i in range(5):
            sub_blocks.append(Custom(f"Sub Block {i+1}", "./diagram_resources/no.png"))

        Rack("ok") >> sub_blocks