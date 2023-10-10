import os

SSH_FILE_PATH = r"C:\Users\bossh\.ssh\gcloud.pub"

USER = "bossh@"

MACHINE_TYPE = "e2-highcpu-16"

PROJECT_ID_1 = "steel-magpie-396010"
PROJECT_ID_2 = "project-2-396114"
# PROJECT_ID_3 = "glowing-box-396212"
# PROJECT_ID_4 = "dazzling-rex-396212"
# PROJECT_ID_5 = "green-objective-396212"
# PROJECT_ID_6 = "helical-root-396212"
# PROJECT_ID_7 = "original-bond-396212"
# PROJECT_ID_8 = "robotic-branch-396212"
# PROJECT_ID_9 = "pure-genius-396212"
# PROJECT_ID_10 = "serene-mender-396212"

PROJECTS = [PROJECT_ID_1, PROJECT_ID_2]

ZONES = ["us-central1-a",
         "us-east1-b",
         "us-east4-a",
         "us-east5-a",
         # "us-west1-b",
         # "us-west2-a",
         # "us-west3-a",
         # "us-west4-a",
         # "us-south1-a",
         # "europe-west1-b",
         # "europe-west2-a",
         # "europe-west3-a",
         # "europe-west4-a",
         # "europe-west6-a",
         # "europe-west8-a",
         # "europe-west9-a",
         # "europe-west12-a",
         # "europe-southwest1-a",
         # "europe-north1-a",
         # "europe-central2-a",
         ]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
