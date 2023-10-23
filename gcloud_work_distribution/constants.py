import os

SSH_FILE_PATH = r"C:\Users\bossh\.ssh\gcloud.pub"

USER = "bossh@"

MACHINE_TYPE = "e2-highcpu-16"

PROJECT_ID_1 = "steel-magpie-396010"
PROJECT_ID_2 = "project-2-396114"

PROJECTS = [PROJECT_ID_1, PROJECT_ID_2]

ZONES = ["us-central1-a",
         "us-east1-b",
         "us-east4-a",
         "us-east5-a",
         ]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
