import subprocess
import datetime
import joblib
import numpy as np

MACHINE_TYPE = "e2-highcpu-16"

PROJECT_ID_1 = "steel-magpie-396010"
PROJECT_ID_2 = "project-2-396114"
PROJECT_ID_3 ="glowing-box-396212"
PROJECT_ID_4 ="dazzling-rex-396212"
PROJECT_ID_5 ="green-objective-396212"
PROJECT_ID_6 ="helical-root-396212"
PROJECT_ID_7 ="original-bond-396212"
PROJECT_ID_8 ="robotic-branch-396212"
PROJECT_ID_9 ="pure-genius-396212"
PROJECT_ID_10 ="serene-mender-396212"

ZONES = ["us-central1-a",
         "us-east1-b",
         "us-east4-a",
         "us-east5-a",
         "us-west1-b",
         "us-west2-a"
         "us-west3-a"
         "us-west4-a"
         "us-south1-a",
         "europe-west1-b",
         "europe-west2-a",
         "europe-west3-a",
         "europe-west4-a",
         "europe-west6-a",
         "europe-west8-a",
         "europe-west9-a",
         "europe-west12-a"
         "europe-southwest1-a",
         "europe-north1-a",
         "europe-central2-a",
         ]
PROJECTS = [PROJECT_ID_1, PROJECT_ID_2, PROJECT_ID_3, PROJECT_ID_4, PROJECT_ID_5, PROJECT_ID_6, PROJECT_ID_7, PROJECT_ID_8, PROJECT_ID_9, PROJECT_ID_10,]
PROJECTS += PROJECTS
SSH_FILE_PATH = r"C:\Users\bossh\.ssh\gcloud.pub"
USER = "bossh@"


def create_instance(instance):
    print(f"Creating {instance}...")
    create_instance_command = [
        "gcloud", "compute", "instances", "create", instance,
        "--project", get_project(instance), "--zone", get_zone(instance), "--machine-type", MACHINE_TYPE,
        f'--metadata-from-file=ssh-keys={SSH_FILE_PATH}'
    ]
    subprocess.run(create_instance_command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, text=True)
    print(f"Created {instance}")
    return instance


def create_new_instances(instances_total):
    gcloud_instances = [f"{ZONES[i]}-{PROJECTS[i]}" for i in range(instances_total)]

    with joblib.Parallel(n_jobs=instances_total or 1) as parallel:
        new_instances = parallel(
            joblib.delayed(create_instance)(gcloud_instance) for gcloud_instance in gcloud_instances)

    print("Instances creation done.")
    return new_instances


def delete_instance(instance):
    print(f"Deleting {instance}...")
    delete_instance_command = [
        "gcloud", "compute", f"instances", "delete",
        instance, "--project", get_project(instance), "--zone", get_zone(instance), "--quiet"
    ]
    subprocess.run(delete_instance_command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, text=True)
    print(f"Deleted {instance}")
    return instance


def get_zone(instance):
    zone = "-".join(instance.split("-")[:3])
    return zone


def get_project(instance):
    project = "-".join(instance.split("-")[3:])
    return project


def delete_instances(gcloud_instances):
    if gcloud_instances != ['']:
        with joblib.Parallel(n_jobs=len(gcloud_instances) or 1) as parallel:
            deleted_instances = parallel(joblib.delayed(delete_instance)(instance) for instance in gcloud_instances)
        print("Instances deletion done")
        return deleted_instances


def get_current_instances():
    gcloud_instances = []
    for project in np.unique(PROJECTS):
        list_instances_command = ["gcloud", "compute", "instances", "list", "--project", project, "--format=value(name)"]
        result = subprocess.run(list_instances_command, shell=True, capture_output=True, text=True)
        gcloud_instances += result.stdout.strip().split('\n')
    print(f"Current gcloud instances: {gcloud_instances}")
    return gcloud_instances


def check_gcloud_sdk_availability():
    try:
        subprocess.run(["gcloud", "--version"], shell=True, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        print("Please install and configure the Google Cloud SDK (gcloud).")
        exit(1)


def start_instance(instance):
    ssh_command = f'gcloud compute instances start {instance} --zone={get_zone(instance)} --project={get_project(instance)}'
    subprocess.run(ssh_command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    return instance


def start_gcloud_instances(gcloud_instances):
    with joblib.Parallel(n_jobs=len(gcloud_instances)) as parallel:
        started_instances = parallel(joblib.delayed(start_instance)(instance) for instance in gcloud_instances)

    print("Instances started")
    return started_instances


def get_script_commands(script_path):
    with open(script_path, 'r') as f:
        script = f.read()
    commands = script.split('\n')
    commands = [cmd for cmd in commands if cmd.strip()]
    return commands


def run_commands_on_instance(instance, commands):
    for cmd in commands:
        print(f"{instance}: {cmd}")
        ssh_command = (
            f'gcloud compute ssh {instance} '
            f'--project={get_project(instance)} '
            f'--zone={get_zone(instance)} '
            f'--command="{cmd}"'
        )
        subprocess.run(
            ssh_command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE
        )
    return instance


def run_script_commands(gcloud_instances, commands):
    with joblib.Parallel(n_jobs=len(gcloud_instances) or 1) as parallel:
        result_instances = parallel(
            joblib.delayed(run_commands_on_instance)(
                instance, commands
            )
            for instance in gcloud_instances
        )

    print("Script commands executed on instances: ", result_instances)
    return result_instances


def run_pipeline_on_instance(instance, cmd):
    print(f"{instance}: {cmd}")
    ssh_command = (
        f'gcloud compute ssh {instance} '
        f'--project={get_project(instance)} '
        f'--zone={get_zone(instance)} '
        f'--command="{cmd}"'
    )
    subprocess.run(
        ssh_command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE
    )
    print("Pipeline done on: ", instance)
    return instance


def run_pipeline(gcloud_instances):
    n = len(gcloud_instances)
    commands = [f"sudo pip install -r pytcpl/requirements.txt && " \
                f"sudo python3 pytcpl/src/pipeline.py " \
                f"--instance_id {i} --instances_total {n} " for i in range(n)]
    with joblib.Parallel(n_jobs=len(gcloud_instances)) as parallel:
        result_instances = parallel(
            joblib.delayed(run_pipeline_on_instance)(
                instance, commands[i]
            )
            for i, instance in enumerate(gcloud_instances)
        )

    return result_instances


def git_commit_instances(gcloud_instances):
    for i, instance in enumerate(gcloud_instances):
        git_commit(instance, i)


def git_commit(instance, instance_id):
    commit_msg = f"'{instance} i={instance_id} @ {datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}'"
    ssh_command = (f'gcloud compute ssh {instance} '
                   f'--project={get_project(instance)} '
                   f'--zone={get_zone(instance)} '
                   f'--command="cd pytcpl && sudo git config --global --add safe.directory /home/bossh/pytcpl && '
                   f'sudo git add . && sudo git commit -m {commit_msg} && '
                   f'sudo git pull origin main --rebase && sudo git push origin main')
    subprocess.run(ssh_command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)


def stop_instance(instance):
    print(f"Stopping {instance}..")
    ssh_command = f'gcloud compute instances stop {instance} --zone={get_zone(instance)} --project={get_project(instance)}'
    subprocess.run(ssh_command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    return instance


def stop_gcloud_instances(gcloud_instances):
    with joblib.Parallel(n_jobs=len(gcloud_instances)) as parallel:
        stopped_instances = parallel(joblib.delayed(stop_instance)(instance) for instance in gcloud_instances)
    print("All instances stopped")
    return stopped_instances
