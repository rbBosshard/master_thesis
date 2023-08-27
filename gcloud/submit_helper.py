import os
import subprocess
import datetime
from datetime import datetime

import joblib
import numpy as np
import concurrent.futures

from gcloud.constants import MACHINE_TYPE, ZONES, PROJECTS, SSH_FILE_PATH

LOG_FOLDER = os.path.join('logs')
LOG_FILE = ""

PROJECTS += PROJECTS


def get_zone(instance):
    zone = "-".join(instance.split("-")[:3])
    return zone


def get_project(instance):
    project = "-".join(instance.split("-")[3:])
    return project


def check_gcloud_sdk_availability():
    try:
        command = ["gcloud", "--version"]
        submit_command(command)

    except subprocess.CalledProcessError:
        print("Please install and configure the Google Cloud SDK (gcloud).")
        exit(1)


def get_instances_for_project(project):
    print(f"Getting instances for: {project}")
    list_instances_command = ["gcloud", "compute", "instances", "list", "--project", project, "--format=value(name)"]
    try:
        completed_process = subprocess.run(list_instances_command, shell=True, capture_output=True, text=True)
        print(completed_process.stdout)
        print(completed_process.stderr)

        instances = completed_process.stdout.strip().split('\n')
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for project: {project}")
        instances = []

    print(f"For {project} found instances: {instances}")
    return instances


def get_current_instances():
    instances = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(get_instances_for_project, np.unique(PROJECTS[1:4]))

    for result in results:
        instances += result

    print(f"Current gcloud instances: {instances}")
    return instances


def delete_instance(instance):
    print(f"Deleting {instance}...")
    command = [
        "gcloud", "compute", f"instances", "delete",
        instance, "--project", get_project(instance), "--zone", get_zone(instance), "--quiet"
    ]
    submit_command(command)

    print(f"Deleted {instance}")
    return instance


def delete_instances(instances):
    if instances != ['']:
        with joblib.Parallel(n_jobs=len(instances) or 1) as parallel:
            deleted_instances = parallel(joblib.delayed(delete_instance)(instance) for instance in instances)
        print("Instances deletion done")
        return deleted_instances


def create_instance(instance):
    print(f"Creating {instance}...")
    command = [
        "gcloud", "compute", "instances", "create", instance,
        "--project", get_project(instance), "--zone", get_zone(instance), "--machine-type", MACHINE_TYPE,
        f'--metadata-from-file=ssh-keys={SSH_FILE_PATH}'
    ]
    submit_command(command)

    print(f"Created {instance}")
    return instance


def create_new_instances(instances_total):
    instances = [f"{ZONES[i]}-{PROJECTS[i]}" for i in range(instances_total)]

    with joblib.Parallel(n_jobs=instances_total or 1) as parallel:
        created_instances = parallel(
            joblib.delayed(create_instance)(instance) for instance in instances)

    print("Instances creation done.")
    return created_instances


def start_instance(instance):
    command = f'gcloud compute instances start {instance} --zone={get_zone(instance)} --project={get_project(instance)}'
    submit_command(command)

    return instance


def start_gcloud_instances(instances):
    with joblib.Parallel(n_jobs=len(instances)) as parallel:
        started_instances = parallel(joblib.delayed(start_instance)(instance) for instance in instances)

    print("Instances started")
    return started_instances


def stop_instance(instance):
    print(f"Stopping {instance}..")
    command = f'gcloud compute instances stop {instance} --zone={get_zone(instance)} --project={get_project(instance)}'
    submit_command(command)

    return instance


def stop_gcloud_instances(instances):
    with joblib.Parallel(n_jobs=len(instances) or 1) as parallel:
        stopped_instances = parallel(joblib.delayed(stop_instance)(instance) for instance in instances)
    print("All instances stopped")
    return stopped_instances


def get_script_commands(script_path):
    with open(script_path, 'r') as f:
        script = f.read()
    commands = script.split('\n')
    commands = [cmd for cmd in commands if cmd.strip()]
    return commands


def run_commands_on_instance(instance, commands):
    for cmd in commands:
        print(f"{instance}: {cmd}")
        command = (
            f'gcloud compute ssh {instance} '
            f'--project={get_project(instance)} '
            f'--zone={get_zone(instance)} '
            f'--command="{cmd}"'
        )
        submit_command(command)

    return instance


def run_script_commands(instances, commands):
    with joblib.Parallel(n_jobs=len(instances) or 1) as parallel:
        result_instances = parallel(
            joblib.delayed(run_commands_on_instance)(
                instance, commands
            )
            for instance in instances
        )

    print("Script commands executed on instances: ", result_instances)
    return result_instances


def run_pipeline_on_instance(instance, cmd):
    print(f"{instance}: {cmd}")
    command = (
        f'gcloud compute ssh {instance} '
        f'--project={get_project(instance)} '
        f'--zone={get_zone(instance)} '
        f'--command="{cmd}"'
    )
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    for line in process.stdout:
        print(line, end='')
    for line in process.stderr:
        print(line, end='')

    print("Pipeline done on: ", instance)
    return instance


def run_pipeline(instances):
    n = len(instances)
    commands = [f"sudo pip install -r pytcpl/requirements.txt && " \
                f"sudo git config --global --add safe.directory /home/bossh/pytcpl && " \
                f"sudo git -C pytcpl/ reset --hard HEAD && " \
                f"sudo git -C pytcpl/ pull origin main --rebase && " \
                f"sudo python3 pytcpl/src/pipeline/pipeline_main.py " \
                f"--instance_id {i} --instances_total {n} " for i in range(n)]
    with joblib.Parallel(n_jobs=len(instances)) as parallel:
        result_instances = parallel(
            joblib.delayed(run_pipeline_on_instance)(
                instance, commands[i]
            )
            for i, instance in enumerate(instances)
        )

    return result_instances


def git_commit(instance, instance_id):
    commit_msg = f"'{instance} i={instance_id} @ {datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}'"
    command = (f'gcloud compute ssh {instance} '
               f'--project={get_project(instance)} '
               f'--zone={get_zone(instance)} '
               f'--command="cd pytcpl && sudo git config --global --add safe.directory /home/bossh/pytcpl && '
               f'sudo git add . && sudo git commit -m {commit_msg} && '
               f'sudo git pull origin main --rebase && sudo git push origin main')
    submit_command(command)


def git_commit_instances(instances):
    for i, instance in enumerate(instances):
        git_commit(instance, i)

    print("Git commit done on instances: ", instances)


def wrap_up(instance):
    cmd = f"sudo python3 pytcpl/src/pipeline/pipeline_wrapup.py"
    print(f"{instance}: {cmd}")
    command = (
        f'gcloud compute ssh {instance} '
        f'--project={get_project(instance)} '
        f'--zone={get_zone(instance)} '
        f'--command="{cmd}"'
    )
    submit_command(command)

    print("Pipeline wrapup done on: ", instance)


def submit_command(command):
    with open(LOG_FILE, 'a', encoding="utf-8") as f:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True)
        for line in process.stdout:
            encoded_line = line.encode("utf-8")  # Encode line as UTF-8
            f.write(encoded_line.decode("utf-8"))  # Decode and write line
        for line in process.stderr:
            encoded_line = line.encode("utf-8")  # Encode line as UTF-8
            f.write(encoded_line.decode("utf-8"))  # Decode and write line
        process.wait()


def init_log_file():
    global LOG_FILE
    os.makedirs(LOG_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_FILE = os.path.join(LOG_FOLDER, f"{timestamp}.log")
    with open(LOG_FILE, 'w', encoding="utf-8") as f:
        pass


# function is called automatically when this module is imported into another script
init_log_file()
