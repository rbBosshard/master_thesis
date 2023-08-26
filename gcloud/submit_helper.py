import subprocess
import datetime
import joblib
import numpy as np
import concurrent.futures

from gcloud.constants import MACHINE_TYPE, ZONES, PROJECTS, SSH_FILE_PATH

PROJECTS += PROJECTS

# Todo: Replace with real-time output for evers subprocess.run -> subprocess.Popen
# with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True) as process:
#     for line in process.stdout:
#         print(line, end='')  # Print the captured output in real-time
#     for line in process.stderr:
#         print(line, end='')  # Print the captured error output in real-time
#     process.wait()  # Wait for the subprocess to finish


def get_zone(instance):
    zone = "-".join(instance.split("-")[:3])
    return zone


def get_project(instance):
    project = "-".join(instance.split("-")[3:])
    return project


def check_gcloud_sdk_availability():
    try:
        completed_process = subprocess.run(["gcloud", "--version"], shell=True, capture_output=True, text=True, check=True)
        print(completed_process.stdout)
        print(completed_process.stderr)
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
    delete_instance_command = [
        "gcloud", "compute", f"instances", "delete",
        instance, "--project", get_project(instance), "--zone", get_zone(instance), "--quiet"
    ]
    completed_process = subprocess.run(delete_instance_command, shell=True, capture_output=True, text=True)
    print(completed_process.stdout)
    print(completed_process.stderr)
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
    create_instance_command = [
        "gcloud", "compute", "instances", "create", instance,
        "--project", get_project(instance), "--zone", get_zone(instance), "--machine-type", MACHINE_TYPE,
        f'--metadata-from-file=ssh-keys={SSH_FILE_PATH}'
    ]
    completed_process = subprocess.run(create_instance_command, capture_output=True, text=True)
    print(completed_process.stdout)
    print(completed_process.stderr)
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
    ssh_command = f'gcloud compute instances start {instance} --zone={get_zone(instance)} --project={get_project(instance)}'
    completed_process = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
    print(completed_process.stdout)
    print(completed_process.stderr)
    return instance


def start_gcloud_instances(instances):
    with joblib.Parallel(n_jobs=len(instances)) as parallel:
        started_instances = parallel(joblib.delayed(start_instance)(instance) for instance in instances)

    print("Instances started")
    return started_instances


def stop_instance(instance):
    print(f"Stopping {instance}..")
    ssh_command = f'gcloud compute instances stop {instance} --zone={get_zone(instance)} --project={get_project(instance)}'
    completed_process = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
    print(completed_process.stdout)
    print(completed_process.stderr)
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
        ssh_command = (
            f'gcloud compute ssh {instance} '
            f'--project={get_project(instance)} '
            f'--zone={get_zone(instance)} '
            f'--command="{cmd}"'
        )
        completed_process = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
        print(completed_process.stdout)
        print(completed_process.stderr)
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
    ssh_command = (
        f'gcloud compute ssh {instance} '
        f'--project={get_project(instance)} '
        f'--zone={get_zone(instance)} '
        f'--command="{cmd}"'
    )
    completed_process = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
    print(completed_process.stdout)
    print(completed_process.stderr)
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
    ssh_command = (f'gcloud compute ssh {instance} '
                   f'--project={get_project(instance)} '
                   f'--zone={get_zone(instance)} '
                   f'--command="cd pytcpl && sudo git config --global --add safe.directory /home/bossh/pytcpl && '
                   f'sudo git add . && sudo git commit -m {commit_msg} && '
                   f'sudo git pull origin main --rebase && sudo git push origin main')
    completed_process = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
    print(completed_process.stdout)
    print(completed_process.stderr)


def git_commit_instances(instances):
    for i, instance in enumerate(instances):
        git_commit(instance, i)


def wrap_up(instance):
    cmd = [f"sudo python3 pytcpl/src/pipeline/pipeline_wrapup.py"]
    print(f"{instance}: {cmd}")
    ssh_command = (
        f'gcloud compute ssh {instance} '
        f'--project={get_project(instance)} '
        f'--zone={get_zone(instance)} '
        f'--command="{cmd}"'
    )
    completed_process = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
    print(completed_process.stdout)
    print(completed_process.stderr)
    print("Pipeline wrapup done on: ", instance)
