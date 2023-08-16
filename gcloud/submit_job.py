import os
import datetime
import subprocess

try:
    subprocess.run(["gcloud", "--version"], shell=True, capture_output=True, text=True, check=True)
except subprocess.CalledProcessError:
    print("Please install and configure the Google Cloud SDK (gcloud).")
    exit(1)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

project_id = "steel-magpie-396010"
zone = "asia-east1-b"
machine_type = "e2-highcpu-16"
new_instance = f"gcloud-instance-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

list_instances_command = ["gcloud", "compute", "instances", "list", "--format=value(name)"]
result = subprocess.run(list_instances_command, shell=True, capture_output=True, text=True)
instances_list = result.stdout.strip().split('\n')

for old_instance in instances_list:
    print(f"Deleting {old_instance}...")
    delete_instance_command = [
        "gcloud", "compute", "instances", "delete",
        old_instance, "--project", project_id, "--zone", zone, "--quiet"
    ]
    subprocess.run(delete_instance_command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, text=True)

print("Instances deletion done.")



create_instance_command = [
    "gcloud", "compute", "instances", "create", new_instance,
    "--project", project_id, "--zone", zone, "--machine-type", machine_type,
]
subprocess.run(create_instance_command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, text=True)


# KEY_FILE_PATH = r"C:\Users\bossh\.ssh\id_ed25519_gcloud.pub"
#
# ssh_command = (f'gcloud compute os-login ssh-keys add '
#                f'--project={project_id} '
#                f'--key-file={KEY_FILE_PATH} ')
# subprocess.run(ssh_command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)


startup_script_path = os.path.join(ROOT_DIR, 'startup_script.sh')
with open(startup_script_path, 'r') as f:
    startup_script = f.read()

# Split the script into individual commands
startup_commands = startup_script.split('\n')

# Remove any empty lines
startup_commands = [cmd for cmd in startup_commands if cmd.strip()]

print("###################################")

for cmd in startup_commands:
    print(f"Command: {cmd}")
    ssh_command = (f'gcloud compute ssh {new_instance} '
                   f'--project={project_id} '
                   f'--zone={zone} '
                   f'--command="{cmd}"')
    subprocess.run(ssh_command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

# Commit
commit_msg = f"pipeline run by {new_instance}"
ssh_command = (f'gcloud compute ssh {new_instance} '
               f'--project={project_id} '
               f'--zone={zone} '
               f'--command="sudo git config --global --add safe.directory /home/bossh/pytcpl && sudo git add . && sudo git commit -m {commit_msg} && sudo git push"')
subprocess.run(ssh_command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
print("Finished")

