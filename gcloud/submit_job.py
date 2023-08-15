import os
import time
import datetime
import subprocess


# Please install gcloud console!
print("Starting the job submission process...")

# Determine the root directory dynamically
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

result = subprocess.run(["where", "gcloud.cmd"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
gcloud = result.stdout.strip().split('\n')[0].strip()


# GitHub Settings
github_username = "rbBosshard"
github_repository = "pytcpl"


# GCP Project and Instance Settings
project_id = "steel-magpie-396010"
zone = "asia-east1-b" #"asia-east1-b"
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
instance_name = f"gcloud-instance-{timestamp}"
machine_type = "e2-highcpu-16" #"e2-highcpu-16"

list_instances_command = f"{gcloud} compute instances list --format=value(name)"

result = subprocess.run(list_instances_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
instances_list = result.stdout.strip().split('\n')

if instances_list:
    for name in instances_list:
        delete_instance_command = f"{gcloud} compute instances delete {name} --project {project_id} --zone {zone} --quiet"
        deletion_process = subprocess.run(delete_instance_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Deletion done: instance: {name}")
    print("Instances deletion done.")
else:
    print("No instances found. Nothing to delete.")

# Create a startup script file
startup_script_path = os.path.join(ROOT_DIR, "startup_script.sh")

# Create and start the instance
create_instance_command =f"{gcloud} compute instances create {instance_name} --project={project_id} --zone={zone} --machine-type={machine_type}"
create_process = subprocess.Popen(create_instance_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


with open(startup_script_path, "r") as startup_script_file:
    startup_script = startup_script_file.read()

# Establish SSH connection and stream logs
gcloud_ssh_command = f"{gcloud} compute ssh {instance_name} --project={project_id} --zone={zone} --command='{startup_script}'"

print(f"SSH Command: {gcloud_ssh_command }")

subprocess.run(gcloud_ssh_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# # Wait for the instance to finish and then shut it down
# time.sleep(60)  # Adjust wait time as needed

# shutdown_instance_command = [
#     gcloud_cmd_path, "compute", "instances", "delete", instance_name,
#     "--project", project_id,
#     "--zone", zone
# ]
# shutdown_process = subprocess.Popen(shutdown_instance_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


time.sleep(3600)  # Adjust wait time as needed
# Clean up the startup script file


