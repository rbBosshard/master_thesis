import os
import time

from gcloud.helper import create_new_instances, delete_instances, get_current_instances, \
    check_gcloud_sdk_availability, start_gcloud_instances, get_script_commands, run_script_commands, \
    stop_gcloud_instances, run_pipeline, git_commit_instances

BUILD = 1
INSTANCES_TOTAL = 4
KEEP = 0
STOP = 0

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# close all open ssh-browser/console windows
# gcloud compute os-login ssh-keys add --key-file="C:\Users\bossh\.ssh\gcloud.pub" --project=project-2-396114


def main():
    check_gcloud_sdk_availability()
    gcloud_instances = get_current_instances()
    try:
        if STOP:
            raise Exception("STOP")
        if BUILD:
            delete_instances(gcloud_instances)
            new_instances = create_new_instances(INSTANCES_TOTAL)
            time.sleep(15)

            gcloud_instances = new_instances
            script_path = os.path.join(ROOT_DIR, 'startup_script.sh')
            commands = get_script_commands(script_path)
            run_script_commands(new_instances, commands)

        start_gcloud_instances(gcloud_instances)
        time.sleep(5)

        script_path = os.path.join(ROOT_DIR, 'update_script.sh')
        commands = get_script_commands(script_path)
        run_script_commands(gcloud_instances, commands)

        time.sleep(1)

        run_pipeline(gcloud_instances)
        time.sleep(1)
        git_commit_instances(gcloud_instances)
        time.sleep(1)

        stop_gcloud_instances(gcloud_instances)
        print("Finished")
        exit(0)

    except Exception as e:
        print(f"Error: {e}")
        stop_gcloud_instances(gcloud_instances)
        exit(1)


if __name__ == '__main__':
    main()
