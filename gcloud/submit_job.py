import os
import time

from gcloud.helper import create_new_instances, delete_all_instances, get_current_instances, \
    check_gcloud_sdk_availability, start_gcloud_instances, get_script_commands, run_script_commands, \
    stop_gcloud_instances, PROJECT_ID, ZONE, run_pipeline

REBUILD = 0
INSTANCES_TOTAL = 2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# close all open ssh-browser/console windows


def main():
    check_gcloud_sdk_availability()
    gcloud_instances = get_current_instances()

    try:
        if REBUILD:
            delete_all_instances(gcloud_instances)
            gcloud_instances = create_new_instances(INSTANCES_TOTAL)
            script_path = os.path.join(ROOT_DIR, 'startup_script.sh')
            commands = get_script_commands(script_path)

        else:
            start_gcloud_instances(gcloud_instances)
            script_path = os.path.join(ROOT_DIR, 'update_script.sh')
            commands = get_script_commands(script_path)

        time.sleep(10)
        run_script_commands(PROJECT_ID, gcloud_instances, commands)
        time.sleep(2)
        run_pipeline(PROJECT_ID, gcloud_instances)
        stop_gcloud_instances(gcloud_instances)
        print("Finished")
        exit(0)

    except Exception as e:
        print(f"Error: {e}")
        stop_gcloud_instances(gcloud_instances)
        exit(1)


if __name__ == '__main__':
    main()
