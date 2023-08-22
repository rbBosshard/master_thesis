import os
from ml.src.constants import OUTPUT_DIR_PATH, REMOTE_DIR_PATH, INPUT_DIR_PATH
from ml.src.ml_helper import load_config

CONFIG, LOGGER = load_config()

with open(os.path.join(INPUT_DIR_PATH, f"{CONFIG['fingerprint_file']}_chemicals.out"), 'r') as f:
    chemicals_with_fingerprint = set(line.strip() for line in f)

with open(os.path.join(REMOTE_DIR_PATH, f"unique_chemicals_tested.out"), 'r') as f:
    chemicals_tested = set(line.strip() for line in f)
    with open(os.path.join(INPUT_DIR_PATH, f"chemicals_tested.out"), 'w') as dest_file:
        for chemical in chemicals_tested:
            dest_file.write(chemical + '\n')


# Intersection of IDs in both files
intersection = chemicals_with_fingerprint.intersection(chemicals_tested)
chemicals_not_tested = chemicals_with_fingerprint.difference(chemicals_tested)
chemicals_without_fingerprint = chemicals_tested.difference(chemicals_with_fingerprint)

with open(os.path.join(OUTPUT_DIR_PATH, 'chemical_counts.out'), 'w') as f:
    f.write(f"Number of chemicals tested: {len(chemicals_tested)} \n")
    f.write(f"Number of chemicals with fingerprint available: {len(chemicals_with_fingerprint)} \n")
    f.write(f"Number of chemicals tested and fingerprint available: {len(intersection)} \n")
    f.write(f"Number of chemicals tested and no fingerprint available: {len(chemicals_without_fingerprint)} \n")
    f.write(f"Number of chemicals not tested but fingerprint available: {len(chemicals_not_tested)} \n")


with open(os.path.join(OUTPUT_DIR_PATH, 'chemicals_tested_with_fingerprint.out'), 'w') as f:
    for id_ in intersection:
        f.write(id_ + '\n')

with open(os.path.join(OUTPUT_DIR_PATH, 'chemicals_not_tested.out'), 'w') as f:
    for id_ in chemicals_not_tested:
        f.write(id_ + '\n')

with open(os.path.join(OUTPUT_DIR_PATH, 'chemicals_without_fingerprint.out'), 'w') as f:
    for id_ in chemicals_without_fingerprint:
        f.write(id_ + '\n')

print("Results written to files.")
