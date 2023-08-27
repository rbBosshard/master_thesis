import os
from ml.src.constants import REMOTE_METADATA_SUBSET_DIR_PATH, METADATA_DIR_PATH
from ml.src.ml_helper import load_config

CONFIG = load_config()[0]

with open(os.path.join(METADATA_DIR_PATH, 'fps', f"{CONFIG['fingerprint_file']}_compounds.out"), 'r') as f:
    compounds_with_fingerprint = set(line.strip() for line in f)

with open(os.path.join(METADATA_DIR_PATH, f"compounds_tested.out"), 'r') as f:
    compounds_tested = set(line.strip() for line in f)
    with open(os.path.join(METADATA_DIR_PATH, f"compounds_tested.out"), 'w') as dest_file:
        for compound in compounds_tested:
            dest_file.write(compound + '\n')

# Intersection of IDs in both files
intersection = compounds_with_fingerprint.intersection(compounds_tested)
compounds_not_tested = compounds_with_fingerprint.difference(compounds_tested)
compounds_without_fingerprint = compounds_tested.difference(compounds_with_fingerprint)

with open(os.path.join(METADATA_DIR_PATH, 'compounds', 'compounds_count.out'), 'w') as f:
    f.write(f"Number of compounds tested: {len(compounds_tested)} \n")
    f.write(f"Number of compounds with fingerprint available: {len(compounds_with_fingerprint)} \n")
    f.write(f"Number of compounds tested and fingerprint available: {len(intersection)} \n")
    f.write(f"Number of compounds tested and no fingerprint available: {len(compounds_without_fingerprint)} \n")
    f.write(f"Number of compounds not tested but fingerprint available: {len(compounds_not_tested)} \n")


with open(os.path.join(METADATA_DIR_PATH, 'compounds', 'compounds_tested_with_fingerprint.out'), 'w') as f:
    for id_ in intersection:
        f.write(id_ + '\n')

with open(os.path.join(METADATA_DIR_PATH, 'compounds', 'compounds_not_tested.out'), 'w') as f:
    for id_ in compounds_not_tested:
        f.write(id_ + '\n')

with open(os.path.join(METADATA_DIR_PATH, 'compounds', 'compounds_without_fingerprint.out'), 'w') as f:
    for id_ in compounds_without_fingerprint:
        f.write(id_ + '\n')

print("Results written to files.")
