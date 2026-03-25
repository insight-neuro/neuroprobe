"""
Dry-run data check — verifies all required files exist for the Neuroprobe Lite benchmark
before running the full GNN pipeline.

Usage:
    ROOT_DIR_BRAINTREEBANK=/storage/eg99/braintreebank_data python check_data.py

    # If using original features.csv instead of test_new_features.csv:
    NEUROPROBE_FEATURES_FILE=features.csv ROOT_DIR_BRAINTREEBANK=... python check_data.py
"""
import os
import sys

ROOT_DIR = os.environ.get('ROOT_DIR_BRAINTREEBANK')
if not ROOT_DIR:
    print("ERROR: ROOT_DIR_BRAINTREEBANK not set")
    sys.exit(1)

FEATURES_FILE = os.environ.get('NEUROPROBE_FEATURES_FILE', 'test_new_features.csv')

print(f"Checking data root: {ROOT_DIR}")
print(f"Features file: {FEATURES_FILE}\n")

LITE_SUBJECT_TRIALS = [
    (1, 1), (1, 2),
    (2, 0), (2, 4),
    (3, 0), (3, 1),
    (4, 0), (4, 1),
    (7, 0), (7, 1),
    (10, 0), (10, 1),
]

# Subject metadata needed for all subjects in Lite
LITE_SUBJECTS = sorted(set(s for s, _ in LITE_SUBJECT_TRIALS))
# CrossSubject train pair
CROSS_SUBJECT_TRAIN = (2, 4)

missing = []

def check(path, label):
    exists = os.path.exists(path)
    status = "OK " if exists else "MISSING"
    print(f"  [{status}] {label}")
    if not exists:
        missing.append(path)

# --- Per-subject files ---
print("=== Per-subject files ===")
for subject_id in LITE_SUBJECTS:
    check(
        os.path.join(ROOT_DIR, f'localization/sub_{subject_id}/depth-wm.csv'),
        f"sub_{subject_id} localization/depth-wm.csv"
    )
    check(
        os.path.join(ROOT_DIR, f'electrode_labels/sub_{subject_id}/electrode_labels.json'),
        f"sub_{subject_id} electrode_labels.json"
    )

# --- Per-trial h5 files ---
print("\n=== Neural data h5 files ===")
for subject_id, trial_id in LITE_SUBJECT_TRIALS:
    check(
        os.path.join(ROOT_DIR, f'sub_{subject_id}_trial{trial_id:03}.h5'),
        f"sub_{subject_id}_trial{trial_id:03}.h5"
    )

# --- Shared files ---
print("\n=== Shared files ===")
check(os.path.join(ROOT_DIR, 'corrupted_elec.json'), 'corrupted_elec.json')
check(os.path.join(ROOT_DIR, 'localization/elec_coords_full.csv'), 'localization/elec_coords_full.csv')

# --- Transcript files (one per movie) ---
print(f"\n=== Transcript files ({FEATURES_FILE}) ===")
transcripts_dir = os.path.join(ROOT_DIR, 'transcripts')
if os.path.isdir(transcripts_dir):
    movies = sorted(os.listdir(transcripts_dir))
    for movie in movies:
        check(
            os.path.join(ROOT_DIR, f'transcripts/{movie}/{FEATURES_FILE}'),
            f"transcripts/{movie}/{FEATURES_FILE}"
        )
else:
    print(f"  [MISSING] transcripts/ directory not found at {transcripts_dir}")
    missing.append(transcripts_dir)

# --- Subject timings ---
print("\n=== Subject timings ===")
for subject_id, trial_id in LITE_SUBJECT_TRIALS:
    check(
        os.path.join(ROOT_DIR, f'subject_timings/sub_{subject_id}_trial{trial_id:03}_timings.csv'),
        f"subject_timings sub_{subject_id}_trial{trial_id:03}"
    )

# --- Scene annotations (for scene_onset task) ---
print("\n=== Scene annotations ===")
check(
    os.path.join(ROOT_DIR, 'scene_annotations.json'),
    'scene_annotations.json'
)

# --- Summary ---
print(f"\n{'='*50}")
if not missing:
    print("All files present — good to go!")
else:
    print(f"MISSING {len(missing)} file(s):")
    for f in missing:
        print(f"  {f}")
    print("\nAdd symlinks for missing items, e.g.:")
    print("  ln -s /storage/czw/braintreebank_data/<item> /storage/eg99/braintreebank_data/")
