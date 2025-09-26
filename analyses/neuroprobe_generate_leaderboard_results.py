import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import math
import time
import neuroprobe.config as neuroprobe_config

### PARSE ARGUMENTS ###

import argparse
parser = argparse.ArgumentParser(description='Create performance figure for BTBench evaluation')
parser.add_argument('--split_type', type=str, default='SS_DM', 
                    help='Split type to use (SS_SM or SS_DM or DS_DM)')
args = parser.parse_args()
split_type = args.split_type

metric = 'AUROC' # 'AUROC'
assert metric == 'AUROC', 'Metric must be AUROC; no other metric is supported'

### DEFINE MODELS ###

# Map split_type to leaderboard folder name
split_type_mapping = {
    'SS_SM': 'Within-Session',
    'SS_DM': 'Cross-Session', 
    'DS_DM': 'Cross-Subject'
}

models = [
    {
        'name': 'Linear (raw voltage)',
        'short_name': 'Linear (voltage)',
        'color_palette': 'viridis',
        'eval_results_path': f'/om2/user/zaho/neuroprobe/data/eval_results_lite_{split_type}/linear_voltage/',
        'description': 'Linear regression model using raw voltage signals',
        'author': 'Andrii Zahorodnii',
        'organization': 'MIT',
        'organization_url': 'https://mit.edu'
    },
    {
        'name': 'Linear (spectrogram)',
        'short_name': 'Linear (spectrogram)',
        'color_palette': 'viridis', 
        'eval_results_path': f'/om2/user/zaho/neuroprobe/data/eval_results_lite_{split_type}/linear_stft_abs_nperseg512_poverlap0.75_maxfreq150/',
        'description': 'Linear regression model using spectrogram features',
        'author': 'Andrii Zahorodnii',
        'organization': 'MIT',
        'organization_url': 'https://mit.edu'
    },
    {
        'name': 'Linear (Laplacian re-referencing + spectrogram)',
        'short_name': 'Linear (Laplacian+spectrogram)',
        'color_palette': 'viridis', 
        'eval_results_path': f'/om2/user/zaho/neuroprobe/data/eval_results_lite_{split_type}/linear_laplacian-stft_abs_nperseg512_poverlap0.75_maxfreq150/',
        'description': 'Linear regression model using Laplacian re-referencing and spectrogram features',
        'author': 'Andrii Zahorodnii',
        'organization': 'MIT',
        'organization_url': 'https://mit.edu'
    },
    {
        'name': 'BrainBERT (untrained, frozen)',
        'color_palette': 'viridis', 
        'eval_results_path': f'/om2/user/zaho/BrainBERT/eval_results_{split_type}/brainbert_randomly_initialized_keepall/',
        'pad_x': 1,
        'description': 'BrainBERT model with random initialization, frozen weights',
        'author': 'Andrii Zahorodnii',
        'organization': 'MIT',
        'organization_url': 'https://mit.edu'
    },
    {
        'name': 'BrainBERT (frozen; Wang et al. 2023)',
        'short_name': 'BrainBERT (frozen)',
        'color_palette': 'viridis', 
        'eval_results_path': f'/om2/user/zaho/BrainBERT/eval_results_{split_type}/brainbert_keepall/',
        'description': 'BrainBERT model with pretrained weights, frozen (Wang et al. 2023)',
        'author': 'Andrii Zahorodnii',
        'organization': 'MIT',
        'organization_url': 'https://mit.edu'
    },
    {
        'name': 'PopulationTransformer (Chau et al. 2024)',
        'short_name': 'PopulationTransformer',
        'color_palette': 'viridis', 
        'eval_results_path': f'/om2/user/zaho/PopTCameraReadyPrep/outputs/neuroprobe_popt_lite/eval_results_{split_type}/',
        'pad_x': 1,
        'description': 'PopulationTransformer model (Chau et al. 2024)',
        'author': 'Andrii Zahorodnii',
        'organization': 'MIT',
        'organization_url': 'https://mit.edu'
    },
]

### DEFINE TASK NAME MAPPING ###

task_name_mapping = {
    'onset': 'Sentence Onset',
    'speech': 'Speech',
    'volume': 'Volume', 
    'delta_volume': 'Delta Volume',
    'pitch': 'Voice Pitch',

    'word_index': 'Word Position',
    'word_gap': 'Inter-word Gap',
    'gpt2_surprisal': 'GPT-2 Surprisal',
    'word_head_pos': 'Head Word Position',
    'word_part_speech': 'Part of Speech',

    'word_length': 'Word Length',
    'global_flow': 'Global Optical Flow',
    'local_flow': 'Local Optical Flow',
    'frame_brightness': 'Frame Brightness',
    'face_num': 'Number of Faces',
    
    # 'delta_pitch': 'Delta Pitch',
    # 'speaker': 'Speaker Identity',
    # 'global_flow_angle': 'Global Flow Angle',
    # 'local_flow_angle': 'Local Flow Angle',
}

subject_trials = neuroprobe_config.NEUROPROBE_LITE_SUBJECT_TRIALS
if split_type == 'DS_DM':
    subject_trials = [(s, t) for s, t in subject_trials if s != neuroprobe_config.DS_DM_TRAIN_SUBJECT_ID]

for model in models:
    if 'short_name' not in model:
        model['short_name'] = model['name'] # if short_name is not defined, use the full name

### DEFINE RESULT PARSING FUNCTIONS ###

def parse_results_for_leaderboard(model):
    """Parse results and return in leaderboard format for each task"""
    leaderboard_results = {}
    
    for task in task_name_mapping.keys():
        # Create the leaderboard format for this task
        task_result = {
            "model_name": model['name'],
            "description": model['description'],
            "author": model['author'],
            "organization": model['organization'],
            "organization_url": model['organization_url'],
            "timestamp": int(time.time()),
            "evaluation_results": {}
        }
        
        for subject_id, trial_id in subject_trials:
            session_key = f'btbank{subject_id}_{trial_id}'
            filename = model['eval_results_path'] + f'population_btbank{subject_id}_{trial_id}_{task}.json'
            
            if not os.path.exists(filename):
                print(f"Warning: File {filename} not found, skipping...")
                continue

            try:
                with open(filename, 'r') as json_file:
                    data = json.load(json_file)
            except json.JSONDecodeError as e:
                # Try parsing without the last character in case of truncation
                try:
                    with open(filename, 'r') as json_file:
                        content = json_file.read()
                        if content:
                            # Remove last character and try parsing again
                            data = json.loads(content[:-1])
                            # Save the corrected content back to the file
                            with open(filename, 'w') as json_file:
                                json.dump(data, json_file, indent=4)
                            print(f"Saved corrected JSON back to {filename}")
                        else:
                            print(f"Warning: Empty file {filename}, skipping...")
                            continue
                except (json.JSONDecodeError, Exception) as e2:
                    print(f"Warning: Still invalid JSON in file {filename} after removing last character: {e2}")
                    continue
            
            # Extract the evaluation results for this session
            session_data = data['evaluation_results'][session_key]['population']
            
            # Use the appropriate time window
            if 'one_second_after_onset' in session_data:
                time_window_data = session_data['one_second_after_onset']
            else:
                time_window_data = session_data['whole_window']  # for BrainBERT only
            
            # Format the results for leaderboard
            task_result["evaluation_results"][session_key] = {
                "population": {
                    "one_second_after_onset": {
                        "time_bin_start": time_window_data.get('time_bin_start', 0.0),
                        "time_bin_end": time_window_data.get('time_bin_end', 1.0),
                        "folds": time_window_data['folds']
                    }
                }
            }
        
        leaderboard_results[task] = task_result
    
    return leaderboard_results

def save_leaderboard_results(model, leaderboard_results):
    """Save results in leaderboard format"""
    # Create model directory
    leaderboard_name = model['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('+', '').replace('-', '').replace('*', '').replace('/', '').replace('\\', '').replace('|', '').replace('"', '').replace("'", '').replace('`', '').replace('~', '').replace('!', '').replace('@', '').replace('#', '').replace('$', '').replace('%', '').replace('^', '').replace('&', '').replace('(', '').replace(')', '').replace('{', '').replace('}', '').replace('[', '').replace(']', '').replace(':', '').replace(';', '').replace(',', '').replace('.', '').replace('<', '').replace('>', '').replace('?', '').replace(' ', '_')
    leaderboard_name = leaderboard_name.replace('__', '_')
    timestamp = int(time.time())
    date_str = time.strftime("%d_%m_%Y", time.localtime(timestamp))
    model_dir = f"leaderboard/{leaderboard_name}_Andrii_Zahorodnii_{date_str}"
    split_dir = f"{model_dir}/{split_type_mapping[split_type]}"
    
    os.makedirs(split_dir, exist_ok=True)
    
    # Save each task result as a separate JSON file
    for task, task_result in leaderboard_results.items():
        filename = f"{split_dir}/population_{task}.json"
        with open(filename, 'w') as f:
            json.dump(task_result, f, indent=2)
        print(f"Saved {filename}")
    
    # Create metadata.json
    metadata = {
        "model_name": model['name'],
        "description": model['description'],
        "author": model['author'],
        "organization": model['organization'],
        "organization_url": model['organization_url'],
        "timestamp": timestamp
    }
    
    metadata_filename = f"{model_dir}/metadata.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {metadata_filename}")
    
    # Create PUBLICATION.bib placeholder
    bib_content = f"""TO BE FILLED"""
    
    bib_filename = f"{model_dir}/PUBLICATION.bib"
    with open(bib_filename, 'w') as f:
        f.write(bib_content)
    print(f"Saved {bib_filename}")
    
    # Create ATTESTATION.txt
    attestation_content = """TO BE FILLED"""
    
    attestation_filename = f"{model_dir}/ATTESTATION.txt"
    with open(attestation_filename, 'w') as f:
        f.write(attestation_content)
    print(f"Saved {attestation_filename}")

# Process each model
for model in models:
    print(f"Processing model: {model['name']}")
    leaderboard_results = parse_results_for_leaderboard(model)
    save_leaderboard_results(model, leaderboard_results)
    print(f"Completed processing for {model['name']}\n")

print(f"All models processed successfully for split type: {split_type}")
print(f"Results saved in leaderboard format under: leaderboard/")
