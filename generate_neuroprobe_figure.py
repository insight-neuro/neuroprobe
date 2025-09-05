import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import glob, math
import pandas as pd
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

separate_overall_yscale = True # Whether to have the "Task Mean" figure panel have a 0.5-0.6 ylim instead of 0.5-0.9 (used to better see the difference between models)
n_fig_legend_cols = 3

### DEFINE MODELS ###

models = [
    {
        'name': 'Linear',
        'color_palette': 'viridis',
        'eval_results_path': f'/om2/user/zaho/neuroprobe/data/eval_results_lite_{split_type}/linear_voltage/'
    },
    {
        'name': 'Linear (spectrogram)',
        'color_palette': 'viridis', 
        'eval_results_path': f'/om2/user/zaho/neuroprobe/data/eval_results_lite_{split_type}/linear_stft_abs_nperseg512_poverlap0.75_maxfreq150/'
    },
    # {
    #     'name': 'Linear (spectrogram)',
    #     'color_palette': 'viridis',
    #     'eval_results_path': f'/om2/user/zaho/btbench/eval_results_lite_{split_type}/linear_fft_abs/'
    # },
    # {
    #     'name': 'BrainBERT',
    #     'color_palette': 'plasma',
    #     'eval_results_path': f'/om2/user/zaho/BrainBERT/eval_results_lite_{split_type}/brainbert_frozen_mean_granularity_{-1}/'
    # },
    # # {
    # #     'name': 'PopT (frozen)',
    # #     'color_palette': 'magma',
    # #     'eval_results_path': f'/om2/user/zaho/btbench/eval_results_popt/population_frozen_{split_type}_results.csv'
    # # },
    # {
    #     'name': 'PopT',
    #     'color_palette': 'magma',
    #     'eval_results_path': f'/om2/user/zaho/btbench/eval_results_popt/popt_{split_type}_results.csv'
    # }
]

### DEFINE TASK NAME MAPPING ###

task_name_mapping = {
    'onset': 'Sentence Onset',
    'speech': 'Speech',
    'volume': 'Volume', 
    'pitch': 'Voice Pitch',
    'speaker': 'Speaker Identity',
    'delta_volume': 'Delta Volume',
    'delta_pitch': 'Delta Pitch',
    'gpt2_surprisal': 'GPT-2 Surprisal',
    'word_length': 'Word Length',
    'word_gap': 'Inter-word Gap',
    'word_index': 'Word Position',
    'word_head_pos': 'Head Word Position',
    'word_part_speech': 'Part of Speech',
    'frame_brightness': 'Frame Brightness',
    'global_flow': 'Global Optical Flow',
    'local_flow': 'Local Optical Flow',
    'global_flow_angle': 'Global Flow Angle',
    'local_flow_angle': 'Local Flow Angle',
    'face_num': 'Number of Faces',
}

subject_trials = neuroprobe_config.NEUROPROBE_LITE_SUBJECT_TRIALS
if split_type == 'DS_DM':
    subject_trials = [(s, t) for s, t in subject_trials if s != neuroprobe_config.DS_DM_TRAIN_SUBJECT_ID]

### DEFINE RESULT PARSING FUNCTIONS ###

performance_data = {}
for task in task_name_mapping.keys():
    performance_data[task] = {}
    for model in models:
        performance_data[task][model['name']] = {}

def parse_results_default(model):
    for task in task_name_mapping.keys():
        subject_trial_means = []
        for subject_id, trial_id in subject_trials:
            filename = model['eval_results_path'] + f'population_btbank{subject_id}_{trial_id}_{task}.json'
            if not os.path.exists(filename):
                print(f"Warning: File {filename} not found, skipping...")
                continue

            with open(filename, 'r') as json_file:
                data = json.load(json_file)
            
            if 'one_second_after_onset' in data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']: # XXX remove this later, have a unified interface for all models
                data = data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']['one_second_after_onset'] 
            else:
                data = data['evaluation_results'][f'btbank{subject_id}_{trial_id}']['population']['whole_window'] # for BrainBERT only
            value = np.nanmean([fold_result['test_roc_auc'] for fold_result in data['folds']])
            subject_trial_means.append(value)

        performance_data[task][model['name']] = {
            'mean': np.mean(subject_trial_means),
            'sem': np.std(subject_trial_means) / np.sqrt(len(subject_trial_means))
        }
for model in models:
    model['parse_results_function'] = parse_results_default

def parse_results_hara(model):
    for task in task_name_mapping.keys():
        subject_trial_means = []
        for subject_id, trial_id in subject_trials:
            pattern = f'/om2/user/hmor/btbench/eval_results_ds_dt_lite_desikan_killiany/DS-DT-FixedTrain-Lite_{task}_test_S{subject_id}T{trial_id}_*.json'
            matching_files = glob.glob(pattern)
            if matching_files:
                filename = matching_files[0]  # Take the first matching file
            else:
                print(f"Warning: No matching file found for pattern {pattern}, skipping...")
            
            with open(filename, 'r') as json_file:
                data = json.load(json_file)
            data = data['final_auroc']
            subject_trial_means.append(data)
        performance_data[task][model['name']] = {
            'mean': np.mean(subject_trial_means),
            'sem': np.std(subject_trial_means) / np.sqrt(len(subject_trial_means))
        }
if split_type == 'DS_DM': # XXX remove this later, have a unified interface for all models
    models[0]['parse_results_function'] = parse_results_hara

def parse_results_popt(model):
    # Read the CSV file
    popt_data = pd.read_csv(model['eval_results_path'])
    # Group by subject_id, trial_id, and task_name to calculate mean across folds
    for task in task_name_mapping.keys():
        subject_trial_means = []
        
        for subject_id, trial_id in subject_trials:
            # Filter data for current subject, trial, and task
            task_data = popt_data[(popt_data['subject_id'] == subject_id) & 
                                (popt_data['trial_id'] == trial_id) & 
                                ((popt_data['task_name'] == task) | (popt_data['task_name'] == task + '_frozen_True'))]
            
            if not task_data.empty:
                # Calculate mean ROC AUC across folds
                value = task_data['test_roc_auc'].mean()
                subject_trial_means.append(value)
            else:
                print(f"Warning: No data found for subject {subject_id}, trial {trial_id}, task {task} in POPT results ({model['eval_results_path']})")
        
        if subject_trial_means:
            performance_data[task][model['name']] = {
                'mean': np.mean(subject_trial_means),
                'sem': np.std(subject_trial_means) / np.sqrt(len(subject_trial_means))
            }
        else:
            performance_data[task][model['name']] = {
                'mean': np.nan,
                'sem': np.nan
            }
for model in models:
    if 'PopT' in model['name']:
        model['parse_results_function'] = parse_results_popt

for model in models:
    model['parse_results_function'](model)
    
### CALCULATE OVERALL PERFORMANCE ###

overall_performance = {}
for model in models:
    means = [performance_data[task][model['name']]['mean'] for task in task_name_mapping.keys()]
    sems = [performance_data[task][model['name']]['sem'] for task in task_name_mapping.keys()]
    overall_performance[model['name']] = {
        'mean': np.nanmean(means),
        'sem': np.sqrt(np.sum(np.array(sems)**2)) / len(sems)  # Combined SEM
    }

### PREPARING FOR PLOTTING ###

# Add Arial font
import matplotlib.font_manager as fm
font_path = 'assets/font_arial.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 12})

# Assign colors to models based on color palette
color_palette_ids = {}
for model in models:
    if model['color_palette'] not in color_palette_ids: color_palette_ids[model['color_palette']] = 0
    model['color_palette_id'] = color_palette_ids[model['color_palette']]
    color_palette_ids[model['color_palette']] += 1
for model in models:
    model['color'] = sns.color_palette(model['color_palette'], color_palette_ids[model['color_palette']])[model['color_palette_id']]

### PLOT STUFF ###

# Create figure with 4x5 grid - reduced size
n_cols = 5
n_rows = math.ceil((len(task_name_mapping)+1)/n_cols)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(8/5*n_cols, 6/4*n_rows+.6 * len(models) / n_fig_legend_cols/3/2))

# Flatten axs for easier iteration
axs_flat = axs.flatten()

# Bar width
bar_width = 0.2

# Plot overall performance in first axis
first_ax = axs_flat[0]
for i, model in enumerate(models):
    perf = overall_performance[model['name']]
    first_ax.bar(i * bar_width, perf['mean'], bar_width,
                yerr=perf['sem'],
                color=model['color'],
                capsize=6)

first_ax.set_title('Task Mean', fontsize=12, pad=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
if metric == 'accuracy':
    first_ax.set_ylim(0.2, 1.0)
else:
    if separate_overall_yscale:
        first_ax.set_ylim(0.4925, 0.75)
        first_ax.set_yticks([0.5, 0.6, 0.7])
    else:
        first_ax.set_ylim(0.48, 0.95)
        first_ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
first_ax.set_xticks([])
first_ax.set_ylabel(metric)
first_ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
first_ax.spines['top'].set_visible(False)
first_ax.spines['right'].set_visible(False)
first_ax.tick_params(axis='y')

# Plot counter - start from 1 for remaining plots
plot_idx = 1

for task, chance_level in task_name_mapping.items():
    ax = axs_flat[plot_idx]
    
    # Plot bars for each model
    x = np.arange(len(models))
    for i, model in enumerate(models):
        perf = performance_data[task][model['name']]
        ax.bar(i * bar_width, perf['mean'], bar_width,
                yerr=perf['sem'], 
                color=model['color'],
                capsize=6)
    
    # Customize plot
    ax.set_title(task_name_mapping[task], fontsize=12, pad=10)
    if metric == 'accuracy':
        ax.set_ylim(0.2, 1.0)
    else:
        ax.set_ylim(0.48, 0.95)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xticks([])
    if (plot_idx % 5 == 0):  # Left-most plots
        ax.set_ylabel(metric)

    # Add horizontal line at chance level
    if metric == 'AUROC':
        chance_level = 0.5
    ax.axhline(y=chance_level, color='black', linestyle='--', alpha=0.5)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make tick labels smaller
    ax.tick_params(axis='y')
    
    plot_idx += 1

# Create a proxy artist for the chance line with the correct style
chance_line = plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.5)

# Add legend at the bottom with custom handles
handles = [plt.Rectangle((0,0),1,1, color=model['color']) for model in models]
handles.append(chance_line)
fig.legend(handles, [model['name'] for model in models] + ["Chance"],
            loc='lower center', 
            bbox_to_anchor=(0.5, 0.1),
            ncol=n_fig_legend_cols,
            frameon=False)

# Adjust layout with space at the bottom for legend
if (len(models)//3 == 2): rect_y = 0.2
else: rect_y = 0.15
plt.tight_layout(rect=[0, rect_y, 1, 1], w_pad=0.4)

# Save figure
save_path = f'data/figures/neuroprobe_eval_lite_{split_type}.pdf'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'Saved figure to {save_path}')
plt.close()
