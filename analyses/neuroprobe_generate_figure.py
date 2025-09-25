import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import math
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
overall_axis_ylim = (0.4925, 0.71) if separate_overall_yscale else (0.48, 0.95)
other_axis_ylim = (0.48, 0.95)

figure_size_multiplier = 1.8

first_ax_n_cols = 2
n_fig_legend_cols = 1

### DEFINE MODELS ###

# assert split_type == 'SS_DM', 'Split type must be SS_DM'

models = [
    {
        'name': 'Linear (raw voltage)',
        'short_name': 'Linear (voltage)',
        'color_palette': 'viridis',
        'eval_results_path': f'/om2/user/zaho/neuroprobe/data/eval_results_lite_{split_type}/linear_voltage/'
    },
    {
        'name': 'Linear (spectrogram)',
        'short_name': 'Linear (spectrogram)',
        'color_palette': 'viridis', 
        'eval_results_path': f'/om2/user/zaho/neuroprobe/data/eval_results_lite_{split_type}/linear_stft_abs_nperseg512_poverlap0.75_maxfreq150/'
    },
    {
        'name': 'Linear (Laplacian re-referencing + spectrogram)',
        'short_name': 'Linear (Laplacian+spectrogram)',
        'color_palette': 'viridis', 
        'eval_results_path': f'/om2/user/zaho/neuroprobe/data/eval_results_lite_{split_type}/linear_laplacian-stft_abs_nperseg512_poverlap0.75_maxfreq150/'
    },
    {
        'name': 'BrainBERT (untrained, frozen)',
        'color_palette': 'viridis', 
        'eval_results_path': f'/om2/user/zaho/BrainBERT/eval_results_{split_type}/brainbert_randomly_initialized_keepall/',
        'pad_x': 1,
    },
    {
        'name': 'BrainBERT (frozen; Wang et al. 2023)',
        'short_name': 'BrainBERT (frozen)',
        'color_palette': 'viridis', 
        'eval_results_path': f'/om2/user/zaho/BrainBERT/eval_results_{split_type}/brainbert_keepall/',
    },
    {
        'name': 'PopulationTransformer (Chau et al. 2024)',
        'short_name': 'PopulationTransformer',
        'color_palette': 'viridis', 
        'eval_results_path': f'/om2/user/zaho/PopTCameraReadyPrep/outputs/neuroprobe_popt_lite/eval_results_{split_type}/',
        'pad_x': 1,
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

            try:
                with open(filename, 'r') as json_file:
                    data = json.load(json_file)
            except json.JSONDecodeError as e:
                # print(f"Warning: Invalid JSON in file {filename}: {e}")
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

# Assign model x-positions
current_x_pos = 0
for i, model in enumerate(models):
    if model.get('pad_x', 0): current_x_pos += model['pad_x']
    model['x_pos'] = current_x_pos
    current_x_pos += 1

### PLOT STUFF ###

# Create figure with modified grid layout using GridSpec
import matplotlib.gridspec as gridspec

n_cols = 5
overall_height = 1.2  # Height of overall axis
margin_height = -0.05   # Margin between overall and task plots
task_rows = math.ceil(len(task_name_mapping)/n_cols)

# Create height ratios: [overall_height, margin_height, task_row_1, task_row_2, ...]
height_ratios = [overall_height, margin_height] + [1.0] * task_rows
n_rows = len(height_ratios)

# Base figure size (no extra legend space needed since legend is in plot area)
base_width = figure_size_multiplier * 8/5 * n_cols
base_height = figure_size_multiplier * 6/4 * n_rows

fig = plt.figure(figsize=(base_width, base_height))
gs = gridspec.GridSpec(n_rows, n_cols, height_ratios=height_ratios, hspace=0.3, wspace=0.2)

# Bar width
bar_width = 0.2

# Plot overall performance in first column only
first_ax = fig.add_subplot(gs[0, 0:first_ax_n_cols])
for i, model in enumerate(models):
    perf = overall_performance[model['name']]
    first_ax.bar(model['x_pos']*bar_width, perf['mean'], bar_width,
                yerr=perf['sem'],
                color=model['color'],
                capsize=6)

first_ax.set_title('Task Mean', fontsize=12, pad=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
first_ax.set_ylim(overall_axis_ylim)
first_ax.set_yticks(np.arange(0.5, overall_axis_ylim[1], 0.1))
first_ax.set_xticks([])
first_ax.set_ylabel(metric)
first_ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
first_ax.spines['top'].set_visible(False)
first_ax.spines['right'].set_visible(False)
first_ax.tick_params(axis='y')

# Create legend in the remaining 4/5th of the first row
legend_ax = fig.add_subplot(gs[0, first_ax_n_cols:])
legend_ax.axis('off')  # Hide the axis

# Create proxy artists for the legend
handles = [plt.Rectangle((0,0),1,1, color=model['color']) for model in models]
chance_line = plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.5)
handles.append(chance_line)

# Add legend to the legend axis
legend_ax.legend(handles, [model['name'] for model in models] + ["Chance"],
                loc='center left', 
                ncol=n_fig_legend_cols,
                frameon=False)

# Plot counter - start from 0 for task plots in remaining rows
plot_idx = 0

for task, chance_level in task_name_mapping.items():
    # Calculate row and column for current task (starting after overall axis and margin)
    row = plot_idx // n_cols + 2  # Start from row 2 (0=overall, 1=margin, 2+=tasks)
    col = plot_idx % n_cols
    ax = fig.add_subplot(gs[row, col])
    
    # Plot bars for each model
    x = np.arange(len(models))
    for i, model in enumerate(models):
        perf = performance_data[task][model['name']]
        ax.bar(model['x_pos']*bar_width, perf['mean'], bar_width,
                yerr=perf['sem'], 
                color=model['color'],
                capsize=6/(models[-1]['x_pos']+1) * 10)
    
    # Customize plot
    ax.set_title(task_name_mapping[task], fontsize=12, pad=10)
    ax.set_ylim(other_axis_ylim)
    ax.set_yticks(np.arange(0.5, other_axis_ylim[1], 0.1))
    ax.set_xticks([])
    if col == 0:  # Left-most plots
        ax.set_ylabel("AUROC")

    # Add horizontal line at chance level
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make tick labels smaller
    ax.tick_params(axis='y')
    
    plot_idx += 1

# Use tight_layout without legend padding since legend is now in the plot area
plt.tight_layout()

# Save figure
save_path = f'analyses/figures/neuroprobe_eval_lite_{split_type}.pdf'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'Saved figure to {save_path}')
plt.close()

### SAVE PERFORMANCE DATA ###

performance_data['overall'] = overall_performance
# print(performance_data)
filename = f'analyses/figures/neuroprobe_eval_lite_{split_type}.json' 
with open(filename, 'w') as f:
    json.dump(performance_data, f)
print(f'Saved performance data to {filename}')

### GENERATE LATEX TABLE ###
# Create LaTeX table
latex_lines = []
latex_lines.append("\\begin{table}[h]")
latex_lines.append("\\centering")

# Calculate number of chunks needed - changed from 5 to 4 columns per chunk
n_columns = len(task_name_mapping) + 1  # +1 for Overall column
n_chunks = math.ceil(n_columns / 4)  # Changed from 5 to 4

for chunk in range(n_chunks):
    # Calculate start and end indices for this chunk
    start_idx = chunk * 4  # Changed from 5 to 4
    end_idx = min((chunk + 1) * 4, n_columns)  # Changed from 5 to 4
    
    # Create tabular environment for this chunk
    latex_lines.append("\\begin{tabular}{l" + "c" * (end_idx - start_idx) + "}")
    latex_lines.append("\\hline")
    
    # Header row
    header = "Model"
    if chunk == 0:
        header += " & Overall"
    for i in range(start_idx, end_idx):
        if i == 0:  # Skip Overall as it's already added
            continue
        task = list(task_name_mapping.keys())[i-1]
        header += f" & {task_name_mapping[task]}"
    latex_lines.append(header + " \\\\")
    latex_lines.append("\\hline")

    # For each model
    for model in models:
        row = [model['short_name']]
        
        # Overall performance (only in first chunk)
        if chunk == 0:
            perf = overall_performance[model['name']]
            row.append(f"{perf['mean']:.3f} $\\pm$ {perf['sem']:.3f}")
        
        # Task performances for this chunk
        for i in range(start_idx, end_idx):
            if i == 0:  # Skip Overall as it's already added
                continue
            task = list(task_name_mapping.keys())[i-1]
            perf = performance_data[task][model['name']]
            row.append(f"{perf['mean']:.3f} $\\pm$ {perf['sem']:.3f}")
        
        # Find best performing model for each column
        formatted_row = []
        for i, value in enumerate(row):
            if i == 0:  # Model name
                formatted_row.append(value)
            else:
                # Determine if this is the best model for this column
                if chunk == 0 and i == 1:  # Overall column
                    best_model = max([m['name'] for m in models], key=lambda m: overall_performance[m]['mean'])
                else:
                    task_idx = start_idx + i - (2 if chunk == 0 else 1)
                    task = list(task_name_mapping.keys())[task_idx-1]
                    best_model = max([m['name'] for m in models], key=lambda m: performance_data[task][m]['mean'])
                
                if model['name'] == best_model:
                    formatted_row.append(f"\\textbf{{{value}}}")
                else:
                    formatted_row.append(value)
        
        latex_lines.append(" & ".join(formatted_row) + " \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    
    # Add some space between chunks if not the last chunk
    if chunk < n_chunks - 1:
        latex_lines.append("\\hspace{1em}")

latex_lines.append("\\caption{Performance comparison across tasks (mean $\\pm$ SEM). Best performing model for each task is shown in bold.}")
latex_lines.append("\\label{tab:performance_comparison}")
latex_lines.append("\\end{table}")

# Save LaTeX table to file
latex_filename = f'analyses/figures/neuroprobe_eval_lite_{split_type}.tex'
with open(latex_filename, 'w') as f:
    f.write('\n'.join(latex_lines))
print(f'Saved LaTeX table to {latex_filename}')
