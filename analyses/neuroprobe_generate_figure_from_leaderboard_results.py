import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import math
import glob

### PARSE ARGUMENTS ###

import argparse
parser = argparse.ArgumentParser(description='Create performance figure for Neuroprobe leaderboard submissions')
parser.add_argument('--split_type', type=str, default='CrossSession',
                    help='Split type to use (WithinSession or CrossSession or CrossSubject)')
parser.add_argument('--leaderboard_dir', type=str, default='leaderboard',
                    help='Path to the leaderboard directory containing model submission folders')
parser.add_argument('--output_dir', type=str, default='analyses/figures',
                    help='Directory to write figures, tables, and JSON output to')
args = parser.parse_args()
split_type = args.split_type
leaderboard_dir = args.leaderboard_dir
output_dir = args.output_dir

# Map the CLI split-type names to the folder names used inside each leaderboard submission.
split_type_to_folder = {
    'WithinSession': 'Within-Session',
    'CrossSession':  'Cross-Session',
    'CrossSubject':  'Cross-Subject',
}
assert split_type in split_type_to_folder, f'Unknown split_type {split_type!r}; expected one of {list(split_type_to_folder)}'
split_folder = split_type_to_folder[split_type]

metric = 'AUROC'

separate_overall_yscale = True
overall_axis_ylim = (0.4925, 0.71) if separate_overall_yscale else (0.48, 0.95)
other_axis_ylim = (0.48, 0.95)

figure_size_multiplier = 1.8
first_ax_n_cols = 2
n_fig_legend_cols = 1

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
}

### DEFINE MODEL DISPLAY NAME ALIASES ###

model_display_name_aliases = {
    'MLP_Laplacian_rereferencing_spectrogram': 'MLP (Laplacian re-referencing + spectrogram)',
    'CNN_Laplacian_rereferencing_spectrogram': 'CNN (Laplacian re-referencing + spectrogram)',
    'DIVER-1_0.1s_tiny_frozen': 'DIVER-1 (0.1s, tiny, frozen)',
    'DIVER-1_0.1s_tiny': 'DIVER-1 (0.1s, tiny)',
}

def get_display_name(model_name):
    return model_display_name_aliases.get(model_name, model_name)

### DISCOVER MODELS FROM LEADERBOARD ###

def discover_models(leaderboard_dir, split_folder):
    """Return list of model dicts for every leaderboard submission that has
    a `<split_folder>` directory. Each dict carries the canonical
    `model_name` from metadata.json plus the path to the split's results."""
    models = []
    for submission_path in sorted(glob.glob(os.path.join(leaderboard_dir, '*'))):
        if not os.path.isdir(submission_path):
            continue
        split_path = os.path.join(submission_path, split_folder)
        if not os.path.isdir(split_path):
            continue
        metadata_path = os.path.join(submission_path, 'metadata.json')
        if not os.path.exists(metadata_path):
            print(f"Warning: {metadata_path} missing, skipping {submission_path}")
            continue
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        model_name = metadata.get('model_name') or os.path.basename(submission_path)
        models.append({
            'name': model_name,
            'short_name': metadata.get('short_name', model_name),
            'eval_results_path': split_path,
            'submission_folder': os.path.basename(submission_path),
        })
    # Sort alphabetically by name for stable ordering across runs.
    models.sort(key=lambda m: m['name'].lower())
    return models

models = discover_models(leaderboard_dir, split_folder)
if not models:
    raise SystemExit(f'No leaderboard submissions found with a {split_folder!r} subfolder under {leaderboard_dir!r}.')
print(f'Found {len(models)} leaderboard submissions for split {split_type} ({split_folder}).')

### DEFINE RESULT PARSING FUNCTIONS ###

all_tasks = ['Overall'] + list(task_name_mapping.keys())

performance_data = {task: {model['name']: {} for model in models} for task in all_tasks}

def _extract_population_block(subject_trial_entry):
    """Pick the population sub-block — prefer 'one_second_after_onset',
    fall back to 'whole_window' (used by some BrainBERT submissions)."""
    population = subject_trial_entry['population']
    if 'one_second_after_onset' in population:
        return population['one_second_after_onset']
    if 'whole_window' in population:
        return population['whole_window']
    # As a last resort, take the first key.
    return next(iter(population.values()))

def parse_results_leaderboard(model):
    for task in task_name_mapping.keys():
        filename = os.path.join(model['eval_results_path'], f'population_{task}.json')
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found, skipping...")
            performance_data[task][model['name']] = {'mean': np.nan, 'sem': np.nan}
            continue
        try:
            with open(filename, 'r') as json_file:
                data = json.load(json_file)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in file {filename}: {e}")
            performance_data[task][model['name']] = {'mean': np.nan, 'sem': np.nan}
            continue

        evaluation_results = data.get('evaluation_results', {})
        subject_trial_means = []
        for subject_trial_key, entry in evaluation_results.items():
            try:
                block = _extract_population_block(entry)
                value = np.nanmean([fold_result['test_roc_auc'] for fold_result in block['folds']])
                if not np.isnan(value):
                    subject_trial_means.append(value)
            except (KeyError, TypeError) as e:
                print(f"Warning: Could not parse {subject_trial_key} in {filename}: {e}")
                continue

        if subject_trial_means:
            performance_data[task][model['name']] = {
                'mean': float(np.mean(subject_trial_means)),
                'sem': float(np.std(subject_trial_means) / np.sqrt(len(subject_trial_means))),
            }
        else:
            performance_data[task][model['name']] = {'mean': np.nan, 'sem': np.nan}

for model in models:
    parse_results_leaderboard(model)

### CALCULATE OVERALL PERFORMANCE ###

for model in models:
    means = [performance_data[task][model['name']].get('mean', np.nan) for task in task_name_mapping.keys()]
    sems = [performance_data[task][model['name']].get('sem', np.nan) for task in task_name_mapping.keys()]
    means_arr = np.array(means, dtype=float)
    sems_arr = np.array(sems, dtype=float)
    valid_count = int(np.sum(~np.isnan(sems_arr)))
    overall_mean = float(np.nanmean(means_arr)) if np.any(~np.isnan(means_arr)) else np.nan
    if valid_count > 0:
        overall_sem = float(np.sqrt(np.nansum(sems_arr ** 2)) / valid_count)
    else:
        overall_sem = np.nan
    performance_data['Overall'][model['name']] = {'mean': overall_mean, 'sem': overall_sem}

### SORT MODELS BY OVERALL PERFORMANCE (DESCENDING) ###

# Models without any overall mean (all NaN) sink to the bottom.
def _sort_key(model):
    mean = performance_data['Overall'][model['name']].get('mean')
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return (1, 0.0)
    return (0, -float(mean))

models.sort(key=_sort_key)

### PREPARING FOR PLOTTING ###

import matplotlib.font_manager as fm
font_path = 'analyses/font_arial.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 12})

# Use a perceptually distinct palette since there can be many leaderboard models.
palette = sns.color_palette('husl', n_colors=len(models))
for i, model in enumerate(models):
    model['color'] = palette[i]
    model['x_pos'] = i

### PLOT STUFF ###

import matplotlib.gridspec as gridspec

n_cols = 5
overall_height = 1.2
margin_height = -0.05
task_rows = math.ceil(len(task_name_mapping) / n_cols)

height_ratios = [overall_height, margin_height] + [1.0] * task_rows
n_rows = len(height_ratios)

base_width = figure_size_multiplier * 8 / 5 * n_cols
base_height = figure_size_multiplier * 6 / 4 * n_rows

fig = plt.figure(figsize=(base_width, base_height))
gs = gridspec.GridSpec(n_rows, n_cols, height_ratios=height_ratios, hspace=0.3, wspace=0.2)

# Bar width scales gently with the number of models so they stay readable.
bar_width = max(0.08, min(0.2, 2.5 / max(len(models), 1)))

# Overall (Task Mean) panel
first_ax = fig.add_subplot(gs[0, 0:first_ax_n_cols])
for model in models:
    perf = performance_data['Overall'][model['name']]
    if np.isnan(perf['mean']):
        continue
    first_ax.bar(model['x_pos'] * bar_width, perf['mean'], bar_width,
                 yerr=perf['sem'] if not np.isnan(perf['sem']) else None,
                 color=model['color'],
                 capsize=4)

first_ax.set_title('Task Mean', fontsize=12, pad=10,
                   bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
first_ax.set_ylim(overall_axis_ylim)
first_ax.set_yticks(np.arange(0.5, overall_axis_ylim[1], 0.1))
first_ax.set_xticks([])
first_ax.set_ylabel(metric)
first_ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
first_ax.spines['top'].set_visible(False)
first_ax.spines['right'].set_visible(False)
first_ax.tick_params(axis='y')

# Legend panel
legend_ax = fig.add_subplot(gs[0, first_ax_n_cols:])
legend_ax.axis('off')
handles = [plt.Rectangle((0, 0), 1, 1, color=model['color']) for model in models]
chance_line = plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.5)
handles.append(chance_line)
legend_ax.legend(handles, [get_display_name(model['name']) for model in models] + ["Chance"],
                 loc='center left',
                 ncol=n_fig_legend_cols,
                 frameon=False,
                 fontsize=9)

# Per-task panels
plot_idx = 0
for task in task_name_mapping:
    row = plot_idx // n_cols + 2
    col = plot_idx % n_cols
    ax = fig.add_subplot(gs[row, col])

    for model in models:
        perf = performance_data[task][model['name']]
        if np.isnan(perf['mean']):
            continue
        ax.bar(model['x_pos'] * bar_width, perf['mean'], bar_width,
               yerr=perf['sem'] if not np.isnan(perf['sem']) else None,
               color=model['color'],
               capsize=6 / (models[-1]['x_pos'] + 1) * 10)

    ax.set_title(task_name_mapping[task], fontsize=12, pad=10)
    ax.set_ylim(other_axis_ylim)
    ax.set_yticks(np.arange(0.5, other_axis_ylim[1], 0.1))
    ax.set_xticks([])
    if col == 0:
        ax.set_ylabel('AUROC')

    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y')

    plot_idx += 1

plt.tight_layout()

os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, f'neuroprobe_eval_leaderboard_{split_type}.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'Saved figure to {save_path}')
jpg_path = os.path.join(output_dir, f'neuroprobe_eval_leaderboard_{split_type}.jpg')
plt.savefig(jpg_path, dpi=300, bbox_inches='tight')
print(f'Saved figure to {jpg_path}')
plt.close()

### SAVE PERFORMANCE DATA ###

filename = os.path.join(output_dir, f'neuroprobe_eval_leaderboard_{split_type}.json')
with open(filename, 'w') as f:
    json.dump(performance_data, f, indent=2)
print(f'Saved performance data to {filename}')

### GENERATE LATEX TABLE ###

latex_lines = []
latex_lines.append("\\begin{table}[h]")
latex_lines.append("\\centering")

task_display_mapping = {'Overall': 'Overall'}
task_display_mapping.update(task_name_mapping)

n_chunks = math.ceil(len(all_tasks) / 4)

def _fmt(perf):
    if perf is None or np.isnan(perf.get('mean', np.nan)):
        return "--"
    sem = perf.get('sem', np.nan)
    sem_str = f"{sem:.3f}" if not np.isnan(sem) else "--"
    return f"{perf['mean']:.3f} $\\pm$ {sem_str}"

for chunk in range(n_chunks):
    start_idx = chunk * 4
    end_idx = min((chunk + 1) * 4, len(all_tasks))
    chunk_tasks = all_tasks[start_idx:end_idx]

    latex_lines.append("\\begin{tabular}{l" + "c" * len(chunk_tasks) + "}")
    latex_lines.append("\\hline")

    header = "Model"
    for task in chunk_tasks:
        header += f" & {task_display_mapping[task]}"
    latex_lines.append(header + " \\\\")
    latex_lines.append("\\hline")

    # Best model per task (ignoring NaNs)
    best_by_task = {}
    for task in chunk_tasks:
        candidates = [(m['name'], performance_data[task][m['name']].get('mean', np.nan)) for m in models]
        candidates = [(name, mean) for name, mean in candidates if not np.isnan(mean)]
        if candidates:
            best_by_task[task] = max(candidates, key=lambda kv: kv[1])[0]

    for model in models:
        row_cells = [model['short_name']]
        for task in chunk_tasks:
            cell = _fmt(performance_data[task][model['name']])
            if best_by_task.get(task) == model['name'] and cell != "--":
                cell = f"\\textbf{{{cell}}}"
            row_cells.append(cell)
        latex_lines.append(" & ".join(row_cells) + " \\\\")

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")

    if chunk < n_chunks - 1:
        latex_lines.append("\\hspace{1em}")

latex_lines.append("\\caption{Leaderboard performance comparison across tasks (mean $\\pm$ SEM). "
                   "Best performing model for each task is shown in bold.}")
latex_lines.append("\\label{tab:leaderboard_performance_" + split_type + "}")
latex_lines.append("\\end{table}")

latex_filename = os.path.join(output_dir, f'neuroprobe_eval_leaderboard_{split_type}.tex')
with open(latex_filename, 'w') as f:
    f.write('\n'.join(latex_lines))
print(f'Saved LaTeX table to {latex_filename}')
