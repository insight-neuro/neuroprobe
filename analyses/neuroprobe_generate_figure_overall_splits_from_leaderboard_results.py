import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
import argparse

### PARSE ARGUMENTS ###

parser = argparse.ArgumentParser(description='Plot overall (Task Mean) leaderboard performance across splits')
parser.add_argument('--input_dir', type=str, default='analyses/figures',
                    help='Directory holding the per-split JSON files written by '
                         'neuroprobe_generate_figure_from_leaderboard_results.py')
parser.add_argument('--output_dir', type=str, default='analyses/figures',
                    help='Directory to write the combined figure to')
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir

### LOAD DATA FROM ALL SPLIT TYPES ###

split_types = ['WithinSession', 'CrossSession', 'CrossSubject']
split_names = {
    'WithinSession': 'Within Session',
    'CrossSession':  'Cross Session',
    'CrossSubject':  'Cross Subject',
}

all_data = {}
for split_type in split_types:
    filename = os.path.join(input_dir, f'neuroprobe_eval_leaderboard_{split_type}.json')
    if not os.path.exists(filename):
        raise SystemExit(
            f'Missing input file: {filename}\n'
            f'Run neuroprobe_generate_figure_from_leaderboard_results.py --split_type {split_type} first.'
        )
    with open(filename, 'r') as f:
        all_data[split_type] = json.load(f)

### DEFINE MODEL DISPLAY NAME ALIASES ###

model_display_name_aliases = {
    'MLP_Laplacian_rereferencing_spectrogram': 'MLP (Laplacian re-referencing + spectrogram)',
    'CNN_Laplacian_rereferencing_spectrogram': 'CNN (Laplacian re-referencing + spectrogram)',
    'DIVER-1_0.1s_tiny_frozen': 'DIVER-1 (0.1s, tiny, frozen)',
    'DIVER-1_0.1s_tiny': 'DIVER-1 (0.1s, tiny)',
}

def get_display_name(model_name):
    return model_display_name_aliases.get(model_name, model_name)

### COLLECT MODELS (UNION ACROSS SPLITS) ###

# A model appears in the figure if it has Overall data for at least one split. We
# keep the union so the cross-subject-only or within-session-only submissions
# still surface in the panels where they actually have data.
model_names = set()
for split_type in split_types:
    model_names.update(all_data[split_type].get('Overall', {}).keys())

def _model_overall_score(name):
    """WithinSession Overall AUROC. Used to sort models so the best performers
    appear first. Models with no usable data sink to the bottom."""
    perf = all_data['WithinSession'].get('Overall', {}).get(name)
    if not perf:
        return (1, 0.0)
    mean = perf.get('mean')
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return (1, 0.0)
    return (0, -float(mean))

models = [{'name': name, 'short_name': name} for name in sorted(model_names, key=_model_overall_score)]

### PREPARING FOR PLOTTING ###

import matplotlib.font_manager as fm
font_path = 'analyses/font_arial.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 12})

# Distinct color per model so each appears the same color across the three split panels.
palette = sns.color_palette('husl', n_colors=len(models))
for i, model in enumerate(models):
    model['color'] = palette[i]
    model['x_pos'] = i

### PLOT FIGURE ###

figure_size_multiplier = 3
base_width = figure_size_multiplier * 3
base_height = figure_size_multiplier

fig, axes = plt.subplots(1, 3, figsize=(base_width, base_height))

bar_width = max(0.08, min(0.2, 2.5 / max(len(models), 1)))
axis_ylim = (0.48, 0.72)

for split_idx, split_type in enumerate(split_types):
    ax = axes[split_idx]
    overall = all_data[split_type].get('Overall', {})

    x = 0
    for model in models:
        perf = overall.get(model['name'])
        if perf is None:
            continue
        mean = perf.get('mean')
        sem = perf.get('sem')
        if mean is None or (isinstance(mean, float) and np.isnan(mean)):
            continue
        yerr = sem if (sem is not None and not (isinstance(sem, float) and np.isnan(sem))) else None
        ax.bar(x * bar_width, mean, bar_width,
               yerr=yerr,
               color=model['color'],
               capsize=4)
        x += 1

    ax.set_title(split_names[split_type], pad=15)
    ax.set_ylim(axis_ylim)
    ax.set_yticks(np.arange(0.5, axis_ylim[1], 0.05))
    ax.set_xticks([])
    ax.set_ylabel('AUROC')
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y')

# Legend below the panels
fig.subplots_adjust(bottom=0.3)
handles = [plt.Rectangle((0, 0), 1, 1, color=model['color']) for model in models]
chance_line = plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.5)
handles.append(chance_line)

legend_ncol = 2
fig.legend(handles, [get_display_name(model['name']) for model in models] + ["Chance"],
           loc='upper center',
           ncol=legend_ncol,
           frameon=False,
           bbox_to_anchor=(0.5, 0.05),
           fontsize=9)

plt.tight_layout()

os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'neuroprobe_eval_leaderboard_overall_splits.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'Saved figure to {save_path}')
jpg_path = os.path.join(output_dir, 'neuroprobe_eval_leaderboard_overall_splits.jpg')
plt.savefig(jpg_path, dpi=300, bbox_inches='tight')
print(f'Saved figure to {jpg_path}')
plt.close()

### SAVE COMBINED OVERALL TABLE ###

combined = {}
for model in models:
    combined[model['name']] = {}
    for split_type in split_types:
        perf = all_data[split_type].get('Overall', {}).get(model['name'])
        combined[model['name']][split_type] = perf if perf else {'mean': None, 'sem': None}

combined_json = os.path.join(output_dir, 'neuroprobe_eval_leaderboard_overall_splits.json')
with open(combined_json, 'w') as f:
    json.dump(combined, f, indent=2)
print(f'Saved combined performance data to {combined_json}')

### GENERATE LATEX TABLE ###

def _fmt(perf):
    if perf is None:
        return "--"
    mean = perf.get('mean')
    sem = perf.get('sem')
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    sem_str = f"{sem:.3f}" if (sem is not None and not (isinstance(sem, float) and np.isnan(sem))) else "--"
    return f"{mean:.3f} $\\pm$ {sem_str}"

latex_lines = []
latex_lines.append("\\begin{table}[h]")
latex_lines.append("\\centering")
latex_lines.append("\\begin{tabular}{l" + "c" * len(split_types) + "}")
latex_lines.append("\\hline")
latex_lines.append("Model & " + " & ".join(split_names[s] for s in split_types) + " \\\\")
latex_lines.append("\\hline")

# Best (per split) for bolding — ignore models with no data for that split.
best_by_split = {}
for split_type in split_types:
    candidates = []
    for model in models:
        perf = all_data[split_type].get('Overall', {}).get(model['name'])
        if not perf:
            continue
        mean = perf.get('mean')
        if mean is None or (isinstance(mean, float) and np.isnan(mean)):
            continue
        candidates.append((model['name'], mean))
    if candidates:
        best_by_split[split_type] = max(candidates, key=lambda kv: kv[1])[0]

for model in models:
    row_cells = [model['short_name']]
    for split_type in split_types:
        perf = all_data[split_type].get('Overall', {}).get(model['name'])
        cell = _fmt(perf)
        if best_by_split.get(split_type) == model['name'] and cell != "--":
            cell = f"\\textbf{{{cell}}}"
        row_cells.append(cell)
    latex_lines.append(" & ".join(row_cells) + " \\\\")

latex_lines.append("\\hline")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\caption{Leaderboard task-mean AUROC across splits (mean $\\pm$ SEM). "
                  "Best performer per split is shown in bold.}")
latex_lines.append("\\label{tab:leaderboard_overall_splits}")
latex_lines.append("\\end{table}")

latex_filename = os.path.join(output_dir, 'neuroprobe_eval_leaderboard_overall_splits.tex')
with open(latex_filename, 'w') as f:
    f.write('\n'.join(latex_lines))
print(f'Saved LaTeX table to {latex_filename}')

print('Overall leaderboard performance comparison across splits completed!')
