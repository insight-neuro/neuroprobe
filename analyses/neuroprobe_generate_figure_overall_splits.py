import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os

### LOAD DATA FROM ALL SPLIT TYPES ###

split_types = ['SS_SM', 'SS_DM', 'DS_DM']
split_names = {
    'SS_SM': 'Within Session', 
    'SS_DM': 'Cross Session',
    'DS_DM': 'Cross Subject'
}

# Load performance data for all splits
all_data = {}
for split_type in split_types:
    filename = f'analyses/figures/neuroprobe_eval_lite_{split_type}.json'
    with open(filename, 'r') as f:
        all_data[split_type] = json.load(f)

### DEFINE MODELS ###

models = [
    {
        'name': 'Linear (raw voltage)',
        'short_name': 'Linear (voltage)',
        'color_palette': 'viridis',
    },
    {
        'name': 'Linear (spectrogram)',
        'short_name': 'Linear (spectrogram)',
        'color_palette': 'viridis', 
    },
    {
        'name': 'Linear (Laplacian re-referencing + spectrogram)',
        'short_name': 'Linear (Laplacian+spectrogram)',
        'color_palette': 'viridis', 
    },
    {
        'name': 'BrainBERT (untrained, frozen)',
        'color_palette': 'viridis', 
        'pad_x': 1,
    },
    {
        'name': 'BrainBERT (frozen; Wang et al. 2023)',
        'short_name': 'BrainBERT (frozen)',
        'color_palette': 'viridis', 
    },
    {
        'name': 'PopulationTransformer (Chau et al. 2024)',
        'short_name': 'PopulationTransformer',
        'color_palette': 'viridis', 
        'pad_x': 1,
    },
]

for model in models:
    if 'short_name' not in model:
        model['short_name'] = model['name'] # if short_name is not defined, use the full name

### PREPARING FOR PLOTTING ###

# Add Arial font
import matplotlib.font_manager as fm
font_path = 'assets/font_arial.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 12})

# Assign colors to models based on color palette
color_palette_ids = {}
for model in models:
    if model['color_palette'] not in color_palette_ids: 
        color_palette_ids[model['color_palette']] = 0
    model['color_palette_id'] = color_palette_ids[model['color_palette']]
    color_palette_ids[model['color_palette']] += 1

for model in models:
    model['color'] = sns.color_palette(model['color_palette'], color_palette_ids[model['color_palette']])[model['color_palette_id']]

# Assign model x-positions
current_x_pos = 0
for i, model in enumerate(models):
    if model.get('pad_x', 0): 
        current_x_pos += model['pad_x']
    model['x_pos'] = current_x_pos
    current_x_pos += 1

### PLOT FIGURE ###

figure_size_multiplier = 3
base_width = figure_size_multiplier * 3
base_height = figure_size_multiplier

fig, axes = plt.subplots(1, 3, figsize=(base_width, base_height))

# Bar width
bar_width = 0.2

# Y-axis limits
axis_ylim = (0.48, 0.68)

for split_idx, split_type in enumerate(split_types):
    ax = axes[split_idx]
    
    # Plot bars for each model
    for i, model in enumerate(models):
        perf = all_data[split_type]['overall'][model['name']]
        ax.bar(model['x_pos']*bar_width, perf['mean'], bar_width,
                yerr=perf['sem'],
                color=model['color'],
                capsize=6)
    
    # Customize plot
    ax.set_title(split_names[split_type], pad=15)
    ax.set_ylim(axis_ylim)
    ax.set_yticks(np.arange(0.5, axis_ylim[1], 0.05))
    ax.set_xticks([])
    ax.set_ylabel('AUROC')
    
    # Add horizontal line at chance level
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make tick labels smaller
    ax.tick_params(axis='y')

# Create legend below the plots
fig.subplots_adjust(bottom=0.25)
handles = [plt.Rectangle((0,0),1,1, color=model['color']) for model in models]
chance_line = plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.5)
handles.append(chance_line)

fig.legend(handles, [model['name'] for model in models] + ["Chance"],
           loc='upper center', 
           ncol=2,
           frameon=False,
           bbox_to_anchor=(0.5, 0.05))

# Use tight_layout
plt.tight_layout()

# Save figure
save_path = 'analyses/figures/neuroprobe_eval_overall_splits.pdf'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'Saved figure to {save_path}')
plt.close()

print('Overall performance comparison across splits completed!')
