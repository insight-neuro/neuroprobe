# Neuroprobe

<p align="center">
  <a href="https://neuroprobe.dev">
    <img src="https://github.com/azaho/neuroprobe/blob/main/website/neuroprobe_animation.gif?raw=True" alt="Neuroprobe Logo" style="height: 10em" />
  </a>
</p>

<p align="center">
    <a href="https://www.python.org/">
        <img alt="Python" src="https://img.shields.io/badge/Python-3.8+-1f425f.svg?color=purple">
    </a>
    <a href="https://pytorch.org/">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg">
    </a>
    <a href="https://mit-license.org/">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg">
    </a>
</p>

<p align="center"><strong>Neuroprobe: Benchmark for Evaluating iEEG Foundation Models.</strong></p>
<p align="center"><strong>Evaluating Intracranial Brain Responses to Naturalistic Stimuli</strong></p>

<p align="center">
    <a href="https://neuroprobe.dev">🌐 Website</a> |
    <a href="https://www.arxiv.org/abs/2509.21671">📄 Paper</a> |
    <a href="https://github.com/azaho/neuroprobe/blob/main/examples/quickstart.ipynb">🚀 Example Usage</a> |
    <a href="https://github.com/azaho/neuroprobe/blob/main/SUBMIT.md">📤 Submit</a>
</p>

---

By **Andrii Zahorodnii¹²***, **Christopher Wang¹***, **Bennett Stankovits¹***, **Charikleia Moraitaki¹**, **Geeling Chau³**, **Andrei Barbu¹**, **Boris Katz¹**, **Ila R Fiete¹²**,

¹MIT CSAIL, CBMM  |  ²MIT McGovern Institute  |  ³Caltech  |  *Equal contribution

## Overview
Neuroprobe is a benchmark for evaluating EEG/iEEG/sEEG/ECoG foundation models and understanding how the brain processes information across multiple tasks. It analyzes intracranial recordings during naturalistic stimuli using techniques from modern natural language processing. By probing neural responses across many tasks simultaneously, Neuroprobe aims to reveal the functional organization of the brain and relationships between different cognitive processes. The benchmark includes tools for decoding neural signals using both simple linear models and advanced neural networks, enabling researchers to better understand how the brain processes information across vision, language, and audio domains.

Please see the full technical paper for more details.

## Getting Started

### Prerequisites

1. Install the package:
```bash
pip install neuroprobe
```

2. If you haven't yet, download the BrainTreebank dataset from [the official release webpage](https://braintreebank.dev/), or using the following script (located [here](https://github.com/azaho/neuroprobe/blob/main/braintreebank_download_extract.py)):
```bash
python braintreebank_download_extract.py --lite
```
(lite is an optional flag; if only using Neuroprobe as a benchmark, this flag will reduce the number of downloaded files by >50% by removing unnecessary files.)

### Code Example

Start experimenting with [quickstart.ipynb](https://github.com/azaho/neuroprobe/blob/main/examples/quickstart.ipynb) to create datasets and evaluate models. For example:
```python
import os, torch
os.environ['ROOT_DIR_BRAINTREEBANK'] = '/path/to/braintreebank/'  # NOTE: Change this to your own path, or define an environment variable elsewhere

from neuroprobe import BrainTreebankSubject, BrainTreebankSubjectTrialBenchmarkDataset
subject = BrainTreebankSubject(subject_id=1, cache=True, 
                               dtype=torch.float32, coordinates_type="cortical")
dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, trial_id=2, 
                                                    dtype=torch.float32, 
                                                    eval_name="gpt2_surprisal") 

data_electrode_labels = dataset.electrode_labels 
data_electrode_coordinates = dataset.electrode_coordinates 

dataset.output_dict = True # Optionally, you can request the output_dict=True to get the data as a dictionary with a bunch of metadata.
dataset.output_indices = False # Optionally, you can request to output indices into the original BrainTreebank h5 files of the sessions, instead of raw data.
print(dataset[0])
```
will give the following output:
```python
{
	'data': torch.tensor, # shape: (n_electrodes, 2048), where 2048 = 1 second at 2048 Hz
	'label': int, # index of the class to be predicted: 0, 1, etc.
	'electrode_labels': list[str], # length: (n_electrodes, )
	'electrode_coordinates': torch.tensor, # shape: (n_electrodes, 3)
	'metadata': {'dataset_identifier': 'braintreebank', 'subject_id': 1, 'trial_id': 2, 'sampling_rate': 2048}
}
```
In case you'd like to use your own pipeline for extracting and preprocessing data, feel free to set `dataset.output_indices = True`, in which case the output will look like:
```python
{
    'data': (index_from, index_to), # tuple of indices: indices into the session's h5 file in the BrainTreebank
    ... # the same as above
}
```

### Leaderboard Requirements

To submit to the Neuroprobe leaderboard, you MUST use the exact train/val/test splits that are provided by the Neuroprobe package:
```python
# options: generate_splits_within_session, generate_splits_cross_session, generate_splits_cross_subject
splits = generate_splits_cross_session(subject, test_trial_id=2, 
                                       eval_name="gpt2_surprisal", output_indices=False)
print(splits[0])
```
will give the following output:
```python
{
    "train_dataset": BrainTreebankSubjectTrialBenchmarkDataset,
    "val_dataset": BrainTreebankSubjectTrialBenchmarkDataset,
    "test_dataset": BrainTreebankSubjectTrialBenchmarkDataset
}
```

### Evaluation Example

Run the linear regression model evaluation using the following example script (located [here](https://github.com/azaho/neuroprobe/blob/main/examples/eval_population.py)):
```bash
python eval_population.py --subject_id SUBJECT_ID --trial_id TRIAL_ID --verbose --eval_name gpt2_surprisal --split_type CrossSession
```

Results will be saved in the `eval_results` directory according to `leaderboard_schema.json`.

## Citation

If you use Neuroprobe in your work, please cite our paper:
```bibtex
@misc{neuroprobe,
      title={Neuroprobe: Evaluating Intracranial Brain Responses to Naturalistic Stimuli}, 
      author={Andrii Zahorodnii and Christopher Wang and Bennett Stankovits and Charikleia Moraitaki and Geeling Chau and Andrei Barbu and Boris Katz and Ila R Fiete},
      year={2025},
      eprint={2509.21671},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.21671}, 
}
```
