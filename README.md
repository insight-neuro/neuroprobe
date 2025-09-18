# Neuroprobe

<p align="center">
  <a href="https://neuroprobe.dev">
    <img src="assets/neuroprobe_animation.gif" alt="Neuroprobe Logo" style="height: 10em" />
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

<p align="center"><strong>Neuroprobe: Evaluating Intracranial Brain Responses to Naturalistic Stimuli</strong></p>

<p align="center">
    <a href="https://neuroprobe.dev">Website</a> |
    <a href="https://azaho.org/papers/NeurIPS_2025__BTBench_paper.pdf">Paper</a> |
    <a href="https://github.com/azaho/neuroprobe/blob/main/examples/quickstart.ipynb">Example Usage</a>
</p>

---

By **Andrii Zahorodnii¹²***, **Bennett Stankovits¹***, **Christopher Wang¹***, **Charikleia Moraitaki¹**, **Geeling Chau³**, **Ila R Fiete¹²**, **Boris Katz¹**, **Andrei Barbu¹**

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

2. If you haven't yet, download the BrainTreebank dataset from [the official release webpage](https://braintreebank.dev/), or using the following script:
```bash
python braintreebank_download_extract.py --lite
```
(lite is an optional flag; if only using Neuroprobe as a benchmark, this flag will reduce the number of downloaded files by >50% by removing unnecessary files.)

3. Start experimenting with `quickstart.ipynb` to create datasets and evaluate models.

## 📊 Evaluation

Run the linear regression model evaluation:
```bash
python single_electrode.py --subject SUBJECT_ID --trial TRIAL_ID --verbose --lite --eval_name onset --split_type CrossSession
```

Results will be saved in the `eval_results` directory according to `leaderboard_schema.json`.

## Citation

If you use Neuroprobe in your work, please cite our paper:
```bibtex
[Citation TBD]
```