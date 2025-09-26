# Submitting to the Neuroprobe Leaderboard

This document describes the steps which you must follow to submit your results to the public Neuroprobe leaderboard at [https://neuroprobe.dev](https://neuroprobe.dev).

## Pretraining guidelines
For the validity of the evaluation on Neuroprobe, **no models can be pretrained on the same data that underlies the Neuroprobe evaluation**. So, the following sessions are off-limits, with pretraining on them NOT allowed:
- Subject 1: Trials 1, 2
- Subject 2: Trials 0, 4
- Subject 3: Trials 0, 1
- Subject 4: Trials 0, 1
- Subject 7: Trials 0, 1
- Subject 10: Trials 0, 1

To be explicit, here are the subjects and trials from the BrainTreebank that pretraining is allowed on (btbankX_Y means brain treebank subject X trial Y):
```
btbank1_0,btbank2_1,btbank2_2,btbank2_3,btbank2_5,btbank2_6,btbank3_2,btbank4_2,btbank5_0,btbank6_0,btbank6_1,btbank6_4,btbank8_0,btbank9_0
```
Note that the recording session list above does not contain any trials of subjects 10 or 7. That is because Neuroprobe contains 2 trials per subject, and in the Braintreebank dataset subjects 7 and 10 only contain two trials. So, we also created the "partial" sessions for subjects 10 and 7, which are taken from the same trials as Neuroprobe but during the times that do not intersect with Neuroprobe indices. Reserachers are allowed to use the following special subject/trials for pretraining:
```
btbank7_100,btbank7_101,btbank7_102,btbank10_100,btbank10_101
```
There's total around ~20 minutes for subject 7 and 10 each in those extra sessions.
Please see these session parts uploaded [here (Google Drive link)](https://drive.google.com/drive/u/0/folders/1eUXKD-Nf0S5bUEVLo_boYxAxvXDRy9q9).
The files in the link above ideally should be fine to just drop into the braintreebank folder, as they follow the same h5 format as the other trials.

## Formatting results
Note that for the Cross-Session split, Neuroprobe contains 12 recording sessions x 15 tasks = 180 different evaluations, which will be eventually combined together on the leaderboard. (For the Cross-Subject split, subject 2 is not evaluated on, which leaves 10 recording sessions x 15 tasks = 150 evaluations.)

The evaluation results are aggregated per task, in files named `population_TASKNAME.json`. Those JSON files must be structured like below:
```json
{
    "model_name": "Linear Regression",           # Name of the model
    "description": "Linear Regression model",    # Brief description of the model
    "author": "John Doe",                        # Name of the submitting author to the leaderboard
    "organization": "MIT",                       # Organization associated with the model (can be an individual). Ideally, a short abbreviation.
    "organization_url": "https://mit.edu",       # URL of the organization
    "timestamp": 0,                              # Timestamp associated with the result.

    "evaluation_results": {
        "btbank1_1": {                           # 
            "population": {
                "one_second_after_onset": {
                    "time_bin_start": 0.0,
                    "time_bin_end": 1.0,
                    "folds": [                   # In case of Cross-Session and Cross-Subject splits, this will be just one fold. For Within-Session, there will be two folds.
                        {"train_accuracy": 0.5, "train_roc_auc": 0.6, "test_accuracy": 0.5, "test_roc_auc": 0.6}
                    ]
                }
            }
        },
        "btbank1_2": { [...] }
    }
}
```
The `TASKNAME` come from the following list of tasks from the Neuroprobe config file (use the keys on the left as the `TASKNAME`):
```python
neuroprobe.config.NEUROPROBE_TASKS_MAPPING = {
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
```

## Submission guidelines
Results are submitted to the Neuroprobe leaderboard via a pull request to the Neuroprobe github repository. Fork the Neuroprobe repository, then, in the folder `leaderboard`, make a directory with the following name: `MODELNAME_AUTHORFIRSTNAME_AUTHORLASTNAME_DAY_MONTH_YEAR`. This directory must contain:
- Either a folder `Within-Session`, or a folder `Cross-Session`, `Cross-Session`, or any combination (whichever exists will determine where the results will be uploaded on the Neuroprobe website). The contents of this folder must follow the structure from the section "Formatting results" above.
- A file `metadata.json`, with the following structure:
```json
{
    "model_name": "Linear Regression",           # Name of the model
    "description": "Linear Regression model",    # Brief description of the model
    "author": "John Doe",                        # Name of the submitting author to the leaderboard
    "organization": "MIT",                       # Organization associated with the model (can be an individual). Ideally, a short abbreviation.
    "organization_url": "https://mit.edu",       # URL of the organization
    "logo_url": "https://mit.edu/logo.jpg",      # Optionally, a URL to the logo of the organization, which may be displayed on the leaderboard entry.
    "timestamp": 0                               # Timestamp associated with the submission.
}
```
- A file `PUBLICATION.bib`, which contains a bibtex entry containing the citation to the publication associated with the model. The publication must describe in detail, or ideally provide code, how to reproduce the reported evaluation results on Neuroprobe. NOTE: this is required to ensure transparency of submissions.
- A file `ATTESTATION.txt`, with the following attestation:
```txt
I attest that the training and test splits of Neuroprobe were respected and taken from the `neuroprobe/train_test_splits.py` function.
SIGN **Full Name Of Submitting Author**

I attest that the submitted model was not pretrained on any data that intersects with any data of Neuroprobe.
SIGN **Full Name Of Submitting Author**
[If the data intersects, please explain here]
```

Finally, submit a pull request from the repository to the Neuroprobe repository. We will review it as soon as possible. If the pull request gets merged, the results will be automatically updated on the Neuroprobe website. 
NOTE: The Neuroprobe repository contains automatic tests to ensure the format of the submission is correct according to the 'Submission Guidelines' section above. If a test fails, please note what was the issue, withdraw the PR, fix the issue, and resubmit the PR. You may run the tests on your own local copy to ensure they pass before submitting the PR.