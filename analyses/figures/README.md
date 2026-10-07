# Leaderboard figure outputs

Regenerate the leaderboard plots and accompanying tables from the repository root:

```sh
python analyses/neuroprobe_generate_figure_from_leaderboard_results.py --split_type WithinSession
python analyses/neuroprobe_generate_figure_from_leaderboard_results.py --split_type CrossSession
python analyses/neuroprobe_generate_figure_from_leaderboard_results.py --split_type CrossSubject
python analyses/neuroprobe_generate_figure_overall_splits_from_leaderboard_results.py
```

These scripts need NumPy, Matplotlib and Seaborn. For a headless environment, set
`MPLBACKEND=Agg`; set `MPLCONFIGDIR` to a writable cache directory if needed.

Commit the generated JSON and LaTeX tables. PDF/JPG plots are intentionally
ignored by the repository's `.gitignore`; regenerate them for use in the paper.
The combined plot must be generated after all three per-split JSON files.

The MAPA update uses the submitted results in
`leaderboard/MAPA_Ben_Tang_09_09_2026/`, including its `PUBLICATION.bib` citation.
It does not rerun model evaluation. Per-task scores average folds within each
subject/session and then average those session scores. The scripts retain the
existing SEM calculation and task-mean aggregation.
