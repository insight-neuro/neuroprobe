# Devon Setup & Run Guide

## 1. Connect to Devon
```bash
ssh devon
```

## 2. Activate conda
```bash
export CONDARC=/storage/eg99/.condarc
export CONDA_ENVS_PATH=/storage/eg99/anaconda3/envs
source /storage/eg99/anaconda3/etc/profile.d/conda.sh
conda activate neuroprobe
```
If you get a permission denied error on `.condarc`, the `export CONDARC` line above fixes it by redirecting conda config away from AFS.

## 3. Get latest code
```bash
cd /storage/eg99/neuroprobe
git pull
```

## 4. Set up data directory (first time only)
Only needed if `/storage/eg99/braintreebank_data` doesn't exist yet:
```bash
mkdir -p /storage/eg99/braintreebank_data

ln -s /storage/czw/braintreebank_data/all_subject_data/*.h5 /storage/eg99/braintreebank_data/
ln -s /storage/czw/braintreebank_data/localization        /storage/eg99/braintreebank_data/
ln -s /storage/czw/braintreebank_data/electrode_labels    /storage/eg99/braintreebank_data/
ln -s /storage/czw/braintreebank_data/corrupted_elec.json /storage/eg99/braintreebank_data/
ln -s /storage/czw/braintreebank_data/subject_metadata    /storage/eg99/braintreebank_data/
ln -s /storage/czw/braintreebank_data/transcripts         /storage/eg99/braintreebank_data/
ln -s /storage/czw/braintreebank_data/subject_timings     /storage/eg99/braintreebank_data/
ln -s /storage/czw/braintreebank_data/scene_annotations.json /storage/eg99/braintreebank_data/
```

## 5. Check all data files are present
```bash
cd /storage/eg99/neuroprobe/examples
NEUROPROBE_FEATURES_FILE=features.csv \
ROOT_DIR_BRAINTREEBANK=/storage/eg99/braintreebank_data \
  python check_data.py
```
Fix any `[MISSING]` items before continuing.

## 6. Run benchmark (pick best GNN variant, ~30 min)
Compares 4 combinations: v1_stgcn + v2_gat × coords + functional, on subject 1 trial 1.
```bash
cd /storage/eg99/neuroprobe/examples
NEUROPROBE_FEATURES_FILE=features.csv \
ROOT_DIR_BRAINTREEBANK=/storage/eg99/braintreebank_data \
  bash run_gnn_benchmark.sh
```
Results go to `eval_results/gnn_benchmark/`. Pick the variant with highest `test_roc_auc`.

## 7. Run full competition (all 16 tasks, all splits)
Update `GNN_VARIANT` and `GNN_GRAPH` based on benchmark results, then:
```bash
cd /storage/eg99/neuroprobe/examples
NEUROPROBE_FEATURES_FILE=features.csv \
ROOT_DIR_BRAINTREEBANK=/storage/eg99/braintreebank_data \
  bash run_gnn_competition.sh
```
Or override variant at runtime:
```bash
GNN_VARIANT=gnn_v2_gat GNN_GRAPH=functional \
NEUROPROBE_FEATURES_FILE=features.csv \
ROOT_DIR_BRAINTREEBANK=/storage/eg99/braintreebank_data \
  bash run_gnn_competition.sh
```
Results go to `eval_results/gnn_competition/{WithinSession,CrossSession,CrossSubject}/`.
Safe to interrupt and resume — uses `--if_exists skip`.

---

## Key paths
| What | Path |
|------|------|
| Conda | `/storage/eg99/anaconda3` |
| Neuroprobe repo | `/storage/eg99/neuroprobe` |
| Data root | `/storage/eg99/braintreebank_data` |
| Collaborator data | `/storage/czw/braintreebank_data` |
| Results | `/storage/eg99/neuroprobe/examples/eval_results` |

## Key env vars
| Var | Value |
|-----|-------|
| `ROOT_DIR_BRAINTREEBANK` | `/storage/eg99/braintreebank_data` |
| `NEUROPROBE_FEATURES_FILE` | `features.csv` (collaborator data has no `test_new_features.csv`) |
| `GNN_VARIANT` | `gnn_v1_stgcn` or `gnn_v2_gat` |
| `GNN_GRAPH` | `coords` or `functional` |
