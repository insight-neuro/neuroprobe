import json
import csv
from pathlib import Path

def parse_eval_name(filename: str) -> str:
    # population_btbank3_0_gpt2_surprisal.json -> gpt2_surprisal
    stem = Path(filename).stem
    parts = stem.split("_")
    return "_".join(parts[3:])  # after population, btbank3, 0

rows = []
root = Path("eval_results")

for json_path in root.rglob("*.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_dir = json_path.parent.name  # e.g. linear_laplacian-stft_abs_...
    eval_name = parse_eval_name(json_path.name)
    model_name = data.get("model_name", "")

    eval_results = data.get("evaluation_results", {})
    for session_id, session_blob in eval_results.items():
        pop = session_blob.get("population", {})
        time_bins = pop.get("time_bins", [])
        for b in time_bins:
            t0 = b.get("time_bin_start")
            t1 = b.get("time_bin_end")
            folds = b.get("folds", [])
            for fold_idx, fold in enumerate(folds):
                row = {
                    "path": str(json_path),
                    "model_dir": model_dir,
                    "model_name": model_name,
                    "eval_name": eval_name,
                    "session_id": session_id,
                    "time_bin_start": t0,
                    "time_bin_end": t1,
                    "fold": fold_idx,
                }
                # keep any numeric metrics present
                for k, v in fold.items():
                    row[k] = v
                rows.append(row)

out_path = Path("eval_results_export_new_with_delta_delta.csv")
fieldnames = sorted({k for r in rows for k in r.keys()})
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} rows -> {out_path}")