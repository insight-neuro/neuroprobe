import os
from pathlib import Path
import argparse
import pandas as pd


BRIGHTNESS_COL = "mean_pixel_brightness"
FACE_COL = "face_num"


def add_delta_features(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """
    Adds first-difference, abs(first-difference), second-difference, abs(second-difference).
    Matches the notebook logic: diff().fillna(0) and abs().
    """
    d1 = df[col].diff().fillna(0)
    df[f"delta_{prefix}"] = d1
    df[f"delta_{prefix}_abs"] = d1.abs()

    d2 = d1.diff().fillna(0)
    df[f"delta_delta_{prefix}"] = d2
    df[f"delta_delta_{prefix}_abs"] = d2.abs()
    return df


def process_movie_transcript(movie_dir: Path, out_name: str, overwrite: bool = False) -> bool:
    """
    Reads transcripts/<movie>/features.csv and writes transcripts/<movie>/test_new_delta_pixel_and_face.csv
    Returns True if written.
    """
    in_path = movie_dir / "features.csv"
    out_path = movie_dir / out_name

    if not in_path.exists():
        return False

    if out_path.exists() and not overwrite:
        # keep existing file
        return False

    df = pd.read_csv(in_path)

    # The dataset loader expects this column to exist (it does .set_index("Unnamed: 0"))
    if "Unnamed: 0" not in df.columns:
        raise ValueError(f"{in_path} missing required column 'Unnamed: 0'")

    # Add brightness deltas if available
    if BRIGHTNESS_COL in df.columns:
        df = add_delta_features(df, BRIGHTNESS_COL, "mean_pixel_brightness")

    # Add face count deltas if available
    if FACE_COL in df.columns:
        df = add_delta_features(df, FACE_COL, "face_num")

    df.to_csv(out_path, index=False)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default=os.environ.get("ROOT_DIR_BRAINTREEBANK", ""),
        help="Braintree root dir (same as ROOT_DIR_BRAINTREEBANK). Should contain transcripts/.",
    )
    parser.add_argument("--out_name", type=str, default="test_new_features.csv", help="Output file name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSVs")
    args = parser.parse_args()

    if not args.root_dir:
        raise ValueError("root_dir not provided and ROOT_DIR_BRAINTREEBANK env var is not set")

    root = Path(args.root_dir)
    transcripts_dir = root / "transcripts"
    if not transcripts_dir.exists():
        raise ValueError(f"Expected transcripts/ at: {transcripts_dir}")
    

    written = 0
    skipped = 0

    for movie_dir in sorted([p for p in transcripts_dir.iterdir() if p.is_dir()]):
        did_write = process_movie_transcript(movie_dir, args.out_name, overwrite=args.overwrite)
        if did_write:
            written += 1
        else:
            skipped += 1

    print(f"Done. Wrote: {written}, skipped: {skipped}. Output file: test_new_delta_pixel_and_face.csv per movie.")


if __name__ == "__main__":
    main()