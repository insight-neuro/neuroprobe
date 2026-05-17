"""
Smoke-check the bundled MNI coordinates and produce a glass-brain figure.

For each of the 10 Braintreebank subjects this script:
  - constructs a BrainTreebankSubject with coordinates_type='mni',
  - calls get_electrode_coordinates() and asserts the tensor is finite and
    of the expected shape,
  - aggregates all electrodes across subjects and renders them on the MNI152
    template via nilearn.plot_markers.

If MNI152 extraction is wrong (sign flip, axis swap, transform error), points
will fall outside the brain envelope or cluster on the wrong side.

Output:
    analyses/electrode_coordinates_mni/glass_brain.png

Usage:
    ROOT_DIR_BRAINTREEBANK=/path/to/braintreebank python \\
        analyses/electrode_coordinates_mni/check_and_plot_mni_coords.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from nilearn import plotting

from neuroprobe.braintreebank_subject import BrainTreebankSubject


SUBJECT_IDS = list(range(1, 11))
OUT_PATH = Path(__file__).parent / 'glass_brain.png'


def main() -> None:
    if 'ROOT_DIR_BRAINTREEBANK' not in os.environ:
        sys.exit("Set ROOT_DIR_BRAINTREEBANK before running this script.")

    all_coords = []
    all_subject_ids = []

    for sid in SUBJECT_IDS:
        subj = BrainTreebankSubject(
            subject_id=sid,
            allow_corrupted=False,
            cache=False,
            dtype=torch.float32,
            coordinates_type='mni',
        )
        coords = subj.get_electrode_coordinates()
        assert coords.shape == (len(subj.electrode_labels), 3), \
            f"btbank{sid}: shape {coords.shape} != ({len(subj.electrode_labels)}, 3)"
        assert torch.isfinite(coords).all(), \
            f"btbank{sid}: non-finite MNI coordinate found"
        print(f"btbank{sid:<2}  {len(subj.electrode_labels):4d} electrodes  "
              f"x={coords[:,0].min().item():+6.1f}..{coords[:,0].max().item():+6.1f}  "
              f"y={coords[:,1].min().item():+6.1f}..{coords[:,1].max().item():+6.1f}  "
              f"z={coords[:,2].min().item():+6.1f}..{coords[:,2].max().item():+6.1f}")

        all_coords.append(coords.numpy())
        all_subject_ids.extend([sid] * len(coords))

    all_coords = np.concatenate(all_coords, axis=0)
    all_subject_ids = np.array(all_subject_ids, dtype=float)
    print(f"\nTotal: {len(all_coords)} electrodes across {len(SUBJECT_IDS)} subjects")

    display = plotting.plot_markers(
        node_values=all_subject_ids,
        node_coords=all_coords,
        node_size=6,
        node_cmap='tab10',
        display_mode='lyrz',
        colorbar=True,
        title=f'{len(all_coords)} electrodes across {len(SUBJECT_IDS)} subjects on MNI152',
    )
    display.savefig(str(OUT_PATH), dpi=180)
    display.close()
    print(f"wrote {OUT_PATH}")


if __name__ == '__main__':
    main()
