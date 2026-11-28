"""Shared recording-time folds; support exclusion precedes balancing and caps."""

from numbers import Integral, Real

import numpy as np

TEMPORAL_SPLIT_PROTOCOL = "shared-temporal-v1"


def temporal_fold_indices(
    start,
    end,
    labels,
    *,
    recording_n_samples,
    sampling_rate,
    n_classes,
    context_width_s=None,
    gap_s=0.0,
    random_seed=42,
    max_samples=None,
):
    """Select two folds from full candidate windows expressed as raw sample indices.

    ``max_samples`` is a total per-fold budget: half train, quarter validation,
    quarter test, rounded down to complete class sets. ``None`` leaves counts
    uncapped. Empty/missing classes produce an empty split, without moving events
    across blocks or silently dropping fold IDs. Bounds are half-open.
    """
    for name, value in (
        ("recording_n_samples", recording_n_samples),
        ("n_classes", n_classes),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, Integral)
        or random_seed < 0
    ):
        raise ValueError("random_seed must be a nonnegative integer")
    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, Integral)
        or max_samples < 1
    ):
        raise ValueError("max_samples must be a positive integer or None")
    if (
        isinstance(sampling_rate, bool)
        or not isinstance(sampling_rate, Real)
        or not np.isfinite(sampling_rate)
        or sampling_rate <= 0
    ):
        raise ValueError("sampling_rate must be positive and finite")
    if (
        isinstance(gap_s, bool)
        or not isinstance(gap_s, Real)
        or not np.isfinite(gap_s)
        or gap_s < 0
    ):
        raise ValueError("gap_s must be nonnegative and finite")
    if context_width_s is not None and (
        isinstance(context_width_s, bool)
        or not isinstance(context_width_s, Real)
        or not np.isfinite(context_width_s)
        or context_width_s <= 0
    ):
        raise ValueError("context_width_s must be positive and finite or None")
    start, end, labels = np.asarray(start), np.asarray(end), np.asarray(labels)
    if start.ndim != 1 or end.shape != start.shape or labels.shape != start.shape:
        raise ValueError(
            "start, end and labels must be matching one-dimensional arrays"
        )
    if any(
        not np.issubdtype(array.dtype, np.integer) for array in (start, end, labels)
    ):
        raise TypeError(
            "start, end and labels must contain integer sample indices/classes"
        )
    if np.any(end <= start) or not np.isin(labels, np.arange(n_classes)).all():
        raise ValueError(
            "Candidate windows must be increasing and labels in [0, n_classes)"
        )
    widths = end - start
    if len(widths) and np.any(widths != widths[0]):
        raise ValueError("Candidate target windows must have one common duration")
    if (
        context_width_s is not None
        and len(widths)
        and context_width_s * sampling_rate < widths[0]
    ):
        raise ValueError("context_width_s cannot be shorter than the target window")
    extra = (
        0
        if context_width_s is None or not len(widths)
        else int(np.ceil((context_width_s * sampling_rate - widths[0]) / 2))
    )
    gap_samples = int(np.ceil(gap_s * sampling_rate))
    result = []
    for fold_idx, (fractions, order) in enumerate(
        (
            ([0, 0.25, 0.5, 1], ("val", "test", "train")),
            ([0, 0.5, 0.75, 1], ("train", "val", "test")),
        )
    ):
        edges = np.ceil(np.array(fractions) * recording_n_samples).astype(np.int64)
        blocks, indices, eligible_counts, counts = {}, {}, {}, {}
        caps = dict.fromkeys(order)
        if max_samples is not None:
            # Preserve historical twofold remainder allocation and split order.
            total = (int(max_samples) // n_classes) * n_classes
            heldout = (total + 1) // 2 if fold_idx == 0 else total // 2
            caps.update(
                train=total - heldout,
                val=heldout // 2,
                test=heldout - heldout // 2,
            )
        for position, split in enumerate(order):
            lo = int(edges[position] + ((gap_samples + 1) // 2 if position else 0))
            hi = int(edges[position + 1] - (gap_samples // 2 if position < 2 else 0))
            if hi <= lo:
                raise ValueError("gap_s removes an entire temporal block")
            blocks[split] = [lo, hi]
            # Compare against padded bounds so unsigned sample indices cannot wrap.
            keep = (start >= lo + extra) & (end <= hi - extra)
            by_class = [
                np.flatnonzero(keep & (labels == label)) for label in range(n_classes)
            ]
            n = min(map(len, by_class))
            cap = caps[split]
            if cap is not None:
                n = min(n, cap // n_classes)
            rng = np.random.default_rng(
                np.random.SeedSequence([random_seed, fold_idx, position])
            )
            selected = np.concatenate(
                [rng.choice(values, n, replace=False) for values in by_class]
            )
            indices[split] = selected[np.argsort(start[selected], kind="stable")]
            eligible_counts[split] = [len(values) for values in by_class]
            counts[split] = [n] * n_classes
        result.append(
            {
                "indices": indices,
                "temporal_split": {
                    "protocol": TEMPORAL_SPLIT_PROTOCOL,
                    "fold_idx": fold_idx,
                    "recording_bounds_samples": [0, int(recording_n_samples)],
                    "blocks_samples": blocks,
                    "sampling_rate": float(sampling_rate),
                    "context_width_s": None
                    if context_width_s is None
                    else float(context_width_s),
                    "gap_s": float(gap_s),
                    "target_window_samples": int(widths[0]) if len(widths) else None,
                    "support_padding_samples": extra,
                    "balance_seed": int(random_seed),
                    "max_samples": None if max_samples is None else int(max_samples),
                    "split_caps": caps,
                    "eligible_class_counts": eligible_counts,
                    "class_counts": counts,
                },
            }
        )
    return result
