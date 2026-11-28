import json
import os

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp")

import neuroprobe.braintreebank_subject as subject_module
import neuroprobe.datasets as dataset_module
from neuroprobe import BrainTreebankSubject, BrainTreebankSubjectTrialBenchmarkDataset
from neuroprobe.temporal_splits import temporal_fold_indices
from neuroprobe.train_test_splits import generate_splits_within_session


@pytest.fixture
def subject(tmp_path, monkeypatch):
    """Real subject/dataset path using small H5 and annotation files."""
    for directory in (
        "localization/sub_1",
        "electrode_labels/sub_1",
        "transcripts/movie",
        "alignment",
    ):
        (tmp_path / directory).mkdir(parents=True)
    pd.DataFrame({"Electrode": ["A1", "A2"]}).to_csv(
        tmp_path / "localization/sub_1/depth-wm.csv", index=False
    )
    pd.DataFrame(
        dict(
            Subject=["sub_1"] * 2, Electrode=["A1", "A2"], X=[1, 2], Y=[3, 4], Z=[5, 6]
        )
    ).to_csv(tmp_path / "localization/elec_coords_full.csv", index=False)
    (tmp_path / "electrode_labels/sub_1/electrode_labels.json").write_text(
        json.dumps(["A1", "A2"])
    )
    with h5py.File(tmp_path / "sub_1_trial001.h5", "w") as handle:
        for channel in range(2):
            handle.create_dataset(
                f"data/electrode_{channel}",
                data=np.arange(100 * 2048, dtype=np.float32) + channel,
            )
    starts = np.arange(0.1, 99.9, 0.5)
    words = pd.DataFrame(
        dict(
            start=starts,
            end=starts + 0.2,
            est_idx=(starts * 2048).astype(int),
            original_index=np.arange(len(starts)),
            is_onset=np.arange(len(starts)) % 2,
            rms=np.arange(len(starts), dtype=float),
            face_num=np.arange(len(starts)) % 3,
        )
    )
    words.to_csv(tmp_path / "alignment/subject1_trial1_words_df.csv", index=False)
    negatives = np.arange(0.3, 99.9, 0.8)
    pd.DataFrame(dict(est_idx=(negatives * 2048).astype(int))).to_csv(
        tmp_path / "alignment/subject1_trial1_nonverbal_df.csv", index=False
    )
    words.to_csv(tmp_path / "transcripts/movie/features.csv")
    monkeypatch.setattr(subject_module, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(dataset_module, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(
        dataset_module, "SAVE_SUBJECT_TRIAL_DF_DIR", str(tmp_path / "alignment")
    )
    monkeypatch.setitem(
        dataset_module.BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING,
        "btbank1_1",
        "movie",
    )
    monkeypatch.setitem(
        dataset_module.NEUROPROBE_LITE_ELECTRODES, "btbank1", ["A1", "A2"]
    )
    monkeypatch.setitem(dataset_module.NEUROPROBE_NANO_ELECTRODES, "btbank1", ["A1"])
    obj = BrainTreebankSubject(1, allow_corrupted=True, cache=True)
    yield obj
    for handle in obj.h5_files.values():
        handle.close()


@pytest.mark.parametrize("width", [1, 9, 15])
def test_real_splits_containment_balancing_and_no_signal_read(subject, width):
    folds = generate_splits_within_session(
        subject,
        1,
        "speech",
        lite=False,
        output_indices=True,
        context_width_s=width,
        gap_s=1,
    )
    assert len(folds) == 2
    for fold in folds:
        meta = fold["temporal_split"]
        blocks = sorted(meta["blocks_samples"].values())
        assert blocks[1][0] - blocks[0][1] == 2048
        for split in ("train", "val", "test"):
            ds = fold[f"{split}_dataset"]
            assert ds.electrode_labels == ["A1", "A2"]
            bounds = [sample["data"] for sample in ds]
            labels = [sample["label"] for sample in ds]
            assert labels.count(0) == labels.count(1) > 0
            lo, hi = meta["blocks_samples"][split]
            assert all(
                a - (width - 1) * 1024 >= lo and b + (width - 1) * 1024 <= hi
                for a, b in bounds
            )
            assert all(b - a == 2048 for a, b in bounds)
    assert subject.neural_data_cache == {}
    assert subject.h5_files == {}


def test_boundaries_do_not_depend_on_task_or_seed(subject):
    speech = generate_splits_within_session(
        subject, 1, "speech", lite=False, output_indices=True, max_samples=40
    )
    volume = generate_splits_within_session(
        subject, 1, "volume", lite=False, output_indices=True
    )
    other_seed = generate_splits_within_session(
        subject,
        1,
        "speech",
        lite=False,
        output_indices=True,
        max_samples=40,
        random_seed=17,
    )
    repeat = generate_splits_within_session(
        subject, 1, "speech", lite=False, output_indices=True, max_samples=40
    )
    for a, b, c, d in zip(speech, volume, other_seed, repeat):
        assert (
            a["temporal_split"]["blocks_samples"]
            == b["temporal_split"]["blocks_samples"]
            == c["temporal_split"]["blocks_samples"]
        )
        for split, expected in (("train", 20), ("val", 10), ("test", 10)):
            assert len(a[f"{split}_dataset"]) == expected
            np.testing.assert_array_equal(
                a[f"{split}_dataset"].indices, d[f"{split}_dataset"].indices
            )
    assert not np.array_equal(
        speech[0]["train_dataset"].indices, other_seed[0]["train_dataset"].indices
    )


def test_late_balancing_recovers_candidates_and_supports_multiclass(subject):
    # Globally balancing before splitting would discard positive speech candidates.
    ds = BrainTreebankSubjectTrialBenchmarkDataset(
        subject,
        1,
        torch.float32,
        "speech",
        lite=False,
        output_indices=True,
        defer_selection=True,
    )
    assert len(ds) == len(ds.all_words_df) + len(ds.nonverbal_df)
    folds = generate_splits_within_session(
        subject, 1, "face_num", lite=False, binary_tasks=False, output_indices=True
    )
    for fold in folds:
        for split in ("train", "val", "test"):
            labels = [sample["label"] for sample in fold[f"{split}_dataset"]]
            assert labels.count(0) == labels.count(1) == labels.count(2) > 0


def test_actual_waveform_and_tier_electrodes(subject):
    fold = generate_splits_within_session(
        subject, 1, "speech", lite=True, nano=True, output_dict=False, max_samples=16
    )[0]
    sample, label = fold["train_dataset"][0]
    assert sample.shape == (1, 2048)
    base = fold["train_dataset"].dataset
    idx = fold["train_dataset"].indices[0]
    (start, end), expected_label = base.__getitem__(idx, force_output_indices=True)
    torch.testing.assert_close(sample[0], torch.arange(start, end, dtype=torch.float32))
    assert label == expected_label
    assert fold["train_dataset"].electrode_labels == ["A1"]


@pytest.mark.parametrize("defer_selection", [False, True])
def test_index_only_dataset_skips_waveform_cache_bounds(
    subject, monkeypatch, defer_selection
):
    def unexpected_waveform_setup(*args, **kwargs):
        raise AssertionError(
            "Index-only datasets must not prepare waveform cache bounds"
        )

    monkeypatch.setattr(
        BrainTreebankSubjectTrialBenchmarkDataset,
        "get_window_bounds",
        unexpected_waveform_setup,
    )
    monkeypatch.setattr(subject, "load_neural_data", unexpected_waveform_setup)
    dataset = BrainTreebankSubjectTrialBenchmarkDataset(
        subject,
        1,
        torch.float32,
        "speech",
        lite=False,
        output_indices=True,
        defer_selection=defer_selection,
    )
    start, end = dataset[0]["data"]
    assert end > start
    assert dataset.cache_window_from is None
    assert dataset.cache_window_to is None
    assert subject.neural_data_cache == {}
    assert subject.h5_files == {}


def test_missing_class_retains_empty_fold_metadata(subject):
    folds = generate_splits_within_session(
        subject, 1, "volume", lite=False, output_indices=True
    )
    # Monotonically increasing volume separates its low/high quartiles in time.
    assert [fold["temporal_split"]["fold_idx"] for fold in folds] == [0, 1]
    assert all(
        len(fold[f"{split}_dataset"]) == 0
        for fold in folds
        for split in ("train", "val", "test")
    )


def test_standalone_dataset_keeps_historical_interleaving(subject):
    ds = BrainTreebankSubjectTrialBenchmarkDataset(
        subject,
        1,
        torch.float32,
        "speech",
        lite=False,
        output_indices=True,
        max_samples=20,
    )
    assert [ds[i]["label"] for i in range(6)] == [1, 0, 1, 0, 1, 0]
    assert len(ds) == 20
    starts, ends, labels = ds.get_window_bounds()
    assert [(int(a), int(b)) for a, b in zip(starts, ends)] == [
        item["data"] for item in ds
    ]
    assert labels.tolist() == [item["label"] for item in ds]


@pytest.mark.parametrize(
    "options",
    [
        dict(context_width_s=0.5),
        dict(context_width_s=float("nan")),
        dict(gap_s=-1),
        dict(gap_s=100),
        dict(random_seed=True),
        dict(max_samples=0),
    ],
)
def test_invalid_temporal_parameters(options):
    with pytest.raises(ValueError):
        temporal_fold_indices(
            np.array([1, 3]),
            np.array([3, 5]),
            np.array([0, 1]),
            recording_n_samples=100,
            sampling_rate=2,
            n_classes=2,
            **options,
        )


def test_exact_boundary_and_odd_sample_gap():
    starts = np.arange(100, dtype=np.int64)
    folds = temporal_fold_indices(
        starts,
        starts + 1,
        starts % 2,
        recording_n_samples=100,
        sampling_rate=1,
        n_classes=2,
        gap_s=1,
    )
    for fold in folds:
        blocks = sorted(fold["temporal_split"]["blocks_samples"].values())
        assert blocks[1][0] - blocks[0][1] == blocks[2][0] - blocks[1][1] == 1
        for split, indices in fold["indices"].items():
            lo, hi = fold["temporal_split"]["blocks_samples"][split]
            assert np.all(starts[indices] >= lo) and np.all(starts[indices] + 1 <= hi)


def test_pre_onset_window_protects_its_full_support(subject):
    folds = generate_splits_within_session(
        subject,
        1,
        "speech",
        lite=False,
        output_indices=True,
        start_neural_data_before_word_onset=1024,
        end_neural_data_after_word_onset=3072,
    )
    for fold in folds:
        assert fold["temporal_split"]["target_window_samples"] == 4096
        for split in ("train", "val", "test"):
            lo, hi = fold["temporal_split"]["blocks_samples"][split]
            assert all(
                lo <= item["data"][0] < item["data"][1] <= hi
                for item in fold[f"{split}_dataset"]
            )


def test_nano_budget_takes_precedence_over_lite(subject, monkeypatch):
    import neuroprobe.train_test_splits as split_module

    monkeypatch.setattr(split_module, "NEUROPROBE_LITE_MAX_SAMPLES", 80)
    monkeypatch.setattr(split_module, "NEUROPROBE_NANO_MAX_SAMPLES", 40)
    fold = generate_splits_within_session(
        subject, 1, "speech", lite=True, nano=True, output_indices=True
    )[0]
    assert [len(fold[f"{s}_dataset"]) for s in ("train", "val", "test")] == [20, 10, 10]


def test_odd_half_budget_keeps_historical_remainder_in_test(subject):
    folds = generate_splits_within_session(
        subject, 1, "speech", lite=False, output_indices=True, max_samples=30
    )
    for fold in folds:
        assert [len(fold[f"{s}_dataset"]) for s in ("train", "val", "test")] == [
            14,
            6,
            8,
        ]


def test_window_notch_does_not_depend_on_heldout_signal(subject, tmp_path):
    from examples.eval_utils import remove_line_noise

    fold = generate_splits_within_session(subject, 1, "speech", lite=False)[0]
    before = remove_line_noise(fold["train_dataset"][0]["data"].numpy())
    subject.clear_neural_data_cache()
    with h5py.File(tmp_path / "sub_1_trial001.h5", "r+") as handle:
        handle["data/electrode_0"][: 25 * 2048] = -1e5
    after = remove_line_noise(fold["train_dataset"][0]["data"].numpy())
    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("sample_dtype", [np.uint32, np.uint64])
def test_unsigned_support_exclusion_matches_signed_indices(sample_dtype):
    kwargs = dict(
        recording_n_samples=100, sampling_rate=1, n_classes=2, context_width_s=15
    )
    labels = np.array([0, 1])
    signed = temporal_fold_indices(
        np.array([0, 10]), np.array([1, 11]), labels, **kwargs
    )
    unsigned = temporal_fold_indices(
        np.array([0, 10], dtype=sample_dtype),
        np.array([1, 11], dtype=sample_dtype),
        labels,
        **kwargs,
    )
    assert signed[1]["indices"]["train"].size == 0
    for left, right in zip(signed, unsigned):
        assert left["temporal_split"] == right["temporal_split"]
        for split in ("train", "val", "test"):
            np.testing.assert_array_equal(
                left["indices"][split], right["indices"][split]
            )


def test_public_temporal_generator_accepts_large_selection_seed(subject):
    seed = 2**32 + 17
    kwargs = dict(lite=False, output_indices=True, max_samples=40, random_seed=seed)
    folds = generate_splits_within_session(subject, 1, "speech", **kwargs)
    repeated = generate_splits_within_session(subject, 1, "speech", **kwargs)
    for fold, repeat in zip(folds, repeated):
        assert fold["temporal_split"]["balance_seed"] == seed
        for split in ("train", "val", "test"):
            selected = fold[f"{split}_dataset"].indices
            assert len(selected) > 0
            np.testing.assert_array_equal(selected, repeat[f"{split}_dataset"].indices)
    assert subject.neural_data_cache == {}
    assert subject.h5_files == {}
