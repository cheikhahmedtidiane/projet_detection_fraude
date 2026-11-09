"""Tests unitaires du module src.split."""

import pandas as pd
import pytest

from src.split import (
    assign_split,
    check_no_temporal_overlap,
    filter_fraud_types,
    load_split,
    save_splits,
    step_to_day,
    summarize_splits,
    temporal_split,
)


@pytest.fixture
def df():
    # Une transaction par jour, du jour 0 au jour 20 (step = jour * 24 + 1)
    days = list(range(21))
    return pd.DataFrame(
        {
            "step": [d * 24 + 1 for d in days],
            "type": pd.Categorical(["TRANSFER", "CASH_OUT", "PAYMENT"] * 7),
            "isFraud": [1 if d % 5 == 0 else 0 for d in days],
        }
    )


def test_step_to_day():
    assert step_to_day(pd.Series([1, 23, 24, 47, 48])).tolist() == [0, 0, 1, 1, 2]


def test_filter_fraud_types(df):
    out = filter_fraud_types(df)
    assert set(out["type"]) == {"TRANSFER", "CASH_OUT"}
    assert "PAYMENT" not in out["type"].cat.categories


def test_assign_split_boundaries(df):
    labels = assign_split(df["step"])
    by_day = dict(zip(step_to_day(df["step"]), labels))
    assert by_day[0] == by_day[12] == "train"
    assert by_day[13] == by_day[14] == "val"
    assert by_day[15] == by_day[16] == "test"
    assert by_day[17] == by_day[20] == "drift"


def test_assign_split_invalid_bounds(df):
    with pytest.raises(ValueError):
        assign_split(df["step"], train_last_day=10, val_last_day=5, test_last_day=12)


def test_temporal_split_is_partition_without_overlap(df):
    splits = temporal_split(df)
    assert list(splits) == ["train", "val", "test", "drift"]
    assert sum(len(p) for p in splits.values()) == len(df)
    check_no_temporal_overlap(splits)


def test_check_no_temporal_overlap_detects_overlap(df):
    splits = temporal_split(df)
    splits["val"] = pd.concat([splits["val"], splits["train"].tail(1)])
    with pytest.raises(ValueError, match="Chevauchement"):
        check_no_temporal_overlap(splits)


def test_summarize_splits(df):
    summary = summarize_splits(temporal_split(df))
    assert summary.loc["train", "n_transactions"] == 13
    assert summary.loc["train", "n_fraudes"] == 3  # jours 0, 5, 10
    assert summary["n_transactions"].sum() == len(df)


def test_save_and_load_splits(df, tmp_path):
    splits = temporal_split(df)
    save_splits(splits, tmp_path)
    loaded = load_split("val", tmp_path)
    pd.testing.assert_frame_equal(loaded, splits["val"].reset_index(drop=True))


def test_load_split_unknown_name(tmp_path):
    with pytest.raises(ValueError):
        load_split("holdout", tmp_path)
