import json
import numpy as np
import pandas as pd
import pytest

from main import (
    audit_and_regularise,
    calculate_metrics,
    extract_temperature,
    make_examples,
    split_chronologically,
)


def test_perfect_prediction():
    """
    disc: check perfect predictions give zero error

    input:
    actual(list): real temps
    predicted(list): same predicted temps

    output:
    None
    """
    actual = [20, 21, 22, 23]
    predicted = [20, 21, 22, 23]

    result = calculate_metrics(actual, predicted)

    assert result["mae"] == 0
    assert result["rmse"] == 0
    assert result["r2"] == 1


def test_known_prediction_error():
    """
    disc: check MAE and RMSE values

    input:
    actual(list): real temps
    predicted(list): predicted temps

    output:
    None
    """
    actual = [20, 22]
    predicted = [21, 20]

    result = calculate_metrics(actual, predicted)

    assert np.isclose(result["mae"], 1.5)
    assert np.isclose(result["rmse"], np.sqrt(2.5))


def test_chronological_split():
    """
    disc: check 70/15/15 split and time order

    input:
    times(pd.DatetimeIndex): 1000 timestamps
    series(pd.Series): test readings

    output:
    None
    """
    times = pd.date_range("2024-01-01", periods=1000, freq="10min", tz="UTC")
    series = pd.Series(np.arange(1000), index=times)

    train, valid, test = split_chronologically(series)

    assert (len(train), len(valid), len(test)) == (700, 150, 150)

    # check time order
    assert train.index.max() < valid.index.min() < test.index.min()


def test_forecast_target_alignment():
    """
    disc: check 3h target and lag times

    input:
    times(pd.DatetimeIndex): 200 timestamps
    series(pd.Series): test readings

    output:
    None
    """
    times = pd.date_range("2024-01-01", periods=200, freq="10min", tz="UTC")
    series = pd.Series(np.arange(200, dtype=float), index=times)

    examples = make_examples(series, 3)
    origin = examples.index[0]

    # target 3h ahead
    assert examples.loc[origin, "target"] == series.loc[
        origin + pd.Timedelta(hours=3)
    ]

    assert examples.loc[origin, "lag_0"] == series.loc[origin]

    assert examples.loc[origin, "lag_1"] == series.loc[
        origin - pd.Timedelta(minutes=10)
    ]
