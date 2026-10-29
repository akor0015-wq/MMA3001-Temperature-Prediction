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


def test_duplicate_timestamps_use_one_median_value():
    """
    disc: check median for duplicate times

    input:
    samples(pd.DataFrame): repeated timestamps and temps

    output:
    None
    """
    samples = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:10:00Z",
        ]),
        "temp_c": [20.0, 20.0, 24.0, 21.0],
    })

    series, audit = audit_and_regularise(samples)

    assert not series.index.has_duplicates
    assert len(series) == 2

    # 20, 20, 24 median = 20
    assert series.iloc[0] == 20.0

    assert audit["duplicate_timestamp_rows"] == 2
    assert audit["conflicting_timestamp_groups"] == 1


def test_missing_grid_intervals_remain_missing():
    """
    disc: check missing times stay NaN

    input:
    samples(pd.DataFrame): readings with a missing time

    output:
    None
    """
    samples = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:10:00Z",
            "2024-01-01T00:30:00Z",
        ]),
        "temp_c": [20, 21, 23],
    })

    series, audit = audit_and_regularise(samples)

    assert len(series) == 4

    # 00:20 missing -> NaN
    assert pd.isna(series.loc[pd.Timestamp("2024-01-01T00:20:00Z")])

    assert audit["missing_grid_points"] == 1


def test_missing_measurements_excluded_from_forecast_examples():
    """
    disc: check rows with missing data are skipped

    input:
    times(pd.DatetimeIndex): 200 timestamps
    series(pd.Series): data with a missing temp

    output:
    None
    """
    times = pd.date_range("2024-01-01", periods=200, freq="10min", tz="UTC")
    series = pd.Series(np.arange(200, dtype=float), index=times)

    # one missing reading
    series.iloc[40] = np.nan

    examples = make_examples(series, 3)

    assert not examples.isna().any().any()

    # less examples if data mising
    assert len(examples) < len(make_examples(series.ffill(), 3))

    assert not examples.index.has_duplicates


def test_json_extracts_temperature_not_other_measures():
    """
    disc: check JSON gives only temp

    input:
    payload(list): temp and humidity data

    output:
    None
    """
    payload = [
        {
            "variable": {"name": "Humidity", "unit": "HR%"},
            "values": [{"time": 1000, "value": 40}],
        },
        {
            "variable": {"name": "Temperature", "unit": "°C"},
            "values": [{"time": 1700000000, "value": 23.7}],
        },
    ]

    # temp only, no humidity
    assert extract_temperature(json.dumps(payload)) == [(1700000000, 23.7)]


def test_duplicate_forecast_origins_are_rejected():
    """
    disc: check repeat times cause an error

    input:
    times(pd.DatetimeIndex): timestamps with a duplicate
    series(pd.Series): data with repeated times

    output:
    None
    """
    times = pd.date_range("2024-01-01", periods=200, freq="10min", tz="UTC")

    # add duplicate time
    times = times.insert(30, times[30])

    series = pd.Series(np.arange(201, dtype=float), index=times)

    with pytest.raises(ValueError, match="unique timestamp"):
        make_examples(series, 3)
