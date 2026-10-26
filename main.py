import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# paths from this file
BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "data" / "raw" / "5EnvSensor_MayToDec2024_180kRows.csv"
PROCESSED = BASE_DIR / "data" / "processed" / "room_temperature.csv"
RESULTS = BASE_DIR / "results"

# sensor location: B60 G.38 Compactus_upper
# unconditioned room? not confirmed
SENSOR_ID = "6012002000869"
TIMEZONE = "Australia/Melbourne"  # check Melbourne timezone
FREQ = "10min"                   # 10min inteval from audit
STEPS_PER_HOUR = 6                # 6 steps per hour
HORIZONS = (1, 3, 6, 12)
ALL_LAGS = (0, 1, 2, 3, 6)
TIME_FEATURES = ("hour_sin", "hour_cos", "year_sin", "year_cos")
FEATURE_SETS = {
    "short_lags": ["lag_0", "lag_1", "lag_2"],
    "long_lags": [f"lag_{lag}" for lag in ALL_LAGS],
    "long_plus_time": [f"lag_{lag}" for lag in ALL_LAGS] + list(TIME_FEATURES),
}


def progress(message: str) -> None:
    """
    disc: show a progress message

    input:
    message(str): text to print

    output:
    None
    """
    print(message, flush=True)


def resolve_raw_csv(path: Path = RAW_CSV) -> Path:
    """
    disc: find the raw CSV file

    input:
    path(Path): csv file path

    output:
    Path: raw CSV path
    """
    if path.is_file():
        return path
    raw_folder = BASE_DIR / "data" / "raw"
    possible = sorted(raw_folder.glob("5EnvSensor*.csv"))
    alternative = raw_folder / "environment.csv"
    if alternative.is_file():
        possible.append(alternative)
    if len(possible) == 1:
        progress(f"[FILE] Using detected CSV: {possible[0].name}")
        return possible[0]
    found = ", ".join(p.name for p in possible) if possible else "none"
    raise FileNotFoundError(
        f"CSV not found: {path}\n"
        f"Put the original environmental CSV in: {raw_folder}\n"
        f"Detected candidate CSV files: {found}"
    )


def extract_temperature(json_text: str) -> list[tuple[int, float]]:
    """
    disc: get temperature readings from JSON

    input:
    json_text(str): sensor JSON data

    output:
    list: pairs of unix time and temp
    """
    payload = json.loads(json_text)
    if not isinstance(payload, list):
        raise ValueError("Expected a list of sensor variable objects")
    readings = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        variable = entry.get("variable") or {}
        if not isinstance(variable, dict):
            continue
        if str(variable.get("name", "")).strip().casefold() != "temperature":
            continue
        unit = str(variable.get("unit", "")).strip()
        if unit not in ("°C", "℃", "deg C", "C"):
            raise ValueError(f"Unexpected temperature unit {unit!r}; expected Celsius")
        for reading in entry.get("values") or []:
            if not isinstance(reading, dict):
                continue
            try:
                timestamp = int(reading["time"])
                temp_c = float(reading["value"])
            except (KeyError, ValueError, TypeError, OverflowError):
                continue
            if np.isfinite(temp_c):
                readings.append((timestamp, temp_c))
    return readings


def inspect_csv(path: Path = RAW_CSV, nrows: int = 8) -> pd.DataFrame:
    """
    disc: show CSV columns and sample data

    input:
    path(Path): raw CSV location
    nrows(int): number of rows to show

    output:
    pd.DataFrame: sample rows
    """
    path = resolve_raw_csv(path)
    progress(f"[INSPECT 1/3] Opening {path}")
    sample = pd.read_csv(path, nrows=nrows, dtype={"sensorid": "string"})
    progress(f"[INSPECT 2/3] Read {len(sample)} rows")
    progress(f"[INSPECT] CSV columns: {sample.columns.tolist()}")
    cols = [name for name in ("id", "sensorid", "devicetype", "status", "createdate") if name in sample]
    print("\nSample identifiers (omitting the very long jsondata column):")
    print(sample[cols].to_string(index=False))
    if "jsondata" in sample.columns:
        for item in sample["jsondata"].dropna():
            try:
                readings = extract_temperature(item)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                progress(f"[INSPECT] Sample JSON could not be decoded: {exc}")
                continue
            if readings:
                seconds, temp = readings[0]
                time_utc = pd.to_datetime(seconds, unit="s", utc=True)
                progress(f"[INSPECT 3/3] Sample temperature: {temp:.2f} °C at {time_utc} (UTC)")
                break
    progress("[NEXT] Use --stage sensors to check which sensor IDs exist in the file.")
    return sample


def list_sensors(path: Path = RAW_CSV) -> pd.DataFrame:
    """
    disc: count sensors in the CSV

    input:
    path(Path): raw CSV location

    output:
    pd.DataFrame: sensor ids and counts
    """
    path = resolve_raw_csv(path)
    progress("[SENSORS] Scanning sensor IDs in CSV chunks...")
    counts = Counter()
    total = 0
    for i, part in enumerate(pd.read_csv(path, usecols=["sensorid"],
                                         dtype={"sensorid": "string"}, chunksize=50_000), 1):
        counts.update(part["sensorid"].dropna().str.strip().tolist())
        total += len(part)
        progress(f"[SENSORS] Chunk {i}: {total:,} rows scanned")
    table = pd.DataFrame(counts.items(), columns=["sensorid", "row_count"])
    table = table.sort_values("row_count", ascending=False).reset_index(drop=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "available_sensors.csv", index=False)
    print("\nAvailable sensors (top 30):\n", table.head(30).to_string(index=False))
    progress(f"[SENSORS] Requested ID {SENSOR_ID}: {counts.get(SENSOR_ID, 0):,} rows")
    progress("[SENSORS] Sensor inventory saved to results/available_sensors.csv")
    return table


def load_sensor(path: Path = RAW_CSV, sensor_id: str = SENSOR_ID) -> pd.DataFrame:
    """
    disc: load temp readings for one sensor

    input:
    path(Path): raw CSV location
    sensor_id(str): sensor to use

    output:
    pd.DataFrame: time and temp readings
    """
    path = resolve_raw_csv(path)
    progress(f"[LOAD 1/4] Extracting temperature for sensor {sensor_id}")
    progress("[LOAD 2/4] Reading only sensorid, devicetype and jsondata in chunks...")
    records = []
    total = matched = malformed = without_temperature = 0
    for i, chunk in enumerate(pd.read_csv(
        path, usecols=["sensorid", "devicetype", "jsondata"],
        dtype={"sensorid": "string", "devicetype": "string"},
        chunksize=50_000,
    ), 1):
        total += len(chunk)
        selected = chunk.loc[
            chunk["sensorid"].str.strip().eq(str(sensor_id)).fillna(False)
            & chunk["devicetype"].str.strip().str.casefold().eq("env").fillna(False)
        ]
        matched += len(selected)
        for item in selected["jsondata"]:
            try:
                readings = extract_temperature(item)
            except (ValueError, TypeError, json.JSONDecodeError):
                malformed += 1
                continue
            if not readings:
                without_temperature += 1
            records.extend(readings)
        progress(f"[LOAD] Chunk {i}: scanned {total:,}; matching rows {matched:,}; "
                 f"temperature observations {len(records):,}")
    progress(f"[LOAD 3/4] CSV scan finished: {total:,} rows, {matched:,} matching sensor rows")
    if not records:
        raise ValueError(
            f"No usable temperature observations for sensor {sensor_id}. "
            "Run --stage sensors to inspect available IDs; verify the room location."
        )
    data = pd.DataFrame(records, columns=["timestamp_unix", "temp_c"])
    data["timestamp"] = pd.to_datetime(data["timestamp_unix"], unit="s", utc=True,
                                        errors="coerce")
    data = data[["timestamp", "temp_c"]]
    progress(f"[LOAD 4/4] Extracted {len(data):,} readings. Bad payloads: {malformed:,}; "
             f"rows without temperature: {without_temperature:,}")
    progress(f"[LOAD] Timestamp range (UTC): {data.timestamp.min()} to {data.timestamp.max()}")
    return data


def audit_and_regularise(raw: pd.DataFrame, freq: str = FREQ) -> tuple[pd.Series, dict]:
    """
    disc: remove repeat times and make time grid

    input:
    raw(pd.DataFrame): raw temp data
    freq(str): time step size

    output:
    tuple: cleaned series and audit data
    """
    progress(f"[AUDIT 1/4] Checking {len(raw):,} raw temperature readings...")
    data = raw.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data["temp_c"] = pd.to_numeric(data["temp_c"], errors="coerce")
    bad_times = int(data["timestamp"].isna().sum())
    invalid_temp = int((~np.isfinite(data["temp_c"])).sum())
    data = data.loc[data["timestamp"].notna() & np.isfinite(data["temp_c"])].copy()
    if data.empty:
        raise ValueError("No valid timestamp/temperature pairs")
    data = data.sort_values("timestamp")

    # check duplicate timestmaps first
    grouped = data.groupby("timestamp")["temp_c"]
    count_by_time = grouped.size()
    unique_values_by_time = grouped.nunique()
    duplicate_rows = int((count_by_time - 1).sum())
    repeated_times = int((count_by_time > 1).sum())
    conflicting_times = int((unique_values_by_time > 1).sum())
    unique_timestamps = len(count_by_time)
    progress(f"[AUDIT 2/4] Unique measurement times: {unique_timestamps:,}; "
             f"extra duplicate rows: {duplicate_rows:,}; "
             f"conflicting timestamps: {conflicting_times:,}")

    # 1 value per unique time
    one_per_timestamp = grouped.median().sort_index()
    if one_per_timestamp.index.has_duplicates:
        raise AssertionError("Duplicate timestamps survived timestamp consolidation")

    diffs = one_per_timestamp.index.to_series().diff().dropna()
    median_interval_minutes = (
        float(diffs.median().total_seconds() / 60)
        if not diffs.empty else float("nan")
    )

    # 10min bins, gaps stay NaN
    series = one_per_timestamp.resample(freq).median().rename("temp_c")
    if series.index.has_duplicates:
        raise AssertionError("The regularized time grid contains duplicate timestamps")
    report = {
        "raw_measurements": len(raw),
        "bad_timestamps": bad_times,
        "invalid_temperatures": invalid_temp,
        "valid_measurements": len(data),
        "unique_measurement_timestamps": unique_timestamps,
        "duplicate_timestamp_rows": duplicate_rows,
        "repeated_timestamp_groups": repeated_times,
        "conflicting_timestamp_groups": conflicting_times,
        "duplicate_resolution": "median per exact timestamp, then median per 10-minute bin",
        "missing_data_handling": "leave NaN; omit incomplete training/evaluation examples",
        "median_unique_interval_minutes": round(median_interval_minutes, 3),
        "grid_points": len(series),
        "observed_grid_points": int(series.notna().sum()),
        "missing_grid_points": int(series.isna().sum()),
        "missing_grid_fraction": float(series.isna().mean()),
        "first_timestamp_utc": str(series.index.min()),
        "last_timestamp_utc": str(series.index.max()),
        "min_temp_c": float(series.min()),
        "max_temp_c": float(series.max()),
    }
    progress(f"[AUDIT 3/4] Built {report['grid_points']:,} unique {freq} time bins. "
             f"Missing: {report['missing_grid_fraction']:.1%}")
    progress(f"[AUDIT 4/4] Median unique raw interval: "
             f"{median_interval_minutes:.2f} min. No missing values filled.")
    if conflicting_times:
        progress("[WARNING] Some identical timestamps have conflicting values. "
                 "Check results/quality_conflicting_timestamps.csv")
    if not (8 <= median_interval_minutes <= 12):
        progress("[WARNING] Median sensor interval is not near 10 min. Verify sampling.")
    return series, report


def save_processed(series: pd.Series, path: Path = PROCESSED) -> None:
    """
    disc: save cleaned data to CSV

    input:
    series(pd.Series): clean temp readings
    path(Path): where to save

    output:
    None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame().to_csv(path, index_label="timestamp")
    progress(f"[SAVE] Wrote {len(series):,} grid rows to {path}")


def load_processed(path: Path = PROCESSED) -> pd.Series:
    """
    disc: open the cleaned CSV

    input:
    path(Path): clean CSV location

    output:
    pd.Series: cleaned temps
    """
    progress(f"[OPEN] Loading processed series from {path}")
    if not path.exists():
        raise FileNotFoundError("Processed file missing. Run --stage prepare first.")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    series = pd.Series(pd.to_numeric(df["temp_c"], errors="coerce").to_numpy(),
                       index=pd.DatetimeIndex(df["timestamp"]), name="temp_c")
    if series.index.has_duplicates:
        raise ValueError("The processed CSV has duplicate timestamps: rerun --stage prepare")
    progress(f"[OPEN] Loaded {len(series):,} unique grid timestamps")
    return series.sort_index()


def split_chronologically(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    disc: split data 70/15/15 by time

    input:
    series(pd.Series): temp data in time order

    output:
    tuple: train, validation and test sets
    """
    n = len(series)
    if n < 500:
        raise ValueError("Time series is too short for 12-hour forecasts and 70/15/15 split")
    a, b = int(n * 0.70), int(n * 0.85)
    train, valid, test = series.iloc[:a], series.iloc[a:b], series.iloc[b:]
    progress(f"[SPLIT] Train {len(train):,} | Validation {len(valid):,} | Test {len(test):,}")
    return train, valid, test


def make_examples(series: pd.Series, horizon_h: int) -> pd.DataFrame:
    """
    disc: make lag features and future targets

    input:
    series(pd.Series): temp data
    horizon_h(int): hours to predict ahead

    output:
    pd.DataFrame: input features and targets
    """
    if horizon_h not in HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon_h}")
    if series.index.has_duplicates:
        raise ValueError("Each forecast origin must have a unique timestamp")
    rows = pd.DataFrame(index=series.index)
    for lag in ALL_LAGS:
        rows[f"lag_{lag}"] = series.shift(lag)
    try:
        local_index = series.index.tz_convert(TIMEZONE)
    except Exception as exc:
        raise RuntimeError("Cannot convert to Melbourne time. On Windows: pip install tzdata") from exc
    hours = local_index.hour + local_index.minute / 60
    year_length = np.where(local_index.is_leap_year, 366, 365)
    annual_phase = 2 * np.pi * (local_index.dayofyear - 1) / year_length
    rows["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    rows["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    rows["year_sin"] = np.sin(annual_phase)
    rows["year_cos"] = np.cos(annual_phase)
    rows["target"] = series.shift(-horizon_h * STEPS_PER_HOUR)
    clean = rows.dropna()
    progress(f"[FEATURES] {horizon_h:>2}h ahead: {len(clean):,} usable examples / "
             f"{len(rows):,} grid points")
    return clean


def calculate_metrics(actual, predicted) -> dict:
    """
    disc: calculate MAE, RMSE and R2

    input:
    actual(list or array): real values
    predicted(list or array): model values

    output:
    dict: MAE, RMSE and R2
    """
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or len(y) == 0:
        raise ValueError("Actual and predicted must be 1-D, nonempty and equal length")
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 0 else float("nan"),
    }


def fit_and_evaluate(train_rows: pd.DataFrame, eval_rows: pd.DataFrame,
                     features: list[str]) -> tuple[dict, pd.DataFrame]:
    """
    disc: train model and check results

    input:
    train_rows(pd.DataFrame): training samples
    eval_rows(pd.DataFrame): samples to score
    features(list[str]): columns used by model

    output:
    tuple: scores and predictions
    """
    progress(f"[FIT] Training linear regression with {len(features)} input features...")
    t0 = perf_counter()
    model = LinearRegression().fit(train_rows[features], train_rows["target"])
    fit_s = perf_counter() - t0
    t0 = perf_counter()
    predicted = model.predict(eval_rows[features])
    predict_s = perf_counter() - t0
    comparison = pd.DataFrame({
        "actual": eval_rows["target"],
        "persistence": eval_rows["lag_0"],
        "regression": predicted,
    }, index=eval_rows.index)
    regression = calculate_metrics(comparison.actual, comparison.regression)
    baseline = calculate_metrics(comparison.actual, comparison.persistence)
    progress(f"[SCORE] Linear MAE={regression['mae']:.3f} °C | "
             f"Persistence MAE={baseline['mae']:.3f} °C | train={fit_s:.3f}s")
    return {
        "regression": regression, "persistence": baseline,
        "fit_seconds": fit_s, "predict_seconds": predict_s,
        "n_evaluated": len(comparison),
    }, comparison


def validate_models(series: pd.Series) -> pd.DataFrame:
    """
    disc: pick model from validation results

    input:
    series(pd.Series): clean temperature series

    output:
    pd.DataFrame: best model for each horizon
    """
    train, valid, _ = split_chronologically(series)
    records = []
    for horizon in HORIZONS:
        progress(f"\n[VALIDATE] ===== Forecast horizon {horizon} hour(s) =====")
        tr = make_examples(train, horizon)
        va = make_examples(valid, horizon)
        if min(len(tr), len(va)) < 20:
            raise ValueError(f"Too few examples at {horizon}h; inspect missing data")
        for label, features in FEATURE_SETS.items():
            progress(f"[VALIDATE] Configuration: {label}")
            scores, _ = fit_and_evaluate(tr, va, features)
            records.append({
                "horizon_h": horizon, "configuration": label,
                "validation_mae": scores["regression"]["mae"],
                "validation_rmse": scores["regression"]["rmse"],
                "validation_r2": scores["regression"]["r2"],
                "persistence_mae": scores["persistence"]["mae"],
                "fit_seconds": scores["fit_seconds"],
                "predict_seconds": scores["predict_seconds"],
            })
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(RESULTS / "validation_scores.csv", index=False)
    best = (table.sort_values(["horizon_h", "validation_mae", "configuration"])
            .drop_duplicates("horizon_h"))
    best[["horizon_h", "configuration", "validation_mae"]].to_csv(
        RESULTS / "selected_models.csv", index=False
    )
    progress("[VALIDATE] Selected models (choose/lock BEFORE final test):")
    print(best[["horizon_h", "configuration", "validation_mae"]].to_string(index=False))
    progress("[VALIDATE] Saved validation_scores.csv and selected_models.csv")
    return best


def final_test_models(series: pd.Series) -> pd.DataFrame:
    """
    disc: test chosen model on held out data

    input:
    series(pd.Series): clean temperature series

    output:
    pd.DataFrame: final model scores
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = RESULTS / "test_scores.csv"
    if destination.exists():
        raise FileExistsError(
            "test_scores.csv already exists. Final test is intentionally locked; "
            "do not retune from test results."
        )
    choice_file = RESULTS / "selected_models.csv"
    if not choice_file.exists():
        raise FileNotFoundError("Run --stage validate before --stage test")
    selected = pd.read_csv(choice_file)
    train, valid, test = split_chronologically(series)
    development = pd.concat([train, valid])
    records = []
    pending_predictions = {}
    for row in selected.itertuples(index=False):
        horizon = int(row.horizon_h)
        label = str(row.configuration)
        progress(f"\n[TEST] Horizon {horizon}h; selected configuration {label}")
        if label not in FEATURE_SETS:
            raise ValueError(f"Unknown configuration {label}")
        tr = make_examples(development, horizon)
        te = make_examples(test, horizon)
        if min(len(tr), len(te)) < 20:
            raise ValueError(f"Too few examples in held-out period at {horizon}h")
        scores, predictions = fit_and_evaluate(tr, te, FEATURE_SETS[label])
        records.append({
            "horizon_h": horizon, "configuration": label,
            "test_mae": scores["regression"]["mae"],
            "test_rmse": scores["regression"]["rmse"],
            "test_r2": scores["regression"]["r2"],
            "baseline_mae": scores["persistence"]["mae"],
            "baseline_rmse": scores["persistence"]["rmse"],
            "baseline_r2": scores["persistence"]["r2"],
            "fit_seconds": scores["fit_seconds"],
            "predict_seconds": scores["predict_seconds"],
            "test_rows": scores["n_evaluated"],
        })
        pending_predictions[horizon] = predictions
    results = pd.DataFrame(records).sort_values("horizon_h")
    for horizon, prediction in pending_predictions.items():
        prediction.to_csv(RESULTS / f"predictions_{horizon}h.csv",
                          index_label="origin_timestamp")
    results.to_csv(destination, index=False)
    progress("\n[TEST] Final held-out results saved to results/test_scores.csv")
    print(results.to_string(index=False))
    return results


def plot_results(series: pd.Series) -> None:
    """
    disc: save graphs for the report

    input:
    series(pd.Series): clean temperature series

    output:
    None
    """
    figdir = RESULTS / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    if not (RESULTS / "test_scores.csv").exists():
        raise FileNotFoundError("Run --stage test before --stage plot")
    progress("[PLOT 1/3] Temperature history")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    series.plot(ax=ax)
    ax.set(title="Indoor temperature history", ylabel="Temperature (°C)", xlabel="Time (UTC)")
    fig.tight_layout()
    fig.savefig(figdir / "temperature_history.png", dpi=180)
    plt.close(fig)

    progress("[PLOT 2/3] MAE against forecast horizon")
    scores = pd.read_csv(RESULTS / "test_scores.csv").sort_values("horizon_h")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scores.horizon_h, scores.baseline_mae, marker="o", label="Persistence")
    ax.plot(scores.horizon_h, scores.test_mae, marker="o", label="Linear regression")
    ax.set(xlabel="Forecast horizon (hours)", ylabel="MAE (°C)",
           title="Forecast error versus prediction horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "mae_by_horizon.png", dpi=180)
    plt.close(fig)

    progress("[PLOT 3/3] Actual versus predicted 3-hour temperature")
    p = pd.read_csv(RESULTS / "predictions_3h.csv", parse_dates=["origin_timestamp"])
    subset = p.head(150)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(subset.origin_timestamp, subset.actual, label="Actual target")
    ax.plot(subset.origin_timestamp, subset.regression, label="Linear forecast")
    ax.set(xlabel="Forecast origin (UTC)", ylabel="Future temperature (°C)",
           title="3-hour forecasts (target occurs 3 hours after origin)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "predictions_3h.png", dpi=180)
    plt.close(fig)
    progress(f"[PLOT] All graphs saved to {figdir}")


def main() -> None:
    """
    disc: run the stage picked in terminal

    input:
    None

    output:
    None
    """
    parser = argparse.ArgumentParser(description="MMA3001 temperature forecasting")
    parser.add_argument("--stage", choices=('inspect', 'sensors', 'prepare', 'validate', 'test', 'plot'),
                        default="inspect", help="Which project task to execute")
    parser.add_argument("--sensor-id", default=SENSOR_ID,
                        help="Sensor ID for prepare; choose only after verifying room")
    args = parser.parse_args()
    started = perf_counter()
    progress("=" * 66)
    progress(f"[START] MMA3001 temperature forecasting | stage: {args.stage}")
    progress("=" * 66)
    try:
        if args.stage == "inspect":
            inspect_csv()
        elif args.stage == "sensors":
            list_sensors()
        elif args.stage == "prepare":
            raw = load_sensor(sensor_id=args.sensor_id)
            series, report = audit_and_regularise(raw)
            report["selected_sensor_id"] = args.sensor_id
            report["raw_csv_filename"] = resolve_raw_csv().name
            RESULTS.mkdir(parents=True, exist_ok=True)
            pd.Series(report).to_csv(RESULTS / "data_audit.csv", header=["value"])
            save_processed(series)
            progress("[PREPARE] Audit saved in results/")
        else:
            series = load_processed()
            if args.stage == "validate":
                validate_models(series)
            elif args.stage == "test":
                final_test_models(series)
            elif args.stage == "plot":
                plot_results(series)
    except Exception as exc:
        progress(f"[ERROR] {type(exc).__name__}: {exc}")
        raise
    progress(f"[DONE] Stage '{args.stage}' finished in {perf_counter() - started:.2f} seconds")


if __name__ == "__main__":
    main()
