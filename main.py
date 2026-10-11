"""MMA3001: short-term temperature forecasts for one selected room.

All modelling code is in this single file. Inspect your real CSV and update
configuration constants before running ``--stage prepare``.
"""


from __future__ import annotations

print("Start")

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Finished Import")

# DAY 1-2: Edit these after inspecting the real CSV header.
RAW_CSV = Path("data/raw/environment.csv")

PROCESSED = Path("data/processed/room_temperature.csv")
RESULTS = Path("results")
SENSOR_ID = "6012003000424"  # Candidate only; confirm an unconditioned room.
SENSOR_COL = "CHANGE_ME_SENSOR_COLUMN"
TIMESTAMP_COL = "CHANGE_ME_TIMESTAMP_COLUMN"
TEMPERATURE_COL = "CHANGE_ME_TEMPERATURE_COLUMN"
# For long-form data: provide both fields below and set TEMPERATURE_COL to
# the numeric reading/value column. Leave both None for wide-form data.
MEASUREMENT_TYPE_COL = None
TEMPERATURE_TYPE_NAME = None
FREQ = "10min"  # Verify from measurements before using.
STEPS_PER_HOUR = 6  # Valid only for a verified 10-minute grid.
HORIZONS = (1, 3, 6, 12)
ALL_LAGS = (0, 1, 2, 3, 6)
CALENDAR = ("hour_sin", "hour_cos", "year_sin", "year_cos")
FEATURE_SETS = {
    "short_lags": ["lag_0", "lag_1", "lag_2"],
    "long_lags": [f"lag_{x}" for x in ALL_LAGS],
    "long_plus_time": [f"lag_{x}" for x in ALL_LAGS] + list(CALENDAR),
}


def inspect_csv(path: Path = RAW_CSV, nrows: int = 8) -> pd.DataFrame:
    """Print the raw CSV header and example rows; return the small sample."""
    print(f"[INSPECT 1/2] Opening CSV: {path}", flush=True)
    sample = pd.read_csv(path, nrows=nrows)
    print(f"[INSPECT 2/2] Read {len(sample)} example rows.", flush=True)
    print("CSV columns:", sample.columns.tolist())
    print("\nFirst rows:\n", sample.to_string(index=False))
    print("\nDtypes:\n", sample.dtypes)
    return sample


def load_sensor(path: Path = RAW_CSV, sensor_id: str = SENSOR_ID) -> pd.DataFrame:
    """Filter one sensor with chunked reading; return timestamp/temperature rows.

    For long-form tables, set MEASUREMENT_TYPE_COL and TEMPERATURE_TYPE_NAME.
    Confirm sensor IDs, units and timestamp formats against the raw dataset.
    """
    print(f"[LOAD 1/4] Starting sensor extraction for ID {sensor_id}", flush=True)
    print(f"[LOAD] Source file: {path}", flush=True)
    required = [SENSOR_COL, TIMESTAMP_COL, TEMPERATURE_COL]
    if MEASUREMENT_TYPE_COL is not None:
        if TEMPERATURE_TYPE_NAME is None:
            raise ValueError("Set TEMPERATURE_TYPE_NAME for long-form data")
        required.append(MEASUREMENT_TYPE_COL)
    if any(name.startswith("CHANGE_ME_") for name in required):
        raise ValueError("Inspect the CSV and replace CHANGE_ME_* column settings")
    print("[LOAD 2/4] Column settings verified; reading CSV in 100,000-row chunks...", flush=True)
    chunks = []
    chunk_number = 0
    total_rows_scanned = 0
    matched_rows = 0
    for part in pd.read_csv(
        path, usecols=list(dict.fromkeys(required)),
        dtype={SENSOR_COL: "string"}, chunksize=100_000,
    ):
        chunk_number += 1
        total_rows_scanned += len(part)
        keep = part[SENSOR_COL].str.strip().eq(str(sensor_id)).fillna(False)
        part = part.loc[keep]
        if MEASUREMENT_TYPE_COL is not None:
            part = part.loc[
                part[MEASUREMENT_TYPE_COL].astype("string")
                .eq(str(TEMPERATURE_TYPE_NAME)).fillna(False)
            ]
        matched_rows += len(part)
        if chunk_number == 1 or chunk_number % 5 == 0:
            print(
                f"[LOAD] Chunk {chunk_number}: scanned {total_rows_scanned:,} rows; "
                f"matching temperature readings {matched_rows:,}",
                flush=True,
            )
        if not part.empty:
            chunks.append(part[[TIMESTAMP_COL, TEMPERATURE_COL]].copy())
    print(
        f"[LOAD 3/4] Finished scanning {total_rows_scanned:,} rows in "
        f"{chunk_number} chunks; matched {matched_rows:,} rows.",
        flush=True,
    )
    if not chunks:
        raise ValueError(f"No readings found for sensor {sensor_id}")
    data = pd.concat(chunks, ignore_index=True)
    data.columns = ["timestamp", "temp_c"]
    # For numeric Unix timestamps, replace this with the confirmed 'unit'.
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["temp_c"] = pd.to_numeric(data["temp_c"], errors="coerce")
    print(f"[LOAD 4/4] Parsed sensor data: {len(data):,} rows.", flush=True)
    return data


def audit_and_regularise(raw: pd.DataFrame, freq: str = FREQ) -> tuple[pd.Series, dict]:
    """Audit measurements and put them on a regular grid, without interpolation.

    Multiple readings at an identical timestamp are averaged only under the
    assumption that they are readings of the same physical quantity in deg C.
    """
    print(f"[AUDIT 1/3] Checking {len(raw):,} readings for missing or invalid values...", flush=True)
    data = raw.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["temp_c"] = pd.to_numeric(data["temp_c"], errors="coerce")
    bad_times = int(data["timestamp"].isna().sum())
    data = data.dropna(subset=["timestamp"])
    if data.empty:
        raise ValueError("No valid timestamps after parsing")
    duplicate_times = int(data["timestamp"].duplicated().sum())
    missing_raw = int(data["temp_c"].isna().sum())
    print(
        f"[AUDIT 2/3] Found {bad_times:,} bad timestamps, "
        f"{duplicate_times:,} duplicate timestamps and {missing_raw:,} missing temperatures.",
        flush=True,
    )
    print(f"[AUDIT] Building regular {freq} measurement grid...", flush=True)
    # A resample bin may contain >1 reading. Investigate irregular raw cadence.
    series = data.set_index("timestamp")["temp_c"].sort_index().resample(freq).mean()
    report = {
        "raw_rows": len(raw), "bad_timestamps": bad_times,
        "duplicate_timestamps": duplicate_times,
        "raw_missing_temperatures": missing_raw,
        "grid_points": len(series),
        "missing_grid_points": int(series.isna().sum()),
        "first_timestamp": str(series.index.min()),
        "last_timestamp": str(series.index.max()),
        "min_temp_c": float(series.min()),
        "max_temp_c": float(series.max()),
    }
    print(
        f"[AUDIT 3/3] Grid contains {len(series):,} time points; "
        f"{int(series.isna().sum()):,} missing grid points.",
        flush=True,
    )
    return series.rename("temp_c"), report


def save_processed(series: pd.Series, path: Path = PROCESSED) -> None:
    """Save the audited sensor-only series in a small CSV for fast reruns."""
    print(f"[SAVE] Saving processed data to: {path}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame().to_csv(path, index_label="timestamp")
    print(f"[SAVE] Finished saving {len(series):,} time points.", flush=True)


def load_processed(path: Path = PROCESSED) -> pd.Series:
    """Read the prepared room-temperature time series."""
    print(f"[DATA] Loading processed time series: {path}", flush=True)
    series = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")["temp_c"].sort_index()
    print(f"[DATA] Loaded {len(series):,} time points.", flush=True)
    return series


def split_chronologically(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split before target shifting: first 70%, next 15%, last 15%."""
    n = len(series)
    if n < 120:
        raise ValueError("Time series too short for this forecast setup")
    train_end, valid_end = int(0.70 * n), int(0.85 * n)
    print(
        f"[SPLIT] Training: {train_end:,}, validation: {valid_end - train_end:,}, "
        f"test: {n - valid_end:,} time points.",
        flush=True,
    )
    return series.iloc[:train_end], series.iloc[train_end:valid_end], series.iloc[valid_end:]


def make_examples(series: pd.Series, horizon_h: int) -> pd.DataFrame:
    """Build causal lag/time features and future target WITHIN one data split.

    Origin is the row index. Target is T(origin + horizon_h hours). No row
    crosses into another split because each split is processed separately.
    """
    print(f"[FEATURES] Generating {horizon_h}h forecast examples from {len(series):,} time points...", flush=True)
    if horizon_h not in HORIZONS:
        raise ValueError("Unsupported forecast horizon")
    rows = pd.DataFrame(index=series.index)
    for lag in ALL_LAGS:
        rows[f"lag_{lag}"] = series.shift(lag)
    hour = series.index.hour + series.index.minute / 60
    # 365.25 is a reasonable calendar-season approximation across leap years.
    day_of_year = series.index.dayofyear
    rows["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    rows["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    rows["year_sin"] = np.sin(2 * np.pi * (day_of_year - 1) / 365.25)
    rows["year_cos"] = np.cos(2 * np.pi * (day_of_year - 1) / 365.25)
    rows["target"] = series.shift(-horizon_h * STEPS_PER_HOUR)
    clean_rows = rows.dropna()
    print(f"[FEATURES] Complete: {len(clean_rows):,} usable examples.", flush=True)
    return clean_rows


def calculate_metrics(actual, prediction) -> dict:
    """Compute MAE/RMSE in deg C and unitless R-squared."""
    y = np.asarray(actual, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if y.shape != p.shape or len(y) == 0:
        raise ValueError("Actual and predicted arrays must match and be nonempty")
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 0 else float("nan"),
    }


def fit_and_evaluate(
    train_rows: pd.DataFrame, eval_rows: pd.DataFrame, feature_cols: list[str]
) -> tuple[dict, pd.DataFrame]:
    """Fit linear regression on train only; score it and persistence on eval."""
    print(
        f"[MODEL] Training linear regression: {len(train_rows):,} training examples, "
        f"{len(feature_cols)} features...",
        flush=True,
    )
    start = perf_counter()
    model = LinearRegression().fit(train_rows[feature_cols], train_rows["target"])
    fit_s = perf_counter() - start
    print(f"[MODEL] Training complete in {fit_s:.3f} s. Predicting {len(eval_rows):,} examples...", flush=True)
    start = perf_counter()
    predicted = model.predict(eval_rows[feature_cols])
    predict_s = perf_counter() - start
    print(f"[MODEL] Predictions ready in {predict_s:.3f} s. Calculating accuracy...", flush=True)
    comparison = pd.DataFrame({
        "actual": eval_rows["target"],
        "persistence": eval_rows["lag_0"],
        "regression": predicted,
    }, index=eval_rows.index)
    regression_metrics = calculate_metrics(comparison["actual"], comparison["regression"])
    persistence_metrics = calculate_metrics(comparison["actual"], comparison["persistence"])
    print(
        f"[MODEL] Done. Regression MAE = {regression_metrics['mae']:.4f} deg C; "
        f"persistence MAE = {persistence_metrics['mae']:.4f} deg C.",
        flush=True,
    )
    return {
        "regression": regression_metrics,
        "persistence": persistence_metrics,
        "fit_seconds": fit_s, "predict_seconds": predict_s,
        "n_evaluated": len(comparison),
    }, comparison


def validate_models(series: pd.Series) -> pd.DataFrame:
    """Compare configurations using validation only; save choices for each horizon."""
    print("[VALIDATE 1/3] Starting validation (test data stays untouched).", flush=True)
    train, valid, _ = split_chronologically(series)
    records = []
    for horizon in HORIZONS:
        print(f"[VALIDATE] Horizon: {horizon} hour(s).", flush=True)
        tr = make_examples(train, horizon)
        va = make_examples(valid, horizon)
        if min(len(tr), len(va)) < 20:
            raise ValueError(f"Insufficient complete rows at {horizon}h")
        for name, features in FEATURE_SETS.items():
            print(f"[VALIDATE] Evaluating configuration: {name}", flush=True)
            result, _ = fit_and_evaluate(tr, va, features)
            records.append({
                "horizon_h": horizon, "configuration": name,
                "validation_mae": result["regression"]["mae"],
                "validation_rmse": result["regression"]["rmse"],
                "validation_r2": result["regression"]["r2"],
                "persistence_mae": result["persistence"]["mae"],
                "fit_seconds": result["fit_seconds"],
                "predict_seconds": result["predict_seconds"],
            })
    print(f"[VALIDATE 2/3] Finished {len(records)} comparisons; saving results...", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(records)
    scores.to_csv(RESULTS / "validation_scores.csv", index=False)
    best = (scores.sort_values("validation_mae")
            .drop_duplicates("horizon_h").sort_values("horizon_h"))
    best[["horizon_h", "configuration", "validation_mae"]].to_csv(
        RESULTS / "selected_models.csv", index=False
    )
    print("Validation choices (review and lock before testing):\n", best.to_string(index=False))
    print("[VALIDATE 3/3] Saved validation_scores.csv and selected_models.csv.", flush=True)
    return best


def final_test_models(series: pd.Series) -> pd.DataFrame:
    """Run the held-out final test once using previously chosen configurations."""
    print("[TEST 1/3] Starting FINAL held-out test. Do not retune afterwards.", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = RESULTS / "test_scores.csv"
    if destination.exists():
        raise FileExistsError("Final test already exists; do not retune using its scores")
    chosen = pd.read_csv(RESULTS / "selected_models.csv")
    train, _, test = split_chronologically(series)
    records = []
    for row in chosen.itertuples(index=False):
        horizon = int(row.horizon_h)
        print(f"[TEST] Horizon: {horizon}h; selected model: {row.configuration}.", flush=True)
        features = FEATURE_SETS[row.configuration]
        tr = make_examples(train, horizon)
        te = make_examples(test, horizon)
        if len(te) < 20:
            raise ValueError(f"Insufficient complete held-out rows at {horizon}h")
        scores, predictions = fit_and_evaluate(tr, te, features)
        records.append({
            "horizon_h": horizon, "configuration": row.configuration,
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
        predictions.to_csv(RESULTS / f"predictions_{horizon}h.csv", index_label="origin_timestamp")
        print(f"[TEST] Saved predictions_{horizon}h.csv.", flush=True)
    print("[TEST 2/3] All horizons evaluated; saving final test results...", flush=True)
    out = pd.DataFrame(records).sort_values("horizon_h")
    out.to_csv(destination, index=False)
    print("FINAL TEST:\n", out.to_string(index=False))
    print("[TEST 3/3] Final evaluation saved to test_scores.csv.", flush=True)
    return out


def plot_results(series: pd.Series) -> None:
    """Save report-ready plots using real prepared data and final test outputs."""
    print("[PLOT 1/4] Creating report figures...", flush=True)
    figdir = RESULTS / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 3.5))
    series.plot()
    plt.ylabel("Indoor air temperature (deg C)")
    plt.xlabel("Timestamp")
    plt.title("Monash room temperature series")
    plt.tight_layout()
    plt.savefig(figdir / "temperature_history.png", dpi=180)
    plt.close()
    print("[PLOT 2/4] Saved temperature_history.png.", flush=True)
    scores = pd.read_csv(RESULTS / "test_scores.csv").sort_values("horizon_h")
    plt.figure(figsize=(7, 4))
    plt.plot(scores["horizon_h"], scores["baseline_mae"], marker="o", label="Persistence")
    plt.plot(scores["horizon_h"], scores["test_mae"], marker="o", label="Linear regression")
    plt.xlabel("Prediction horizon (hours)")
    plt.ylabel("Test MAE (deg C)")
    plt.title("Forecasting error versus horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "mae_by_horizon.png", dpi=180)
    plt.close()
    print("[PLOT 3/4] Saved mae_by_horizon.png.", flush=True)
    p = pd.read_csv(RESULTS / "predictions_3h.csv", parse_dates=["origin_timestamp"])
    plt.figure(figsize=(9, 3.5))
    subset = p.head(150)
    plt.plot(subset["origin_timestamp"], subset["actual"], label="Observed future temp")
    plt.plot(subset["origin_timestamp"], subset["regression"], label="Predicted future temp")
    plt.xlabel("Forecast origin timestamp")
    plt.ylabel("Temperature (deg C)")
    plt.title("3-hour forecasts (target is 3h after each origin)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "predictions_3h.png", dpi=180)
    plt.close()
    print("[PLOT 4/4] Saved predictions_3h.png.", flush=True)
    print("Saved plots in", figdir)


def main() -> None:
    """Run only the named stage; inspect/prepare/validate/test/plot."""
    parser = argparse.ArgumentParser(description="MMA3001 room temperature forecasting")
    parser.add_argument("--stage", choices=["inspect", "prepare", "validate", "test", "plot"], required=True)
    args = parser.parse_args()
    stage_start = perf_counter()
    print("=" * 65, flush=True)
    print(f"[START] MMA3001 temperature forecasting | stage: {args.stage}", flush=True)
    print("=" * 65, flush=True)
    if args.stage == "inspect":
        inspect_csv()
    elif args.stage == "prepare":
        raw = load_sensor()
        series, audit = audit_and_regularise(raw)
        print("AUDIT:", audit)
        save_processed(series)
        RESULTS.mkdir(parents=True, exist_ok=True)
        pd.Series(audit).to_csv(RESULTS / "data_audit.csv", header=["value"])
        print("Saved", PROCESSED)
    else:
        series = load_processed()
        if args.stage == "validate":
            validate_models(series)
        elif args.stage == "test":
            final_test_models(series)
        elif args.stage == "plot":
            plot_results(series)
    print("=" * 65, flush=True)
    print(f"[DONE] Stage '{args.stage}' completed in {perf_counter() - stage_start:.2f} seconds.", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
