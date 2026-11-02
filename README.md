# MMA3001-Temperature-Prediction

# Indoor Temperature Forecasting

**MMA3001 — Numerical Methods and Machine Learning, By Ali Koroglu (36603961)**

A Python project that uses multiple linear regression to predict indoor temperatures 1, 3, 6 and 12 hours ahead. Different input features are compared with a persistence baseline. HVAC control is a possible future application, not part of this program.

## Setup

Install Python 3 and the required libraries:

```bash
python -m pip install numpy pandas scikit-learn matplotlib pytest tzdata
```

Place the course-provided `5EnvSensor_MayToDec2024_180kRows.csv` in `data/raw/` (subject to Monash's sharing permissions).

## Run

From the repository folder, run:

```bash
python main.py --stage prepare
python main.py --stage quality
python main.py --stage validate
python main.py --stage test
python main.py --stage plot
```

The `test` stage preserves an existing `results/test_scores.csv` rather than overwriting it.

## Outputs

- Results: `results/test_scores.csv`
- Graphs: `results/figures/`
- Tests: `python -m pytest -v test_main.py`
- Code documentation: `docs/main.html`

The selected model achieved a **6-hour test MAE of 0.607°C**, compared with **1.241°C** for the persistence baseline. The program was verified with **9 passing tests**.
