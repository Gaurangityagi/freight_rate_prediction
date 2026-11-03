# Freight Rate Prediction

Predicts `posted_rate` for 12,000 loads dated Nov-Dec 2025 from labelled loads dated Jan-Oct 2025.

**Walk-forward result:** 1.51% MAPE ($36 MAE) on rows with sound labels, 3.86% MAPE on all rows.
About 1.4% of labels are corrupted and cannot be predicted. Full write-up: `report/freight_rate_report.pdf`.

## Run

Python 3.10+. Put the provided `train_test.csv`, `validation.csv` and
`validation_predictions_template.csv` in `data/`. They are not committed.
`data/december_chart_inputs.csv` is committed with predictions filled in.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.run_pipeline      # about 5 min: trains, writes both prediction files
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv

python -m src.backtest          # about 20 min: all validation results, saved to outputs/
python report/build_report.py   # rebuilds the PDF from outputs/
pytest                          # unit tests
```

## Approach

- **Forward-in-time validation.** Each fold trains on the past and predicts the next two months.
- **`quote_signal` is excluded.** It copies the target in some months, inverts it in others, and
  is noise in August and in validation. A random split scores well with it; the honest score is
  several times worse.
- **Two-stage model on log rate per mile.** A linear time model (market index, trend, quarter-end
  ramp by equipment) can extrapolate into Nov-Dec. LightGBM on static features learns the lane and
  load shape with time removed, and a shrunken per-lane offset finishes it.
- **Cold-start cities.** Eight validation cities are missing from training, so LightGBM uses
  coordinates and never city IDs.

## Layout

| Path | Purpose |
| --- | --- |
| `src/config.py` | Paths, feature lists, hyperparameters |
| `src/data.py`, `src/prepare.py` | Loading and input cleaning |
| `src/features.py` | Static and time features |
| `src/model.py` | Two-stage model |
| `src/backtest.py` | Walk-forward, baselines, `quote_signal` and unseen-city experiments |
| `src/run_pipeline.py` | Final training and predictions |
| `score.py` | Provided scorer, unchanged |
| `outputs/` | Saved metrics and coefficients |

Predictions are reproducible on the same machine and library versions.
