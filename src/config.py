"""Paths, feature lists and hyperparameters. Everything tunable lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

TRAIN_CSV = DATA / "train_test.csv"
VALIDATION_CSV = DATA / "validation.csv"
TEMPLATE_CSV = DATA / "validation_predictions_template.csv"
DECEMBER_CSV = DATA / "december_chart_inputs.csv"
PREDICTIONS_CSV = ROOT / "validation_predictions.csv"

SEED = 42

# quote_signal is deliberately absent: see README and report. It copies the target in
# some months, inverts it in others and is pure noise in August and in Nov-Dec.
STATIC_FEATURES = [
    "distance",
    "weight_clean",
    "weight_missing",
    "equipment",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "straight_line_miles",
    "lat_change",
    "lon_change",
]

# A corrupted label sits far outside the roughly +/-5% spread of honest residuals.
OUTLIER_LOG_RESIDUAL = 0.20

# Shrinkage for lane corrections: a lane seen n times keeps n / (n + LANE_SHRINKAGE) of its
# mean residual. Picked from {1, 3, 10, 30} on the walk-forward folds.
LANE_SHRINKAGE = 3

LGBM_PARAMS = dict(
    objective="l1",
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=1.0,
    reg_lambda=5.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
