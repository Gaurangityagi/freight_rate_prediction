"""Validation of the approach. Run: python -m src.backtest

Each fold trains on the past and predicts the next two months, the same two-month horizon
as the real task. Cutoffs Apr 30 and Jul 31 also match its calendar: month two and month
three of a quarter, so the model must forecast a quarter-end ramp it has not seen yet.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, data, metrics
from .model import TIME_FEATURES, TwoStageRateModel, lane_key, log_rate_per_mile
from .prepare import fit_lookups, prepare

warnings.filterwarnings("ignore", category=UserWarning)

CUTOFFS = ["2025-04-30", "2025-06-30", "2025-07-31", "2025-08-31"]
NAIVE_FEATURES = config.STATIC_FEATURES + ["log_market_index", "days_since_start"]
HELD_OUT_CITIES = ["Lexington", "Memphis", "Tulsa", "Reno", "Albany", "Mobile", "Richmond", "Amarillo"]


def split(frame, cutoff):
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.offsets.MonthEnd(2)
    return frame[frame["date"] <= cutoff], frame[(frame["date"] > cutoff) & (frame["date"] <= end)]


def fit_lgbm(frame, features):
    clean = np.abs(log_rate_per_mile(frame) - log_rate_per_mile(frame).median()) < 0.75
    model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
    return model.fit(frame.loc[clean, features], log_rate_per_mile(frame[clean]))


def lgbm_rate(frame, test, features):
    return np.exp(fit_lgbm(frame, features).predict(test[features])) * test["distance"]


def lane_median_rate(train, test):
    """Baseline: lane's historic median $/mile, then the global median for unseen lanes."""
    per_mile = train["posted_rate"] / train["distance"]
    by_lane = per_mile.groupby(lane_key(train)).median()
    return lane_key(test).map(by_lane).fillna(per_mile.median()) * test["distance"]


def walk_forward(frame):
    rows = []
    specs = {
        "two-stage (final)": lambda tr, te: TwoStageRateModel().fit(tr).predict_rate(te),
        "two-stage, no quarter-end ramp": lambda tr, te: TwoStageRateModel(
            time_features=TIME_FEATURES[:2]
        ).fit(tr).predict_rate(te),
        "two-stage, no market index": lambda tr, te: TwoStageRateModel(
            time_features=TIME_FEATURES[1:]
        ).fit(tr).predict_rate(te),
        "LightGBM on date features": lambda tr, te: lgbm_rate(tr, te, NAIVE_FEATURES),
        "lane median $/mile": lane_median_rate,
    }
    for cutoff in CUTOFFS:
        train, test = split(frame, cutoff)
        for name, predict in specs.items():
            result = metrics.score(test["posted_rate"], predict(train, test))
            rows.append({"model": name, "cutoff": cutoff, **result})
            print(f"{cutoff}  {name:34s} MAPE clean {result['mape_clean']:.2f}", flush=True)
    return pd.DataFrame(rows)


def quote_signal_trap(frame):
    """What happens to a model using quote_signal, and to the score that would flatter it."""
    with_quote = NAIVE_FEATURES + ["quote_signal"]
    rows = []
    shuffled = frame.sample(frac=1, random_state=config.SEED)
    cut = int(len(shuffled) * 0.8)
    train, test = shuffled.iloc[:cut], shuffled.iloc[cut:]
    rows.append(("random 80/20 split, with quote_signal", train, test, with_quote))
    august = frame["date"].dt.month == 8
    rows.append(("August held out, with quote_signal", frame[~august], frame[august], with_quote))
    rows.append(("August held out, without quote_signal", frame[~august], frame[august], NAIVE_FEATURES))
    results = []
    for name, train, test, features in rows:
        result = metrics.score(test["posted_rate"], lgbm_rate(train, test, features))
        results.append({"experiment": name, **result})
        print(f"{name:42s} MAPE clean {result['mape_clean']:.2f}", flush=True)
    return pd.DataFrame(results)


def cold_city_holdout(frame):
    """Hide eight cities from training, as validation hides eight, and predict later dates."""
    train, test = split(frame, CUTOFFS[-1])
    hidden = lambda f: f["pickup"].isin(HELD_OUT_CITIES) | f["delivery"].isin(HELD_OUT_CITIES)
    model = TwoStageRateModel().fit(train[~hidden(train)])
    rows = []
    for name, part in (("rows touching a hidden city", test[hidden(test)]), ("all other rows", test[~hidden(test)])):
        result = metrics.score(part["posted_rate"], model.predict_rate(part))
        rows.append({"segment": name, **result})
        print(f"{name:32s} MAPE clean {result['mape_clean']:.2f}", flush=True)
    return pd.DataFrame(rows)


def quote_signal_monthly(train, validation):
    """Correlation of quote_signal with a label-free $/mile proxy, by month, for both files."""
    prepared = train[train["posted_rate"].between(1.2 * train["distance"], 4.5 * train["distance"])]
    design = lambda f: np.column_stack([
        np.ones(len(f)), 1 / f["distance"], np.log(f["distance"]), np.log(f["distance"]) ** 2,
        f["equipment"] == "Flatbed", f["equipment"] == "Reefer",
    ]).astype(float)
    coef, *_ = np.linalg.lstsq(design(prepared), log_rate_per_mile(prepared), rcond=None)
    rows = []
    for name, part in (("train", train), ("validation", validation)):
        proxy = pd.Series(design(part) @ coef, index=part.index)
        grouped = part.groupby(part["date"].dt.month)
        for month, group in grouped:
            rows.append({"file": name, "month": month,
                         "corr": np.corrcoef(proxy[group.index], group["quote_signal"])[0, 1]})
    return pd.DataFrame(rows)


def main():
    config.OUTPUTS.mkdir(exist_ok=True)
    train_raw, validation_raw = data.load_raw()
    market, weight = fit_lookups(train_raw, validation_raw)
    frame = prepare(train_raw, market, weight)

    results = walk_forward(frame)
    results.to_csv(config.OUTPUTS / "walk_forward_folds.csv", index=False)
    columns = ["mae_all", "mape_all", "rmse_all", "mae_clean", "mape_clean", "rmse_clean"]
    summary = results.groupby("model", sort=False)[columns].mean()
    summary.to_csv(config.OUTPUTS / "walk_forward_summary.csv")
    print(summary.round(2).to_string())

    quote_signal_trap(frame).to_csv(config.OUTPUTS / "quote_signal_experiments.csv", index=False)
    quote_signal_monthly(train_raw, validation_raw).to_csv(config.OUTPUTS / "quote_signal_by_month.csv", index=False)
    cold_city_holdout(frame).to_csv(config.OUTPUTS / "cold_city_holdout.csv", index=False)


if __name__ == "__main__":
    main()
