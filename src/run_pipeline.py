"""Trains on all labelled data and writes both prediction files. Run: python -m src.run_pipeline"""
import json

import numpy as np
import pandas as pd

from . import config, data
from .model import TIME_FEATURES, TwoStageRateModel, log_rate_per_mile
from .prepare import fit_lookups, prepare


def december_frame(december, coordinates):
    """Chart rows only give a date, so its market index comes from that date's validation loads."""
    frame = december.drop(columns="predicted_rate").copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for side in ("pickup", "delivery"):
        frame[f"{side}_lat"] = frame[side].map(coordinates["lat"])
        frame[f"{side}_lon"] = frame[side].map(coordinates["lon"])
    frame["market_index"] = np.nan
    return frame


def write_validation(model, validation):
    predictions = pd.DataFrame({
        "load_id": validation["load_id"].to_numpy(),
        "predicted_rate": np.round(model.predict_rate(validation), 2),
    })
    template = pd.read_csv(config.TEMPLATE_CSV, usecols=["load_id"])
    template.merge(predictions, on="load_id", how="left", validate="one_to_one").to_csv(
        config.PREDICTIONS_CSV, index=False
    )


def write_december(model, market, weight, coordinates):
    december = pd.read_csv(config.DECEMBER_CSV)
    prepared = prepare(december_frame(december, coordinates), market, weight)
    december["predicted_rate"] = np.round(model.predict_rate(prepared), 2)
    december.to_csv(config.DECEMBER_CSV, index=False)
    return prepared


def save_time_effects(model, prepared_december):
    coefficients = dict(zip(["intercept"] + TIME_FEATURES, map(float, model.time_coef_)))
    (config.OUTPUTS / "time_coefficients.json").write_text(json.dumps(coefficients, indent=2))
    no_ramp = prepared_december.copy()
    no_ramp[["ramp_dry_van", "ramp_flatbed", "ramp_reefer"]] = 0.0
    sensitivity = pd.DataFrame({
        "date": prepared_december["date"],
        "with_ramp": model.predict_rate(prepared_december),
        "without_ramp": model.predict_rate(no_ramp),
    })
    sensitivity.to_csv(config.OUTPUTS / "december_ramp_sensitivity.csv", index=False)


def save_daily_time_effect(model, train):
    """Observed daily level with lane and load effects removed, next to the fitted time curve."""
    clean = train[model.is_clean_]
    observed = log_rate_per_mile(clean) - model.predict_log(clean) + model.time_part(clean)
    daily = pd.DataFrame({"date": clean["date"], "observed": observed, "fitted": model.time_part(clean)})
    daily.groupby("date").mean().to_csv(config.OUTPUTS / "daily_time_effect.csv")


def main():
    config.OUTPUTS.mkdir(exist_ok=True)
    train_raw, validation_raw = data.load_raw()
    market, weight = fit_lookups(train_raw, validation_raw)
    coordinates = data.city_coordinates(train_raw, validation_raw)

    train = prepare(train_raw, market, weight)
    validation = prepare(validation_raw, market, weight)
    model = TwoStageRateModel().fit(train)
    print(f"Excluded {(~model.is_clean_).sum()} rows with corrupted labels out of {len(train):,}.")

    write_validation(model, validation)
    prepared_december = write_december(model, market, weight, coordinates)
    save_time_effects(model, prepared_december)
    save_daily_time_effect(model, train)
    print(f"Wrote {config.PREDICTIONS_CSV.name} and {config.DECEMBER_CSV.name}.")


if __name__ == "__main__":
    main()
