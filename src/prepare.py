"""Turns raw frames into model-ready frames."""
from . import data, features


def prepare(frame, market_by_date, weight_median):
    frame = data.clean_rows(frame, market_by_date, weight_median)
    return features.add_time_features(features.add_static_features(frame))


def fit_lookups(train, validation):
    """Target-free lookups. Dates in validation need its own market_index values."""
    return data.daily_market_index(train, validation), train["weight"].abs().median()
