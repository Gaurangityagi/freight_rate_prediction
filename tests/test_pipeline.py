import numpy as np
import pandas as pd
import pytest

from src import config, data, features
from src.backtest import split
from src.model import TIME_FEATURES


def row(**overrides):
    base = dict(date=pd.Timestamp("2025-12-31"), weight=30000.0, market_index=1.0, equipment="Dry Van")
    return pd.DataFrame([{**base, **overrides}])


def test_negative_weight_becomes_positive():
    market = pd.Series({pd.Timestamp("2025-12-31"): 1.0})
    assert data.clean_rows(row(weight=-20000.0), market, 31000)["weight_clean"].iloc[0] == 20000


def test_missing_values_are_filled_and_flagged():
    market = pd.Series({pd.Timestamp("2025-12-31"): 0.95})
    cleaned = data.clean_rows(row(weight=np.nan, market_index=np.nan), market, 31000).iloc[0]
    assert cleaned["weight_clean"] == 31000
    assert cleaned["weight_missing"] == 1
    assert cleaned["market_index_clean"] == 0.95


@pytest.mark.parametrize(
    "date,expected", [("2025-12-31", 1.0), ("2025-11-30", 0.0), ("2026-01-01", 0.0), ("2025-06-30", 1.0)]
)
def test_ramp_peaks_at_quarter_end_and_resets(date, expected):
    frame = row(date=pd.Timestamp(date), market_index_clean=1.0)
    assert features.add_time_features(frame)["quarter_end_ramp"].iloc[0] == pytest.approx(expected)


def test_ramp_only_fires_for_own_equipment():
    frame = row(equipment="Reefer", market_index_clean=1.0)
    result = features.add_time_features(frame).iloc[0]
    assert result["ramp_reefer"] == 1.0 and result["ramp_dry_van"] == 0.0


def test_quote_signal_can_never_reach_the_model():
    assert "quote_signal" not in config.STATIC_FEATURES + TIME_FEATURES


def test_splits_never_train_on_the_future():
    frame = pd.DataFrame({"date": pd.date_range("2025-01-01", "2025-10-31")})
    train, test = split(frame, "2025-07-31")
    assert train["date"].max() < test["date"].min()
    assert test["date"].max() == pd.Timestamp("2025-09-30")
