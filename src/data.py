"""Loading and row-level cleaning shared by train, validation and December inputs."""
import pandas as pd

from . import config


def load_raw():
    train = pd.read_csv(config.TRAIN_CSV, parse_dates=["date"])
    validation = pd.read_csv(config.VALIDATION_CSV, parse_dates=["date"])
    return train, validation


def daily_market_index(*frames):
    """Mean market_index per date. Load-level values sit tightly around it (sd about 0.025)."""
    combined = pd.concat([f[["date", "market_index"]] for f in frames])
    return combined.groupby("date")["market_index"].mean()


def city_coordinates(*frames):
    """City -> (lat, lon). Each city has exactly one coordinate pair in the source data."""
    parts = []
    for frame in frames:
        for side in ("pickup", "delivery"):
            part = frame[[side, f"{side}_lat", f"{side}_lon"]]
            parts.append(part.set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def clean_rows(frame, market_by_date, weight_median):
    """Fix the input-feature defects. Never looks at the target."""
    out = frame.copy()
    out["weight_missing"] = out["weight"].isna().astype(int)
    # Negative weights are sign errors: their residuals match positive weights of equal size.
    out["weight_clean"] = out["weight"].abs().fillna(weight_median)
    # Load-level market_index carries extra signal, so keep it and fill only the gaps.
    out["market_index_clean"] = out["market_index"].fillna(out["date"].map(market_by_date))
    return out
