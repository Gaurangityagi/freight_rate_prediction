"""Feature engineering. Time features and static features are built separately on purpose."""
import numpy as np
import pandas as pd

EARTH_RADIUS_MILES = 3958.8
RAMP_DAYS = 31
EQUIPMENT_SLUGS = {"Dry Van": "dry_van", "Flatbed": "flatbed", "Reefer": "reefer"}


def straight_line_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(h))


def add_static_features(frame):
    out = frame.copy()
    out["straight_line_miles"] = straight_line_miles(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    out["lat_change"] = out["delivery_lat"] - out["pickup_lat"]
    out["lon_change"] = out["delivery_lon"] - out["pickup_lon"]
    out["equipment"] = pd.Categorical(out["equipment"], categories=["Dry Van", "Flatbed", "Reefer"])
    return out


def add_time_features(frame):
    out = frame.copy()
    date = out["date"]
    quarter_end = date.dt.to_period("Q").dt.end_time.dt.normalize()
    out["days_to_quarter_end"] = (quarter_end - date).dt.days
    out["days_since_start"] = (date - pd.Timestamp("2025-01-01")).dt.days
    # 0 until RAMP_DAYS before quarter end, then climbs linearly to 1 on the last day.
    out["quarter_end_ramp"] = np.clip(1 - out["days_to_quarter_end"] / RAMP_DAYS, 0, 1)
    # Flatbed ramps about three times as hard as Dry Van, in every quarter end we can see.
    for name, slug in EQUIPMENT_SLUGS.items():
        out[f"ramp_{slug}"] = out["quarter_end_ramp"] * (out["equipment"] == name)
    out["log_market_index"] = np.log(out["market_index_clean"])
    return out
