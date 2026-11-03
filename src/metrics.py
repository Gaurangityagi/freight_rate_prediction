import numpy as np

from . import config


def score(actual, predicted):
    """Errors on all rows, and on rows whose label is not corrupted."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    clean = np.abs(np.log(actual / predicted)) < config.OUTLIER_LOG_RESIDUAL
    error = np.abs(actual - predicted)
    return {
        "rows": len(actual),
        "clean_share": clean.mean(),
        "mae_all": error.mean(),
        "mape_all": (error / actual).mean() * 100,
        "rmse_all": np.sqrt((error**2).mean()),
        "mae_clean": error[clean].mean(),
        "mape_clean": (error / actual)[clean].mean() * 100,
        "rmse_clean": np.sqrt((error[clean] ** 2).mean()),
        "bias_clean_pct": (np.log(actual / predicted)[clean]).mean() * 100,
    }
