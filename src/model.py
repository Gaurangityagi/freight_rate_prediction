"""Two-stage rate model, fitted on log(rate per mile).

Time model: linear regression on market index, trend and quarter-end ramp. Being linear,
it can extrapolate into Nov-Dec, which is outside the training date range.

Static model: LightGBM on what a lane and load look like, with no date input. Trees cannot
extrapolate time, so they only ever see the target with the time effect removed.

Lane correction: a small shrunken offset per pickup-delivery pair, learned from out-of-fold
residuals. Lanes never seen in training get no offset and rely on the coordinates.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config

TIME_FEATURES = [
    "log_market_index",
    "days_since_start",
    "ramp_dry_van",
    "ramp_flatbed",
    "ramp_reefer",
]


def log_rate_per_mile(frame):
    return np.log(frame["posted_rate"] / frame["distance"])


def lane_key(frame):
    return frame["pickup"] + ">" + frame["delivery"]


class TwoStageRateModel:
    def __init__(self, time_features=None, lgbm_params=None, rounds=2):
        self.time_features = list(time_features or TIME_FEATURES)
        self.lgbm_params = {**config.LGBM_PARAMS, **(lgbm_params or {})}
        self.rounds = rounds

    def _time_matrix(self, frame):
        return np.column_stack([np.ones(len(frame)), frame[self.time_features].to_numpy(float)])

    def _fit_static(self, frame, target):
        model = lgb.LGBMRegressor(**self.lgbm_params)
        return model.fit(frame[config.STATIC_FEATURES], target)

    def _static_part(self, frame):
        return self.static_model_.predict(frame[config.STATIC_FEATURES])

    def time_part(self, frame):
        return self._time_matrix(frame) @ self.time_coef_

    def fit(self, frame):
        target = log_rate_per_mile(frame).to_numpy()
        self.static_model_ = self._fit_static(frame, target)  # L1 loss shrugs off bad labels
        for _ in range(self.rounds):
            residual = target - self._static_part(frame)
            keep = np.abs(residual - np.median(residual)) < 2 * config.OUTLIER_LOG_RESIDUAL
            self.time_coef_, *_ = np.linalg.lstsq(
                self._time_matrix(frame)[keep], residual[keep], rcond=None
            )
            without_time = target - self.time_part(frame)
            full_residual = without_time - self._static_part(frame)
            clean = np.abs(full_residual) < config.OUTLIER_LOG_RESIDUAL
            self.static_model_ = self._fit_static(frame[clean], without_time[clean])
        self._fit_lane_effects(frame, target - self.time_part(frame))
        residual = target - self.predict_log(frame)
        self.is_clean_ = np.abs(residual) < config.OUTLIER_LOG_RESIDUAL
        return self

    def _fit_lane_effects(self, frame, without_time):
        # Out-of-fold, because in-sample trees have already soaked up part of each lane's mean.
        out_of_fold = np.zeros(len(frame))
        folds = KFold(3, shuffle=True, random_state=config.SEED)
        for fit_idx, held_idx in folds.split(frame):
            model = self._fit_static(frame.iloc[fit_idx], without_time[fit_idx])
            out_of_fold[held_idx] = model.predict(frame.iloc[held_idx][config.STATIC_FEATURES])
        residual = pd.Series(without_time - out_of_fold, index=frame.index)
        clean = residual.abs() < config.OUTLIER_LOG_RESIDUAL
        grouped = residual[clean].groupby(lane_key(frame[clean])).agg(["sum", "size"])
        self.lane_effects_ = grouped["sum"] / (grouped["size"] + config.LANE_SHRINKAGE)

    def _lane_part(self, frame):
        return lane_key(frame).map(self.lane_effects_).fillna(0.0).to_numpy()

    def predict_log(self, frame):
        return self._static_part(frame) + self.time_part(frame) + self._lane_part(frame)

    def predict_rate(self, frame):
        return np.exp(self.predict_log(frame)) * frame["distance"].to_numpy()
