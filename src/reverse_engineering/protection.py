"""Protection (obfuscation) of measurement data.

These functions implement the data owner's defenses: downsampling, additive
Gaussian (or Laplacian) noise, and channel suppression. :func:`apply_protection`
ties them together according to a :class:`~reverse_engineering.classes.ProtectionConfig`.
"""

from datetime import timedelta
import logging
from typing import Optional

import numpy as np
import pandas as pd

from reverse_engineering.classes import CncDataTransformations, ProtectionConfig
from reverse_engineering.data_loading import DataAvailabilityScenarios

logger = logging.getLogger(__name__)


def laplacian_noise(df: pd.DataFrame, scale: float, random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Adds differential private noise to a pandas.Series.

    :param df: The DataFrame to which noise will be added.
    :param scale: The scale parameter for the Laplace distribution. Higher values result in more noise.
    :param random_state: An optional random seed for reproducibility.
    :return: A new pandas.Series with added noise.
    """
    if scale <= 0:
        raise ValueError("Scale must be greater than 0.")

    df = df.copy()
    if random_state is not None:
        np.random.seed(random_state)
    for column in df.columns:
        noise = np.random.laplace(loc=0.0, scale=scale, size=len(df))
        noisy_series = df[column] + noise
        df[column] = noisy_series

    return df


def add_gaussian_noise(df: pd.DataFrame, std_multiplier: float, random_state: Optional[int] = None) -> pd.DataFrame:
    """Add zero-mean Gaussian noise scaled per column to a DataFrame.

    For each column the noise standard deviation is ``std_multiplier`` times the
    column's own standard deviation, so the noise level is relative to the signal.

    :param df: The DataFrame to which noise will be added.
    :param std_multiplier: Noise std as a multiple of each column's std (> 0).
    :param random_state: Optional seed for reproducibility.
    :return: A new DataFrame with added noise.
    """
    if std_multiplier <= 0:
        raise ValueError("std_multiplier must be greater than 0.")

    df = df.copy()
    if random_state is not None:
        np.random.seed(random_state)
    for column in df.columns:
        column_std = df[column].std()
        std = column_std * std_multiplier
        noise = np.random.normal(loc=0.0, scale=std, size=len(df))
        noisy_series = df[column] + noise
        df[column] = noisy_series

    return df


def downsample_signals(df: pd.DataFrame, dt: pd.Timedelta | timedelta, keep_original: bool = False) -> pd.DataFrame:
    """
    Downsamples a DataFrame to a specified time interval.
    :param df: The DataFrame to downsample. It should have a datetime index.
    :param dt: The time interval to which the series should be downsampled.
    :param keep_original: If True, the original timestamps will be kept and interpolated. If False, only the downsampled timestamps will be kept.
    :return: A new DataFrame that is downsampled to the specified time interval with the index of the original DataFrame.
    """
    df = df.copy()
    df.sort_index(inplace=True)

    original_index = df.index.copy()
    end = df.index.max().floor(dt)
    df = df.loc[:end, :]
    df = df.resample(dt, label="right", closed="right").last()
    if keep_original:
        upsample_signals_to_original_index(df, original_index)
    df.dropna(inplace=True)
    return df


def upsample_signals_to_original_index(df: pd.DataFrame, original_index: pd.Index):
    out = df.reindex(original_index)
    out = out.interpolate(method="time", limit_area="inside")
    out = out.ffill()

    if len(out) != len(original_index):
        logger.warning("After reindexing, length changed from %d to %d.", len(original_index), len(out))

    return out


def suppress_signals_by_scenario(measurement_data: pd.DataFrame, scenario: DataAvailabilityScenarios) -> pd.DataFrame:
    """Keep only the channels published under the given availability scenario.

    :param measurement_data: DataFrame containing position and velocity columns.
    :param scenario: Which channels to keep (see :class:`DataAvailabilityScenarios`).
    :return: DataFrame restricted to the published columns.
    """
    measurement_data = measurement_data.copy()
    if scenario == DataAvailabilityScenarios.ALL:
        filtered_data = measurement_data[["x_pos_mm", "y_pos_mm", "x_vel_mm_per_s", "y_vel_mm_per_s"]]
    elif scenario == DataAvailabilityScenarios.POSITION:
        filtered_data = measurement_data[["x_pos_mm", "y_pos_mm"]]
    elif scenario == DataAvailabilityScenarios.VELOCITY:
        filtered_data = measurement_data[["x_vel_mm_per_s", "y_vel_mm_per_s"]]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return filtered_data


def apply_protection(data: CncDataTransformations, protection_cfg: ProtectionConfig) -> CncDataTransformations:
    """Apply the full protection chain to a measurement.

    Runs downsampling, channel suppression and noise addition in order, storing the
    intermediate results on ``data`` (``aggregated``, ``suppressed``, ``noisy``).
    The ``noisy`` signal is what an adversary would observe.

    :param data: Container holding at least ``original`` and ``dt_original``.
    :param protection_cfg: Protection settings to apply.
    :return: The same ``data`` object with the protection fields populated.
    """
    if protection_cfg.downsampling_rate_ms is not None:
        data.aggregated = downsample_signals(
            data.original,
            timedelta(milliseconds=protection_cfg.downsampling_rate_ms),
            keep_original=protection_cfg.keep_original_index_before_downsampling,
        )
    else:
        data.aggregated = data.original.copy()
    data.dt_aggregation = data.aggregated.index.diff().median()

    data.suppressed = suppress_signals_by_scenario(data.aggregated, protection_cfg.data_availability_scenario)
    if (
        protection_cfg.noise_standard_deviation_multiplier is not None
        and protection_cfg.noise_standard_deviation_multiplier > 0
    ):
        data.noisy = add_gaussian_noise(
            data.suppressed, protection_cfg.noise_standard_deviation_multiplier, protection_cfg.random_state
        )
    else:
        data.noisy = data.suppressed.copy()

    return data
