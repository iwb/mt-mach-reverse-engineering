"""Configuration and data-container dataclasses used throughout the pipeline.

These objects are deliberately plain ``dataclass`` containers so they can be
pickled between the parallel worker processes used by the batch scripts.

The two central objects are:

* :class:`EvalConfig` -- describes *what* to do (how to protect the data and how
  the adversary attacks it).
* :class:`CncDataTransformations` -- collects every intermediate signal produced
  while a single measurement flows through the pipeline.
"""

import dataclasses
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, Optional

import pandas as pd

from reverse_engineering.data_loading import DataAvailabilityScenarios
from reverse_engineering.velocity_segmentation import VelocitySegment


@dataclass
class ProtectionConfig:
    """Settings for protecting (obfuscating) a measurement signal.

    :param data_availability_scenario: Which channels are published to the
        adversary (all, positions only, or velocities only).
    :param noise_standard_deviation_multiplier: Gaussian noise standard deviation
        as a multiple of each channel's own standard deviation. ``None`` or ``0``
        disables noise.
    :param downsampling_rate_ms: Target sampling period in milliseconds. ``None``
        keeps the original sampling rate.
    :param keep_original_index_before_downsampling: If ``True``, re-interpolate the
        downsampled signal back onto the original (dense) time index.
    :param random_state: Seed for the noise generator, for reproducibility.
    """

    data_availability_scenario: DataAvailabilityScenarios
    noise_standard_deviation_multiplier: Optional[float]
    downsampling_rate_ms: Optional[int]
    keep_original_index_before_downsampling: bool
    random_state: int


@dataclass
class AttackConfig:
    """Assumptions and settings available to the adversary reversing the protection.

    :param start_pos_known: Whether the true start position is known (used to anchor
        position reconstruction when only velocities are published).
    :param end_pos_known: Whether the true end position is known.
    :param noise_estimation_window_duration_s: Window length (seconds) used to
        estimate the added noise level from the published signal.
    """

    start_pos_known: bool
    end_pos_known: bool
    noise_estimation_window_duration_s: Optional[float]


@dataclass
class VelocitySegmentationConfig:
    """Thresholds used to segment a trajectory into motion states.

    :param g0_threshold_mm_per_s: Velocity above which motion counts as rapid
        traverse (``G0``).
    :param standstill_threshold_mm_per_s: Velocity below which motion counts as
        standstill.
    :param min_segment_duration_s: Minimum duration of a detected segment.
    :param smoothing_window_duration_s: Optional smoothing window applied to the
        velocity magnitude before segmentation.
    """

    g0_threshold_mm_per_s: float
    standstill_threshold_mm_per_s: float
    min_segment_duration_s: float
    smoothing_window_duration_s: Optional[float]


@dataclass
class EvalConfig:
    """Full configuration of a single evaluation: protection + attack (+ segmentation)."""

    protection: ProtectionConfig
    attack: AttackConfig
    vel_segmentation: Optional[VelocitySegmentationConfig] = None


#: Filters available to reverse the protection. ``"none"`` means no filtering.
FILTER_TYPES = Literal["none", "kalman", "savgol", "butter", "spline"]


@dataclasses.dataclass
class CncDataTransformations:
    """Container holding every intermediate signal for one measurement.

    A single instance is threaded through the whole pipeline; each stage fills in
    the next set of fields. Only ``original`` and ``dt_original`` are required up
    front -- the rest are populated by :func:`~reverse_engineering.protection.apply_protection`,
    :func:`~reverse_engineering.reversal.reverse_protection` and the segmentation step.

    :param original: The unmodified ("ground truth") measurement.
    :param dt_original: Sampling period of ``original``.
    :param dt_aggregation: Sampling period after downsampling.
    :param aggregated: Signal after downsampling (still all channels).
    :param suppressed: ``aggregated`` reduced to the published channels.
    :param noisy: ``suppressed`` with additive noise -- this is what the adversary sees.
    :param noise_est: Per-channel estimate of the added noise level.
    :param feed_rate_est: Estimated feed-rate (velocity magnitude) series.
    :param vel_segments: Motion-state segments (``G0``/``G1``/standstill).
    :param noise_est_window: Window length (in samples) used for noise estimation.
    :param vel_mag_est_smoothing_window: Smoothing window used before segmentation.
    :param reversed: Reconstructions keyed by filter name.
    :param filter_params: The best hyper-parameters found per filter.
    """

    original: pd.DataFrame
    dt_original: timedelta
    dt_aggregation: Optional[timedelta] = None
    aggregated: Optional[pd.DataFrame] = None
    suppressed: Optional[pd.DataFrame] = None
    noisy: Optional[pd.DataFrame] = None
    noise_est: Optional[dict[str, Optional[float]]] = None
    feed_rate_est: Optional[pd.Series] = None
    vel_segments: list[VelocitySegment] = None
    noise_est_window: Optional[int] = None
    vel_mag_est_smoothing_window: Optional[float] = None
    reversed: dict[FILTER_TYPES, pd.DataFrame] = dataclasses.field(default_factory=dict)
    filter_params: dict[FILTER_TYPES, dict[str, Any]] = dataclasses.field(default_factory=dict)
