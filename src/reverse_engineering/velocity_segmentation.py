"""Segmentation of a trajectory into motion states by velocity magnitude.

Uses change-point detection (``ruptures`` PELT with a custom cost) to split the
velocity-magnitude signal into segments, each classified as standstill, rapid
traverse (``G0``) or cutting (``G1``).
"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import logging

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from ruptures.base import BaseCost

logger = logging.getLogger(__name__)


class VelocityState(str, Enum):
    """Velocity-based motion state."""

    UNDEFINED = "UNDEFINED"  # Not classified
    G0 = "G0"  # Rapid positioning
    G1 = "G1"  # Cutting motion
    STANDSTILL = "STANDSTILL"  # Not moving


@dataclass
class VelocitySegment:
    """A segment classified by velocity."""

    segment_index: int  # starts at zero
    state: VelocityState
    velocity_mm_per_s: float  # mm/s
    start_loc: int
    end_loc: int  # inclusive
    start_iloc: int
    end_iloc: int  # inclusive
    spindle_rpm: float = None  # optional spindle speed associated with the segment
    feed: float = None  # optional feedrate associated with the segment


def create_velocity_segments(
    v_mag: pd.Series,
    standstill_threshold_mm_per_s: float,
    g0_threshold_mm_per_s: float,
    min_segment_length: int,
    jump: int,
) -> list[VelocitySegment]:
    """
    Segment trajectory by velocity magnitude.

    :param v_mag: Velocity magnitude (mm/s)
    :param g0_threshold_mm_per_s: Threshold for G0 motion (mm/s)
    :param standstill_threshold_mm_per_s: Threshold for standstill (mm/s)
    :param min_segment_length: Minimum segment length in points
    :param jump: Jump parameter for ruptures
    :return: List of VelocitySegment
    """
    logger.debug(f"Velocity filtering: {len(v_mag)} points")
    logger.debug(
        f"  v_g0_threshold={g0_threshold_mm_per_s:.1f} mm/s, v_standstill={standstill_threshold_mm_per_s:.1f} mm/s"
    )
    algo = rpt.Pelt(
        model=None,
        custom_cost=CncSegmentationCost(
            standstill_threshold_mm_per_s=standstill_threshold_mm_per_s,
            g0_threshold_mm_per_s=g0_threshold_mm_per_s,
        ),
        min_size=min_segment_length,
        jump=jump,
    )
    algo.fit(v_mag.values)
    penalty_bic = np.log(len(v_mag)) / 1e2
    breakpoints = algo.predict(pen=penalty_bic)

    velocity_segments = list()
    start = 0
    for i, bp in enumerate(breakpoints):
        segment_velocities = v_mag[start:bp]
        median_velocity = np.median(segment_velocities)
        max_velocity = np.max(segment_velocities)
        if median_velocity <= standstill_threshold_mm_per_s:
            state = VelocityState.STANDSTILL
            velocity = 0.0
        elif max_velocity >= g0_threshold_mm_per_s:
            state = VelocityState.G0
            velocity = max_velocity
        else:
            state = VelocityState.G1
            velocity = median_velocity
        segment = VelocitySegment(
            segment_index=i,
            start_loc=v_mag.index[start],
            end_loc=v_mag.index[bp - 1],
            start_iloc=start,
            end_iloc=bp - 1,
            velocity_mm_per_s=velocity,
            state=state,
        )
        velocity_segments.append(segment)
        start = bp

    seg_counts = Counter(seg.state for seg in velocity_segments)
    logger.debug(
        f"  Created {len(velocity_segments)} segments: "
        f"G0={seg_counts.get(VelocityState.G0, 0)}, "
        f"G1={seg_counts.get(VelocityState.G1, 0)}, "
        f"STANDSTILL={seg_counts.get(VelocityState.STANDSTILL, 0)}"
    )

    return velocity_segments


class CncSegmentationCost(BaseCost):
    """Custom Cost for CNC velocity segmentation."""

    model = ""
    min_size = 1

    def __init__(self, standstill_threshold_mm_per_s: float, g0_threshold_mm_per_s: float):
        super().__init__()
        self.signal = None

        self.standstill_threshold_mm_per_s = standstill_threshold_mm_per_s
        self.g0_threshold_mm_per_s = g0_threshold_mm_per_s

        self.penalty_standstill = 3.0
        self.penalty_g0 = 10.0
        self.penalty_g1 = 1
        return

    def fit(self, signal):
        self.signal = signal
        return self

    def error(self, start: int, end: int):
        sub = self.signal[start:end]
        median = np.median(sub)
        max_ = np.max(sub)
        if median <= self.standstill_threshold_mm_per_s:
            n_points_above_threshold = np.sum(sub > self.standstill_threshold_mm_per_s)
            return n_points_above_threshold * self.penalty_standstill
        elif max_ >= self.g0_threshold_mm_per_s:
            n_points_below_threshold = np.sum(sub < self.standstill_threshold_mm_per_s)
            return n_points_below_threshold * self.penalty_g0
        else:
            n_points_outside_threshold = np.sum(
                (sub < self.standstill_threshold_mm_per_s) | (sub > self.g0_threshold_mm_per_s)
            )
            mae = n_points_outside_threshold * self.penalty_g1  # + 5 * np.mean(np.abs(sub - median))
            return mae


def plot_segmentation(
    vel_magnitude: pd.Series, velocity_segments: list[VelocitySegment], standstill_threshold, g0_threshold
):
    """Plot the velocity magnitude coloured by detected motion state.

    :param vel_magnitude: Velocity magnitude series (mm/s) over time.
    :param velocity_segments: Segments returned by :func:`create_velocity_segments`.
    :param standstill_threshold: Standstill threshold to draw as a reference line.
    :param g0_threshold: Rapid-traverse threshold to draw as a reference line.
    :return: The created matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, dpi=300)
    state_to_color = {"G0": "k", "G1": "r", "STANDSTILL": "y", "UNDEFINED": "grey"}
    ax.axhline(g0_threshold, color="k", linestyle="--", label="Rapid traverse threshold")
    ax.axhline(
        standstill_threshold,
        color="k",
        linestyle="-.",
        label="Standstill threshold",
    )
    for j, segment in enumerate(velocity_segments):
        start_idx = segment.start_iloc
        end_idx = segment.end_iloc
        state = segment.state
        ax.plot(
            vel_magnitude.index[start_idx : end_idx + 1].total_seconds(),
            vel_magnitude.iloc[start_idx : end_idx + 1],
            label="Velocity",
            color=state_to_color[state.name],
            alpha=0.1,
        )
        # Draw the segment's representative velocity as a horizontal line spanning
        # the segment's time range (data coordinates, not axes fractions).
        ax.hlines(
            segment.velocity_mm_per_s,
            xmin=vel_magnitude.index[start_idx].total_seconds(),
            xmax=vel_magnitude.index[end_idx].total_seconds(),
            color=state_to_color[state.name],
        )
    ax.set_title("Found velocity segments")
    ax.set_ylabel("Velocity (mm/s)")
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    return fig
