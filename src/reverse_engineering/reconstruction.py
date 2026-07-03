"""Disclosure of process information from a reconstructed trajectory.

Once the protection has been reversed, these functions extract the actual
manufacturing know-how an adversary is after: the tool path (position integrated
from velocity), the fraction of time spent in each operating state, the feed per
tooth, the radial engagement of consecutive passes, and the machined geometry.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_simpson
from shapely import LineString

from reverse_engineering.classes import CncDataTransformations
from reverse_engineering.velocity_segmentation import VelocitySegment, VelocityState


def reconstruct_positions_from_velocity(
    df: pd.DataFrame, pos_initial: tuple[float, float] = (0.0, 0.0)
) -> pd.DataFrame:
    """Integrate x/y velocities into positions (Simpson rule).

    :param df: DataFrame with ``x_vel_mm_per_s`` / ``y_vel_mm_per_s`` columns.
    :param pos_initial: Known start position ``(x0, y0)``; defaults to the origin.
    :return: ``df`` with ``x_pos_mm`` / ``y_pos_mm`` columns added.
    """

    def _integrate_with_fixed_dt(values, dt, initial: float = 0.0):
        if isinstance(dt, timedelta):
            dt = dt.total_seconds()
        values_int = cumulative_simpson(values, x=dt, initial=initial)
        return values_int

    df = df.copy()
    if pos_initial is None:
        pos_initial = (0.0, 0.0)

    x0, y0 = pos_initial
    timedeltas = (df.index - df.index[0]).total_seconds().values

    df["x_pos_mm"] = _integrate_with_fixed_dt(df["x_vel_mm_per_s"], timedeltas, x0)
    df["y_pos_mm"] = _integrate_with_fixed_dt(df["y_vel_mm_per_s"], timedeltas, y0)
    return df


def disclose_utilization(velocity_segments: list[VelocitySegment]) -> float:
    """Fraction of points that belong to cutting (``G1``) segments."""
    n_points_g1 = 0
    n_points_others = 0
    for segment in velocity_segments:
        segment_len = segment.end_iloc - segment.start_iloc + 1
        if segment.state == VelocityState.G1:
            n_points_g1 += segment_len
        else:
            n_points_others += segment_len
    utilization = n_points_g1 / (n_points_g1 + n_points_others)
    return utilization


def disclose_feed(
    velocity_segments: list[VelocitySegment], spindle_rpm: pd.Series, n_teeth: int = 1
) -> list[VelocitySegment]:
    """Estimate the feed per tooth for every cutting (``G1``) segment.

    Computes ``f_z = v * 60 / rpm / n_teeth`` and stores it (together with the
    median spindle speed) on each cutting segment.

    :param velocity_segments: Segments produced by velocity segmentation.
    :param spindle_rpm: Spindle speed in rev/min, indexed like the trajectory.
    :param n_teeth: Number of cutting teeth on the tool.
    :return: The same segment list, with ``feed`` / ``spindle_rpm`` filled in for
        cutting segments.
    """
    for segment in velocity_segments:
        if not (segment.state == VelocityState.G1):
            continue

        spindle_rpm_segment = np.median(spindle_rpm.loc[segment.start_loc : segment.end_loc])
        segment.spindle_rpm = spindle_rpm_segment
        segment.feed = segment.velocity_mm_per_s * 60.0 / float(spindle_rpm_segment) / n_teeth
    return velocity_segments


def disclose_radial_engagement(pts_first: pd.DataFrame, pts_second: pd.DataFrame) -> float:
    """Estimate the radial engagement between two parallel milling passes.

    A line is fitted to each pass and the mean perpendicular (y) distance between
    the two lines over their overlapping x-range is returned (mm).

    :param pts_first: Points of the first pass (``x_pos_mm`` / ``y_pos_mm``).
    :param pts_second: Points of the second pass.
    :return: Mean radial engagement ``a_e`` in mm.
    """
    # fit line to each of the points, then calculate mean y distance between the points
    line_1 = np.polyfit(pts_first["x_pos_mm"], pts_first["y_pos_mm"], 1)
    line_2 = np.polyfit(pts_second["x_pos_mm"], pts_second["y_pos_mm"], 1)

    # create points along each line
    x_min = max(pts_first["x_pos_mm"].min(), pts_second["x_pos_mm"].min())
    x_max = min(pts_first["x_pos_mm"].max(), pts_second["x_pos_mm"].max())
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals_1 = line_1[0] * x_vals + line_1[1]
    y_vals_2 = line_2[0] * x_vals + line_2[1]

    radial_engagement = float(np.mean(y_vals_2 - y_vals_1))
    return radial_engagement


def disclose_radius(x_pos: list, y_pos: list):
    """Fit a circle to a set of points (least squares) and return its radius (mm)."""
    # fit a circle to the points and return the radius
    x = np.array(x_pos)
    y = np.array(y_pos)
    A = np.c_[x, y, np.ones(x.shape[0])]
    B = x**2 + y**2
    C = np.linalg.lstsq(A, B, rcond=None)[0]
    xc = C[0] / 2
    yc = C[1] / 2
    radius = np.sqrt(xc**2 + yc**2 + C[2])
    return radius


def disclose_operating_states(data: CncDataTransformations):
    """Fraction of time spent in standstill, cutting (``G1``) and rapid (``G0``).

    :param data: Container whose ``vel_segments`` have already been computed.
    :return: Tuple ``(fraction_standstill, fraction_g1, fraction_g0)``.
    """
    n_points_g1 = 0
    n_points_g0 = 0
    total_len = 0
    for segment in data.vel_segments:
        segment_len = segment.end_iloc - segment.start_iloc + 1
        if segment.state == VelocityState.G1:
            n_points_g1 += segment_len
        elif segment.state == VelocityState.G0:
            n_points_g0 += segment_len
        total_len += segment_len
    fraction_g1 = n_points_g1 / total_len
    fraction_g0 = n_points_g0 / total_len
    fraction_standstill = 1 - (fraction_g0 + fraction_g1)
    return fraction_standstill, fraction_g1, fraction_g0


def disclose_machined_area(data: CncDataTransformations, method: str, tool_radius: float, jump: int = 1):
    """Reconstruct the machined area as the union of tool sweeps over cutting segments.

    Each cutting (``G1``) segment's tool-centre path is buffered by ``tool_radius``
    and the buffers are unioned into a single (multi)polygon approximating the
    removed material.

    :param data: Container with ``reversed`` reconstructions and ``vel_segments``.
    :param method: Which reconstruction in ``data.reversed`` to use.
    :param tool_radius: Tool radius in mm used to buffer the path.
    :param jump: Use every ``jump``-th point to speed up the geometry union.
    :return: A shapely (multi)polygon, or ``None`` if no cutting segment qualifies.
    """
    data_reversed = data.reversed[method]
    x_pos = data_reversed["x_pos_mm"]
    y_pos = data_reversed["y_pos_mm"]
    union = None
    for segment in data.vel_segments:
        start_idx = segment.start_iloc
        end_idx = min(len(x_pos), segment.end_iloc + 2)
        x_pos_segment = x_pos.iloc[start_idx:end_idx].iloc[::jump]
        y_pos_segment = y_pos.iloc[start_idx:end_idx].iloc[::jump]

        if len(x_pos_segment) == 1:
            continue
        if segment.state == VelocityState.G1:
            line = LineString(np.array(list(zip(x_pos_segment.to_numpy(), y_pos_segment.to_numpy())))).buffer(
                tool_radius
            )
            if union is None:
                union = line
            else:
                union = union.union(line)
    return union
