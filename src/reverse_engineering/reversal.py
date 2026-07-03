"""Reversal of the protection ("attack").

Given a protected (downsampled, noisy, possibly channel-suppressed) signal, these
functions try to recover the original trajectory. Several smoothing/estimation
filters are offered (Kalman/RTS smoother, Savitzky-Golay, spline, Butterworth);
:func:`reverse_protection` runs a small hyper-parameter grid per filter and keeps
the reconstruction with the lowest position+velocity RMSE against the ground truth.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Literal, Optional, Tuple, Union

from filterpy.kalman import KalmanFilter, rts_smoother
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, filtfilt, savgol_filter

from reverse_engineering.classes import FILTER_TYPES, CncDataTransformations, EvalConfig
from reverse_engineering.data_loading import DataAvailabilityScenarios
from reverse_engineering.helpers import ceil_to_positive_odd_integer
from reverse_engineering.protection import upsample_signals_to_original_index
from reverse_engineering.reconstruction import reconstruct_positions_from_velocity
from reverse_engineering.velocity_segmentation import VelocitySegment

logger = logging.getLogger(__name__)


def estimate_laplacian_noise(df: pd.DataFrame, filter_window: int) -> dict[str, Optional[float]]:
    """
    Estimate the scale parameter (b) of Laplacian noise for each column in the DataFrame using the Median Absolute Deviation (MAD) method.
    :param df: The DataFrame for which to estimate the noise.
    :param filter_window: The window size for the Savitzky-Golay filter (
    must be a positive odd integer).
    :return: A dictionary mapping each column name to its estimated scale parameter (b).
    """
    df = df.copy()
    b_hat = {col: None for col in df.columns}
    for column in df.columns:
        if len(df[column]) < filter_window:
            filter_window_orig = filter_window
            filter_window = len(df[column])
            if filter_window % 2 == 0:
                filter_window -= 1
            logger.warning(
                f"Column {column} has less than {filter_window_orig} elements. Adjusting filter window to {filter_window}."
            )
        smoothed = df[column].rolling(window=filter_window, center=True, min_periods=1).median()
        residuals = df[column] - smoothed
        mad = np.median(np.abs(residuals))
        b_hat_ = mad / np.log(2)
        if b_hat_ is None:
            logger.warning(f"Could not estimate noise for column {column}. Setting to None")
        b_hat[column] = b_hat_
    return b_hat


def estimate_gaussian_noise(df: pd.DataFrame, filter_window: int) -> dict[str, Optional[float]]:
    """
    Estimate the std parameter of Gaussian noise for each column in the DataFrame.
    :param df: The DataFrame for which to estimate the noise.
    :param filter_window: The window size
    :return: A dictionary mapping each column name to its estimated scale parameter (b).
    """
    df = df.copy()
    std = {col: None for col in df.columns}
    for column in df.columns:
        if len(df[column]) < filter_window:
            filter_window_orig = filter_window
            filter_window = len(df[column])
            if filter_window % 2 == 0:
                filter_window -= 1
            logger.warning(
                f"Column {column} has less than {filter_window_orig} elements. Adjusting filter window to {filter_window}."
            )
        smoothed = df[column].rolling(window=filter_window, center=True, min_periods=1).mean()
        residuals = df[column] - smoothed
        mad = np.median(np.abs(residuals))
        std_ = 1.4826 * mad
        if std_ is None:
            logger.warning(f"Could not estimate noise for column {column}. Setting to None")
        std[column] = std_
    return std


def reverse_laplacian_noise(
    df,
    segments: list[VelocitySegment],
    scale: dict[str, float],
    min_feature_size_mm: float,
    dt_s: float,
):
    """
    Attempts to reverse the effect of Laplacian noise added to a DataFrame by applying a median filter.

    :param df: The DataFrame to process.
    :param segments: List of velocity segments to guide the filtering process.
    :param scale: Estimated scale parameters (b) for Laplacian noise for each column
    :param min_feature_size_mm: Minimum feature size in mm to preserve.
    :param dt_s: Time step in seconds between consecutive points in df.
    :return: A new DataFrame with the noise partially reversed.
    """
    df = df.copy()
    for segment in segments:
        min_feature_length = min_feature_size_mm / max(segment.velocity_mm_per_s, 0.1) / dt_s
        logger.debug(f"Min feature length in points for segment {segment.segment_index}: {min_feature_length:.2f}")
        for column in df.columns:
            if column not in scale:
                logger.debug("Skipping column %s", column)
                continue

            window_length = get_window_size(scale[column], min_feature_length)
            if window_length % 2 == 0:
                window_length -= 1
            logger.debug(f"Window length for column {column} in segment {segment.segment_index}: {window_length}")

            filtered = (
                df.loc[segment.start_loc : segment.end_loc, column]
                .rolling(window=window_length, center=True, min_periods=1, closed="both")
                .median()
            )
            df.loc[segment.start_loc : segment.end_loc, column] = filtered

    return df


def get_window_size(b: float, min_feature_length: float) -> int:
    if np.isnan(b):
        raise ValueError("Scale parameter b is NaN")

    window_length = np.ceil(b * min_feature_length).astype(int)
    if window_length % 2 == 0:
        window_length += 1  # Make it odd
    return window_length


@dataclass
class AxisPVSDebug:
    q_j: float
    dt: float
    innov: np.ndarray  # (N,2) [pos, vel]
    innov_std: np.ndarray
    huber_w: np.ndarray
    x_filt: np.ndarray
    P_filt: np.ndarray
    x_pred: np.ndarray
    P_pred: np.ndarray
    x_smooth: np.ndarray
    P_smooth: np.ndarray


Mode = Literal["pos_vel", "pos", "vel"]


def smooth_ca_kf_3state(
    *,
    fs: float,
    mode: Mode,
    pos: Optional[Dict[str, pd.Series]] = None,
    vel: Optional[Dict[str, pd.Series]] = None,
    sigma_pos: Optional[Union[float, Dict[str, float]]] = None,  # std
    sigma_vel: Optional[Union[float, Dict[str, float]]] = None,  # std
    q_j: Optional[Union[float, Dict[str, float]]] = None,  # jerk spectral density (process noise scale)
    x0: Optional[Dict[str, Tuple[float, float, float]]] = None,  # (p0, v0, a0)
    P0: Optional[np.ndarray] = None,  # 3x3
) -> pd.DataFrame:
    """
    Basic 3-state constant-acceleration Kalman filter + RTS smoother.

    State: x = [p, v, a]^T
    Process: constant acceleration with white jerk noise (scaled by q_j)
    Measurement:
      - mode="pos_vel": z = [p, v]^T
      - mode="pos":     z = [p]
      - mode="vel":     z = [v]

    Notes:
      - This is a *basic* KF: no robust weighting, no "missing_var" tricks, no bias terms.
      - If you want per-axis parameters, pass dicts for sigma_* and q_j.
    """
    if fs <= 0:
        raise ValueError("fs must be > 0.")
    dt = 1.0 / float(fs)

    if mode not in ("pos_vel", "pos", "vel"):
        raise ValueError("mode must be one of: 'pos_vel', 'pos', 'vel'.")

    if mode in ("pos_vel", "pos") and pos is None:
        raise ValueError("pos must be provided for mode 'pos_vel' or 'pos'.")
    if mode in ("pos_vel", "vel") and vel is None:
        raise ValueError("vel must be provided for mode 'pos_vel' or 'vel'.")

    # --- helpers -------------------------------------------------------------
    def _per_axis(val: Optional[Union[float, Dict[str, float]]], ax: str) -> Optional[float]:
        if val is None:
            return None
        if isinstance(val, dict):
            if ax not in val:
                raise ValueError(f"Missing value for axis '{ax}'.")
            return float(val[ax])
        return float(val)

    def _check_series_dict(d: Dict[str, pd.Series], name: str, idx_ref, n_ref):
        axes_local = set(d.keys())
        for ax, s in d.items():
            if not isinstance(s, pd.Series):
                raise TypeError(f"{name}['{ax}'] must be a pd.Series.")
            if idx_ref is not None:
                if len(s) != n_ref:
                    raise ValueError(f"All series must have same length. '{name}:{ax}' differs.")
                if not s.index.equals(idx_ref):
                    raise ValueError(f"All series must share identical index. '{name}:{ax}' differs.")
        return axes_local

    # --- validate & align axes/index ----------------------------------------
    axes = set()
    idx = None
    N = None

    if pos is not None:
        first = next(iter(pos.values()))
        idx = first.index
        N = len(first)
        axes |= _check_series_dict(pos, "pos", idx, N)

    if vel is not None:
        if idx is None:
            first = next(iter(vel.values()))
            idx = first.index
            N = len(first)
        axes |= _check_series_dict(vel, "vel", idx, N)

    if idx is None or N is None:
        raise ValueError("No data found.")

    axes = sorted(axes)

    # --- model matrices (3-state CA) ----------------------------------------
    F = np.array(
        [
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    Q_base = np.array(
        [
            [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
            [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
            [dt**3 / 6.0, dt**2 / 2.0, dt],
        ],
        dtype=float,
    )

    if P0 is None:
        P0 = np.diag([1.0, 1.0, 10.0]).astype(float)
    else:
        P0 = np.asarray(P0, dtype=float)
        if P0.shape != (3, 3):
            raise ValueError("P0 must be shape (3,3).")

    # Measurement matrices per mode
    if mode == "pos_vel":
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        dim_z = 2
    elif mode == "pos":
        H = np.array([[1.0, 0.0, 0.0]], dtype=float)
        dim_z = 1
    else:  # "vel"
        H = np.array([[0.0, 1.0, 0.0]], dtype=float)
        dim_z = 1

    out = pd.DataFrame(index=idx)

    # --- run per axis --------------------------------------------------------
    for ax in axes:
        y_pos = pos[ax].astype(float).to_numpy() if (pos is not None and ax in pos) else None
        y_vel = vel[ax].astype(float).to_numpy() if (vel is not None and ax in vel) else None

        sp = _per_axis(sigma_pos, ax)
        sv = _per_axis(sigma_vel, ax)
        qj = _per_axis(q_j, ax)

        if mode in ("pos_vel", "pos") and sp is None:
            raise ValueError(f"Axis '{ax}': sigma_pos required for mode '{mode}'.")
        if mode in ("pos_vel", "vel") and sv is None:
            raise ValueError(f"Axis '{ax}': sigma_vel required for mode '{mode}'.")

        Q = float(qj) * Q_base

        # Measurement covariance R
        if mode == "pos_vel":
            R = np.diag([float(sp) ** 2, float(sv) ** 2]).astype(float)
        elif mode == "pos":
            R = np.array([[float(sp) ** 2]], dtype=float)
        else:
            R = np.array([[float(sv) ** 2]], dtype=float)

        # Initial state
        if x0 is not None and ax in x0:
            x_init = np.array(x0[ax], dtype=float).reshape(3, 1)
        else:
            p0 = float(y_pos[0]) if (y_pos is not None) else 0.0
            v0 = float(y_vel[0]) if (y_vel is not None) else 0.0
            a0 = 0.0
            x_init = np.array([p0, v0, a0], dtype=float).reshape(3, 1)

        # Basic KF
        kf = KalmanFilter(dim_x=3, dim_z=dim_z)
        kf.F = F
        kf.H = H
        kf.Q = Q
        kf.R = R
        kf.x = x_init
        kf.P = P0.copy()

        x_filt = np.zeros((N, 3), dtype=float)
        P_filt = np.zeros((N, 3, 3), dtype=float)

        # Filter pass
        for k in range(N):
            kf.predict()

            if mode == "pos_vel":
                z = np.array([[y_pos[k]], [y_vel[k]]], dtype=float)
            elif mode == "pos":
                z = np.array([[y_pos[k]]], dtype=float)
            else:
                z = np.array([[y_vel[k]]], dtype=float)

            kf.update(z)

            x_filt[k] = kf.x[:, 0]
            P_filt[k] = kf.P

        # RTS smoother
        Fs = np.repeat(F[None, :, :], N, axis=0)
        Qs = np.repeat(Q[None, :, :], N, axis=0)
        xs, Ps, _, _ = rts_smoother(x_filt, P_filt, Fs, Qs)

        # Output
        out[f"{ax}_pos_smooth"] = xs[:, 0]
        out[f"{ax}_vel_smooth"] = xs[:, 1]
        out[f"{ax}_acc_smooth"] = xs[:, 2]

    return out


POS_COLS = ["x_pos_mm", "y_pos_mm"]
VEL_COLS = ["x_vel_mm_per_s", "y_vel_mm_per_s"]
AXES = [("x_pos_mm", "x_vel_mm_per_s"), ("y_pos_mm", "y_vel_mm_per_s")]


def reverse_protection(data: CncDataTransformations, cfg: EvalConfig, methods: list[str]) -> CncDataTransformations:
    """Reconstruct the original trajectory from the protected signal.

    For every requested method a small parameter grid (see :data:`PARAM_GRIDS`) is
    swept; the parameters minimising ``0.5 * (rmse_position + rmse_velocity)``
    against ``data.original`` are kept. Reconstructions are stored in
    ``data.reversed[method]`` and the winning parameters in
    ``data.filter_params[method]``. The pseudo-method ``"none"`` stores the
    unfiltered signal (with the missing channel reconstructed) as a baseline.

    :param data: Container that has already been through :func:`apply_protection`.
    :param cfg: Evaluation configuration (protection + attack settings).
    :param methods: Filter names to try, e.g. ``["none", "kalman", "savgol"]``.
    :return: The same ``data`` object with ``reversed`` / ``filter_params`` filled in.
    """
    protection_cfg = cfg.protection
    attack_cfg = cfg.attack
    data.reversed["none"] = data.noisy.copy()
    if protection_cfg.data_availability_scenario == DataAvailabilityScenarios.POSITION:
        data.reversed["none"] = add_velocities(data.reversed["none"])
    elif protection_cfg.data_availability_scenario == DataAvailabilityScenarios.VELOCITY:
        start_pos = (
            (float(data.aggregated.iloc[0]["x_pos_mm"]), float(data.aggregated.iloc[0]["y_pos_mm"]))
            if attack_cfg.start_pos_known
            else None
        )
        data.reversed["none"] = reconstruct_positions_from_velocity(data.reversed["none"], start_pos)

    data.noise_est_window = max(
        3,
        ceil_to_positive_odd_integer(
            attack_cfg.noise_estimation_window_duration_s / data.dt_aggregation.total_seconds()
        ),
    )
    data.noise_est = estimate_gaussian_noise(data.noisy, data.noise_est_window)

    scenario = cfg.protection.data_availability_scenario

    best_by_method: dict[str, float] = {}
    filter_specs = build_filter_specs()
    for method in methods:
        if method == "none":
            continue
        param_list = filter_specs[method]
        for params in param_list:
            try:
                rev = FILTERS[method](data, scenario, **params)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] filter {method} with params {params} raised: {exc}")
                continue

            if rev is None:
                r_pos = r_vel = float("nan")
            else:
                rev_upsampled = upsample_signals_to_original_index(rev, data.original.index)
                r_pos = rmse_position(data.original, rev_upsampled)
                r_vel = rmse_velocity(data.original, rev_upsampled)

            score = 0.5 * (r_pos + r_vel)

            cur_best = best_by_method.get(method)
            if cur_best is None or (np.isfinite(score) and score < cur_best):
                best_by_method[method] = float(score)
                data.reversed[method] = rev
                data.filter_params[method] = params

    return data


def rmse_position(df1: pd.DataFrame, df2: pd.DataFrame):
    """RMSE of the Euclidean position error between two trajectories (mm)."""
    pos1 = df1[["x_pos_mm", "y_pos_mm"]].to_numpy()
    pos2 = df2[["x_pos_mm", "y_pos_mm"]].to_numpy()
    euclid_pos = np.linalg.norm(pos1 - pos2, axis=1)
    rmse = np.sqrt((euclid_pos**2).mean())
    return rmse


def rmse_velocity(df1: pd.DataFrame, df2: pd.DataFrame):
    """RMSE of the feed-rate (velocity magnitude) error between two trajectories (mm/s)."""
    orig_vel = df1[["x_vel_mm_per_s", "y_vel_mm_per_s"]].to_numpy()
    rev_vel = df2[["x_vel_mm_per_s", "y_vel_mm_per_s"]].to_numpy()
    orig_fr = np.linalg.norm(orig_vel, axis=1)
    rev_fr = np.linalg.norm(rev_vel, axis=1)
    euclid_vel = orig_fr - rev_fr
    rmse = np.sqrt((euclid_vel**2).mean())
    return rmse


def _fill_missing(df, scenario, start_pos):
    if scenario == DataAvailabilityScenarios.POSITION:
        df = add_velocities(df)
    elif scenario == DataAvailabilityScenarios.VELOCITY:
        df = reconstruct_positions_from_velocity(df, pos_initial=(start_pos["x_pos_mm"], start_pos["y_pos_mm"]))
    return df


def add_velocities(df) -> pd.DataFrame:
    """Derive x/y velocities from positions by numerical differentiation."""
    df = df.copy()
    df["x_vel_mm_per_s"] = np.gradient(df["x_pos_mm"], (df.index - df.index[0]).total_seconds())
    df["y_vel_mm_per_s"] = np.gradient(df["y_pos_mm"], (df.index - df.index[0]).total_seconds())
    return df


def _empty_out(data: CncDataTransformations):
    return pd.DataFrame(index=data.suppressed.index, columns=POS_COLS + VEL_COLS, dtype=float)


def _start_pos(data: CncDataTransformations):
    return {c: float(data.aggregated[c].iloc[0]) for c in POS_COLS}


SCENARIO_TO_MODE = {
    DataAvailabilityScenarios.ALL: "pos_vel",
    DataAvailabilityScenarios.POSITION: "pos",
    DataAvailabilityScenarios.VELOCITY: "vel",
}
PARAM_GRIDS: dict[str, list[dict]] = {
    "kalman": [{"q_j": q} for q in np.logspace(0, 10, 11)],
    "savgol": [{"window_s": w, "polyorder": p} for w in [0.05, 0.1, 0.2, 0.4, 0.8] for p in [2, 3, 4]],
    "spline": [{"s_factor": s} for s in np.logspace(-2, 2, 9)],
    "butter": [{"cutoff_hz": c, "order": o} for c in [1, 2, 5, 10, 20, 50] for o in [2, 4]],
}


def filter_savgol(data: CncDataTransformations, scenario, window_s, polyorder):
    dt = data.dt_aggregation.total_seconds()
    n = len(data.noisy)
    win = max(polyorder + 2, int(round(window_s / dt)))
    if win % 2 == 0:
        win += 1
    win = min(win, n if n % 2 == 1 else n - 1)
    if win <= polyorder:
        return None
    out = _empty_out(data)
    for col in data.noisy.columns:
        out[col] = savgol_filter(data.noisy[col].to_numpy(), win, polyorder, mode="interp")
    return _fill_missing(out, scenario, _start_pos(data))


def filter_spline(data: CncDataTransformations, scenario, s_factor):
    t = data.noisy.index.total_seconds().to_numpy()
    out = _empty_out(data)
    splines = {}
    for col in data.noisy.columns:
        y = data.noisy[col].to_numpy()
        sigma = float(np.nanstd(np.diff(y))) / np.sqrt(2.0)
        s = s_factor * len(y) * (sigma**2 if sigma > 0 else 1e-6)
        spl = UnivariateSpline(t, y, k=4, s=s)
        splines[col] = spl
        out[col] = spl(t)
    if scenario == DataAvailabilityScenarios.POSITION:
        for pcol, vcol in AXES:
            out[vcol] = splines[pcol].derivative()(t)
    elif scenario == DataAvailabilityScenarios.VELOCITY:
        sp = _start_pos(data)
        out = reconstruct_positions_from_velocity(out, pos_initial=(sp["x_pos_mm"], sp["y_pos_mm"]))
    return out


def filter_butter(data: CncDataTransformations, scenario, cutoff_hz, order=4):
    dt = data.dt_aggregation.total_seconds()
    fs = 1.0 / dt
    nyq = 0.5 * fs
    if cutoff_hz >= nyq:
        return None
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    if len(data.noisy) <= 3 * (max(len(a), len(b)) - 1):
        return None
    out = _empty_out(data)
    for col in data.noisy.columns:
        out[col] = filtfilt(b, a, data.noisy[col].to_numpy())
    return _fill_missing(out, scenario, _start_pos(data))


def filter_kalman(data: CncDataTransformations, scenario, q_j):
    dt = data.dt_aggregation.total_seconds()
    fs = 1.0 / dt
    win = max(3, ceil_to_positive_odd_integer(0.1 / dt))
    noise_est = estimate_gaussian_noise(data.noisy, win)

    pos = vel = b_pos = b_vel = x0 = None
    if scenario == DataAvailabilityScenarios.ALL:
        pos = {"x": data.noisy["x_pos_mm"], "y": data.noisy["y_pos_mm"]}
        vel = {"x": data.noisy["x_vel_mm_per_s"], "y": data.noisy["y_vel_mm_per_s"]}
        b_pos = {"x": noise_est["x_pos_mm"], "y": noise_est["y_pos_mm"]}
        b_vel = {"x": noise_est["x_vel_mm_per_s"], "y": noise_est["y_vel_mm_per_s"]}
    elif scenario == DataAvailabilityScenarios.POSITION:
        pos = {"x": data.noisy["x_pos_mm"], "y": data.noisy["y_pos_mm"]}
        x0 = {
            "x": [data.noisy["x_pos_mm"].iloc[0], data.aggregated["x_vel_mm_per_s"].iloc[0], 0],
            "y": [data.noisy["y_pos_mm"].iloc[0], data.aggregated["y_vel_mm_per_s"].iloc[0], 0],
        }
        b_pos = {"x": noise_est["x_pos_mm"], "y": noise_est["y_pos_mm"]}
    else:
        vel = {"x": data.noisy["x_vel_mm_per_s"], "y": data.noisy["y_vel_mm_per_s"]}
        x0 = {
            "x": [data.aggregated["x_pos_mm"].iloc[0], data.noisy["x_vel_mm_per_s"].iloc[0], 0],
            "y": [data.aggregated["y_pos_mm"].iloc[0], data.noisy["y_vel_mm_per_s"].iloc[0], 0],
        }
        b_vel = {"x": noise_est["x_vel_mm_per_s"], "y": noise_est["y_vel_mm_per_s"]}

    sm = smooth_ca_kf_3state(
        pos=pos,
        vel=vel,
        fs=fs,
        sigma_pos=b_pos,
        sigma_vel=b_vel,
        x0=x0,
        q_j=q_j,
        mode=SCENARIO_TO_MODE[scenario],
    )

    out = _empty_out(data)
    out["x_pos_mm"] = sm["x_pos_smooth"]
    out["y_pos_mm"] = sm["y_pos_smooth"]
    out["x_vel_mm_per_s"] = sm["x_vel_smooth"]
    out["y_vel_mm_per_s"] = sm["y_vel_smooth"]
    return out


FILTERS = {
    "kalman": filter_kalman,
    "savgol": filter_savgol,
    "spline": filter_spline,
    "butter": filter_butter,
}


def build_filter_specs() -> dict[FILTER_TYPES, list[dict[str, Any]]]:
    specs = {filter_: list() for filter_ in FILTERS}
    for method, grid in PARAM_GRIDS.items():
        for params in grid:
            params: dict[str, Any]
            specs[method].append(dict(params))
    return specs
