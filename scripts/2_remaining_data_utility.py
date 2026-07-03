"""Step 2: residual utility of the *protected* data (no reversal applied).

Quantifies how much utility survives protection directly, by measuring the
deviation of a fault position and of the disclosed machine utilization on the
protected signal. Reads ``results/<prefix>protected.pkl``.
"""

import copy
import pickle
from typing import Optional

import numpy as np
import pandas as pd

from reverse_engineering.classes import CncDataTransformations, EvalConfig
from reverse_engineering.data_loading import DataAvailabilityScenarios
from reverse_engineering.helpers import get_empty_dataframe_for_results


def evaluate(
    prefix: Optional[str],
    fault_timestamp: pd.Timedelta,
    g0_threshold_mm_per_s: float,
    standstill_threshold_mm_per_s: float,
):
    if prefix is not None:
        prefix = prefix + "_"
    else:
        prefix = ""

    with open(f"results/{prefix}protected.pkl", "rb") as f:
        results: dict[str, tuple[CncDataTransformations, EvalConfig]] = pickle.load(f)

    results_predictive_quality = copy.deepcopy(results)

    data_true: CncDataTransformations = results_predictive_quality[
        f"{DataAvailabilityScenarios.ALL.name}_ds_{2}_b_{0.0}_seed_{0}"
    ][0]
    nearest_idx = data_true.original.index.get_indexer([fault_timestamp], method="nearest")[0]
    fault_position_true = data_true.original.iloc[nearest_idx][["x_pos_mm", "y_pos_mm"]].to_numpy()
    print("True fault position", fault_position_true)
    distance_diff = get_empty_dataframe_for_results(results_predictive_quality)
    for name, (data, cfg) in results_predictive_quality.items():
        data_availability = cfg.protection.data_availability_scenario.value
        if data_availability not in (DataAvailabilityScenarios.ALL.value, DataAvailabilityScenarios.POSITION.value):
            continue

        seed = cfg.protection.random_state
        rate = cfg.protection.downsampling_rate_ms
        noise = cfg.protection.noise_standard_deviation_multiplier
        fault_position = get_fault_position(data, fault_timestamp)
        distance_diff.loc[(data_availability, rate, noise), (seed, "none")] = np.linalg.norm(
            fault_position - fault_position_true
        )

    with open(f"results/{prefix}predictive_quality_deviations.pkl", "wb") as f:
        pickle.dump(distance_diff, f)

    results_state_monitoring = copy.deepcopy(results)

    results_utilization_noisy = get_empty_dataframe_for_results(results_state_monitoring)
    for name, (data, cfg) in results_state_monitoring.items():
        if cfg.protection.data_availability_scenario not in (DataAvailabilityScenarios.VELOCITY,):
            continue
        seed = cfg.protection.random_state
        rate = cfg.protection.downsampling_rate_ms
        data_availability = cfg.protection.data_availability_scenario.value
        noise = cfg.protection.noise_standard_deviation_multiplier

        utilization = get_utilization(data, standstill_threshold_mm_per_s, g0_threshold_mm_per_s)
        results_utilization_noisy.loc[(data_availability, rate, noise), (seed, "none")] = utilization

    with open(f"results/{prefix}state_monitoring_deviations.pkl", "wb") as f:
        pickle.dump(results_utilization_noisy, f)


def get_fault_position(
    data: CncDataTransformations,
    timestamp_fault: pd.Timedelta,
):
    rate = data.dt_aggregation.total_seconds()
    nearest_idx = data.noisy.index.get_indexer([timestamp_fault], method="nearest")[0]
    fault_position = (
        data.noisy.iloc[int(nearest_idx - (1 / rate / 10)) : int(nearest_idx + (1 / rate / 10))][
            ["x_pos_mm", "y_pos_mm"]
        ]
        .median()
        .to_numpy()
    )
    return fault_position


def get_utilization(
    data: CncDataTransformations, standstill_threshold_mm_per_s: float, g0_threshold_mm_per_s: float
) -> float:
    velocity = (data.noisy["x_vel_mm_per_s"] ** 2 + data.noisy["y_vel_mm_per_s"] ** 2) ** 0.5
    time_standstill = (velocity < standstill_threshold_mm_per_s).sum()
    time_g0 = (velocity > g0_threshold_mm_per_s).sum()
    total_time = len(velocity)
    utilization = (total_time - time_standstill - time_g0) / total_time
    return utilization


if __name__ == "__main__":
    g0_threshold_mm_per_s = 200
    standstill_threshold_mm_per_s = 15
    evaluate("contour_milling", pd.Timedelta("10s"), g0_threshold_mm_per_s, standstill_threshold_mm_per_s)
    evaluate("face_milling", pd.Timedelta("55s"), g0_threshold_mm_per_s, standstill_threshold_mm_per_s)
