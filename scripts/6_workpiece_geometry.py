"""Disclose the machined workpiece geometry (and a fitted radius) per configuration.

Reads ``results/<prefix>velocity_segmentation.pkl`` and writes
``results/<prefix>geometries.pkl`` (and ``<prefix>features.pkl`` when a time window
is given). Run after ``4_velocity_segmentation.py``.
"""

import pickle
from typing import Optional

from matplotlib import pyplot as plt
import pandas as pd

from reverse_engineering.data_loading import DataAvailabilityScenarios
from reverse_engineering.helpers import get_empty_dataframe_for_results
from reverse_engineering.reconstruction import disclose_machined_area, disclose_radius


def evaluate(
    prefix: Optional[str],
    tool_radius: float,
    timestamp_start: Optional[pd.Timedelta],
    timestamp_end: Optional[pd.Timedelta],
    method: str = "kalman",
):
    if prefix is not None:
        prefix = prefix + "_"
    else:
        prefix = ""
    with open(f"results/{prefix}velocity_segmentation.pkl", "rb") as f:
        results_2 = pickle.load(f)

    if timestamp_start is not None and timestamp_end is not None:
        for key, result in results_2.items():
            data, cfg = result
            data_reversed = data.reversed[method]
            seed = cfg.protection.random_state
            rate = cfg.protection.downsampling_rate_ms
            data_availability = cfg.protection.data_availability_scenario.value
            noise = cfg.protection.noise_standard_deviation_multiplier
            if not (
                noise == 0 and seed == 0 and data_availability == DataAvailabilityScenarios.ALL.value and rate == 2
            ):
                continue
            fig, ax = plt.subplots(dpi=600)
            start_loc, end_loc = timestamp_start, timestamp_end
            sub_index = (data_reversed.index >= start_loc) & (data_reversed.index <= end_loc)
            sub_df = data_reversed.loc[sub_index]
            ax.plot(data_reversed["x_pos_mm"], data_reversed["y_pos_mm"])
            ax.scatter(
                sub_df["x_pos_mm"],
                sub_df["y_pos_mm"],
                label="radius",
                s=5,
                marker="x",
                alpha=0.5,
                zorder=9,
                color="orange",
            )
            ax.set_aspect("equal")
            fig.show()
        features = get_empty_dataframe_for_results(results_2)
        for key, (data, cfg) in results_2.items():
            data_reversed = data.reversed[method]
            seed = cfg.protection.random_state
            rate = cfg.protection.downsampling_rate_ms
            data_availability = cfg.protection.data_availability_scenario.value
            noise = cfg.protection.noise_standard_deviation_multiplier
            radii = list()
            start_loc, end_loc = timestamp_start, timestamp_end
            sub_index = (data_reversed.index >= start_loc) & (data_reversed.index <= end_loc)
            sub_df = data_reversed.loc[sub_index]
            x, y = sub_df["x_pos_mm"].values, sub_df["y_pos_mm"].values
            radius = abs(disclose_radius(x, y))
            radii.append(radius)

            features.loc[(data_availability, rate, noise), (seed, method)] = radii

        with open(f"results/{prefix}features.pkl", "wb") as f:
            pickle.dump(features, f)

    jump = 10
    geometries = get_empty_dataframe_for_results(results_2)
    for name, (data, cfg) in results_2.items():
        seed = cfg.protection.random_state
        rate = cfg.protection.downsampling_rate_ms
        data_availability = cfg.protection.data_availability_scenario.value
        noise = cfg.protection.noise_standard_deviation_multiplier

        union = disclose_machined_area(data, method, tool_radius=tool_radius, jump=jump)
        geometries.loc[(data_availability, rate, noise), (seed, method)] = union

    with open(f"results/{prefix}geometries.pkl", "wb") as f:
        pickle.dump(geometries, f)


if __name__ == "__main__":
    evaluate(
        "contour_milling",
        tool_radius=3,
        timestamp_start=pd.Timedelta(seconds=9.59),
        timestamp_end=pd.Timedelta(seconds=10.41),
    )
    evaluate("face_milling", tool_radius=116 / 2, timestamp_start=None, timestamp_end=None)
