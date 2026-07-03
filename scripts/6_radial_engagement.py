"""Disclose the radial engagement of two milling passes per configuration.

Reads ``results/<prefix>velocity_segmentation.pkl`` and writes
``results/<prefix>radial_engagements.pkl``. Run after ``4_velocity_segmentation.py``.
"""

import pickle
from typing import Optional

import pandas as pd

from reverse_engineering.classes import CncDataTransformations, EvalConfig
from reverse_engineering.helpers import get_empty_dataframe_for_results
from reverse_engineering.reconstruction import disclose_radial_engagement


def evaluate(
    prefix: Optional[str],
    timestamp_start_first_pass: pd.Timedelta,
    timestamp_end_first_pass: pd.Timedelta,
    timestamp_start_second_pass: pd.Timedelta,
    timestamp_end_second_pass: pd.Timedelta,
    method: str = "kalman",
):
    if prefix is not None:
        prefix = prefix + "_"
    else:
        prefix = ""
    with open(f"results/{prefix}velocity_segmentation.pkl", "rb") as f:
        results: dict[str, tuple[CncDataTransformations, EvalConfig]] = pickle.load(f)

    radial_engagements = get_empty_dataframe_for_results(results)
    for key, (data, cfg) in results.items():
        seed = cfg.protection.random_state
        rate = cfg.protection.downsampling_rate_ms
        data_availability = cfg.protection.data_availability_scenario.value
        noise = cfg.protection.noise_standard_deviation_multiplier

        data_reversed = data.reversed[method]
        start_loc_1, end_loc_1 = timestamp_start_first_pass, timestamp_end_first_pass
        sub_index_1 = (data_reversed.index >= start_loc_1) & (data_reversed.index <= end_loc_1)
        start_loc_2, end_loc_2 = timestamp_start_second_pass, timestamp_end_second_pass
        sub_index_2 = (data_reversed.index >= start_loc_2) & (data_reversed.index <= end_loc_2)

        if sub_index_2.sum() > 0 and sub_index_1.sum() > 0:
            radial_engagement_finishing = disclose_radial_engagement(
                data_reversed.loc[sub_index_1, :], data_reversed.loc[sub_index_2, :]
            )
        else:
            print(f"skipping {key} because no finishing or roughing segment found")
            radial_engagement_finishing = None
        radial_engagements.loc[(data_availability, rate, noise), (seed, method)] = [
            radial_engagement_finishing,
        ]

    with open(f"results/{prefix}radial_engagements.pkl", "wb") as f:
        pickle.dump(radial_engagements, f)


if __name__ == "__main__":
    evaluate(
        "contour_milling",
        timestamp_start_first_pass=pd.Timedelta(seconds=4.5),
        timestamp_end_first_pass=pd.Timedelta(seconds=5.57),
        timestamp_start_second_pass=pd.Timedelta(seconds=11.5),
        timestamp_end_second_pass=pd.Timedelta(seconds=12.95),
    )
    evaluate(
        "face_milling",
        timestamp_start_first_pass=pd.Timedelta(seconds=0),
        timestamp_end_first_pass=pd.Timedelta(seconds=20.3),
        timestamp_start_second_pass=pd.Timedelta(seconds=22),
        timestamp_end_second_pass=pd.Timedelta(seconds=41),
    )
