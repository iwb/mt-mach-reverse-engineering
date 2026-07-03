"""Disclose the feed per tooth for each configuration.

Reads ``results/<prefix>velocity_segmentation.pkl`` and writes
``results/<prefix>feeds.pkl``. Run after ``4_velocity_segmentation.py``.
"""

import pickle
from typing import Optional

from reverse_engineering.helpers import get_empty_dataframe_for_results
from reverse_engineering.reconstruction import disclose_feed


def evaluate(prefix: Optional[str], n_teeth: int, method: str = "kalman"):
    if prefix is not None:
        prefix = prefix + "_"
    else:
        prefix = ""
    with open(f"results/{prefix}velocity_segmentation.pkl", "rb") as f:
        results = pickle.load(f)

    feeds = get_empty_dataframe_for_results(results)
    for name, (data, cfg) in results.items():
        seed = cfg.protection.random_state
        rate = cfg.protection.downsampling_rate_ms
        data_availability = cfg.protection.data_availability_scenario.value
        noise = cfg.protection.noise_standard_deviation_multiplier
        feed_disclosed = disclose_feed(data.vel_segments, data.aggregated["s_vel_deg_per_s"] / 360 * 60, n_teeth)
        feeds.loc[(data_availability, rate, noise), (seed, method)] = feed_disclosed
    with open(f"results/{prefix}feeds.pkl", "wb") as f:
        pickle.dump(feeds, f)


if __name__ == "__main__":
    evaluate("contour_milling", n_teeth=3)
    evaluate("face_milling", n_teeth=6)
