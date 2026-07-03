"""Disclose the operating-state fractions (standstill / G1 / G0) per configuration.

Reads ``results/<prefix>velocity_segmentation.pkl`` and writes
``results/<prefix>operating_states.pkl``. Run after ``4_velocity_segmentation.py``.
"""

import pickle

from reverse_engineering.helpers import get_empty_dataframe_for_results
from reverse_engineering.reconstruction import disclose_operating_states


def evaluate(prefix: str = None, method: str = "kalman"):
    if prefix is not None:
        prefix = prefix + "_"
    else:
        prefix = ""
    with open(f"results/{prefix}velocity_segmentation.pkl", "rb") as f:
        results = pickle.load(f)

    utilization = get_empty_dataframe_for_results(results)
    for name, (data, cfg) in results.items():
        seed = cfg.protection.random_state
        rate = cfg.protection.downsampling_rate_ms
        data_availability = cfg.protection.data_availability_scenario.value
        noise = cfg.protection.noise_standard_deviation_multiplier
        fraction_standstill, fraction_g1, fraction_g0 = disclose_operating_states(data)
        utilization.loc[(data_availability, rate, noise), (seed, method)] = [
            fraction_standstill,
            fraction_g1,
            fraction_g0,
        ]
    with open(f"results/{prefix}operating_states.pkl", "wb") as f:
        pickle.dump(utilization, f)


if __name__ == "__main__":
    evaluate("contour_milling")
    evaluate("face_milling")
