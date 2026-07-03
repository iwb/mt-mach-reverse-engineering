"""Small shared utilities used across the pipeline and the batch scripts."""

import numpy as np
import pandas as pd

from reverse_engineering.classes import CncDataTransformations, EvalConfig
from reverse_engineering.data_loading import DataAvailabilityScenarios


def ceil_to_positive_odd_integer(x: float) -> int:
    """Round ``x`` up to the nearest odd integer (handy for filter window lengths)."""
    x_ceil = int(np.ceil(x))
    if x_ceil % 2 == 0:
        x_ceil += 1
    return x_ceil


def get_empty_dataframe_for_results(results: dict[str, tuple[CncDataTransformations, EvalConfig]]) -> pd.DataFrame:
    """Build an empty results table indexed by the configuration grid.

    Rows are a ``(data availability, sampling rate, noise)`` MultiIndex and columns
    a ``(seed, filtering)`` MultiIndex, derived from the configurations present in
    ``results``. The batch scripts fill this skeleton with disclosed quantities.

    :param results: Mapping of run name to ``(data, config)`` pairs.
    :return: An empty DataFrame with the appropriate MultiIndex on rows and columns.
    """
    seeds = list()
    rates = list()
    noise = list()
    for name, (data, cfg) in results.items():
        seeds.append(cfg.protection.random_state)
        rates.append(cfg.protection.downsampling_rate_ms)
        noise.append(cfg.protection.noise_standard_deviation_multiplier)
    seeds = np.unique(seeds)
    rates = np.unique(rates)
    noises = np.unique(noise)
    data_availability_scenarios = [
        DataAvailabilityScenarios.ALL.value,
        DataAvailabilityScenarios.POSITION.value,
        DataAvailabilityScenarios.VELOCITY.value,
    ]

    index = pd.MultiIndex.from_product(
        [data_availability_scenarios, rates, noises], names=["Data availability", "Sampling rate", "Noise"]
    )
    columns = pd.MultiIndex.from_product([seeds, ["none", "kalman"]], names=["Seed", "Filtering"])
    df = pd.DataFrame(index=index, columns=columns)
    return df
