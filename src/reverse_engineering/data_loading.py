"""Loading of CNC measurement data and the channel-availability scenarios."""

from enum import Enum
import pathlib

import pandas as pd


def load_csv_measurement_data(file_path: pathlib.Path | str) -> pd.DataFrame:
    """Load a CNC measurement CSV into a time-indexed DataFrame.

    The first column is parsed as the elapsed-time index and converted to a
    :class:`pandas.TimedeltaIndex`. See the README ("Data format") for the expected
    columns.

    :param file_path: Path to the CSV file.
    :return: DataFrame indexed by a ``TimedeltaIndex``.
    """
    file_path = pathlib.Path(file_path)
    data = pd.read_csv(file_path, index_col=0)
    data.index = pd.TimedeltaIndex(data.index)
    return data


class DataAvailabilityScenarios(Enum):
    """Which measurement channels are published to the adversary.

    * ``ALL`` -- both positions and velocities are available.
    * ``POSITION`` -- only positions are published (velocities are reconstructed).
    * ``VELOCITY`` -- only velocities are published (positions are reconstructed).
    """

    ALL = "all"
    POSITION = "position"
    VELOCITY = "velocity"
