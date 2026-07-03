"""Step 1: apply protection over the full configuration grid.

Loads a measurement CSV and, for every combination of data-availability scenario,
downsampling rate, noise level and random seed, applies the protection chain. The
protected results are pickled to ``results/<name>protected.pkl`` for the later steps.
Runs the grid in parallel with a ``ProcessPoolExecutor``.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import os
import pathlib
import pickle

import numpy as np
import pandas as pd

from reverse_engineering.classes import AttackConfig, CncDataTransformations, EvalConfig, ProtectionConfig
from reverse_engineering.data_loading import DataAvailabilityScenarios, load_csv_measurement_data
from reverse_engineering.protection import apply_protection


def evaluate(measurement_data_file_path: pathlib.Path, evaluation_name: str = None):
    if evaluation_name is not None:
        evaluation_name = evaluation_name + "_"
    else:
        evaluation_name = ""

    configs = pd.DataFrame(
        columns=[
            "name",
            "data_availability_scenario",
            "noise_std_multiplier",
            "seed",
            "downsampling_rate_ms",
        ]
    )
    noise_estimation_window_duration_s = 0.1
    seeds = [0, 42, 100, 99, 1337, 1, 2, 3, 4, 5]
    downsampling_rates = [2, 100]
    noise_std_multiplier = np.arange(0, 1.01, 0.1).round(1)

    for data_availability_scenario in (
        DataAvailabilityScenarios.ALL,
        DataAvailabilityScenarios.POSITION,
        DataAvailabilityScenarios.VELOCITY,
    ):
        for downsampling_rate, laplace_noise in itertools.product(downsampling_rates, noise_std_multiplier):
            for seed in seeds:
                name = f"{data_availability_scenario.name}_ds_{downsampling_rate}_b_{laplace_noise}_seed_{seed}"
                configs.loc[len(configs)] = {
                    "name": name,
                    "data_availability_scenario": data_availability_scenario,
                    "noise_std_multiplier": laplace_noise,
                    "seed": seed,
                    "downsampling_rate_ms": downsampling_rate,
                }
                if laplace_noise == 0:
                    break

    results = dict()
    try:
        tasks = list()  # list of args for worker
        total = len(configs)

        for index, row in configs.iterrows():
            filter_window = noise_estimation_window_duration_s
            cfg = EvalConfig(
                protection=ProtectionConfig(
                    data_availability_scenario=row["data_availability_scenario"],
                    noise_standard_deviation_multiplier=row["noise_std_multiplier"],
                    downsampling_rate_ms=row["downsampling_rate_ms"],
                    keep_original_index_before_downsampling=False,
                    random_state=row["seed"],
                ),
                attack=AttackConfig(
                    start_pos_known=True,
                    end_pos_known=False,
                    noise_estimation_window_duration_s=filter_window,
                ),
            )
            name = row["name"]
            tasks.append((measurement_data_file_path, index, name, cfg))
            print(f"[ENQUEUE] {index + 1}/{total}: {name}")

        if tasks:
            max_workers = min(len(tasks), (os.cpu_count() or 1) // 2 + 1)
            print(f"Start parallel _apply_protection (processes), workers={max_workers}, tasks={len(tasks)}")
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_apply_protection_worker, args): (args[2], args[3]) for args in tasks}
                completed = 0
                for fut in as_completed(futures):
                    completed += 1
                    try:
                        index, name, data, cfg = fut.result()
                    except Exception as exc:
                        task_info = futures.get(fut, ("?", "?"))
                        print(f"[ERROR] exception while computing {task_info}: {exc}")
                        raise
                    results[name] = (data, cfg)
                    print(f"[RUN] {name} (completed {completed}/{len(futures)})")
    finally:
        with open(f"results/{evaluation_name}protected.pkl", "wb") as f:
            pickle.dump(results, f)


def _apply_protection_worker(args):
    data_path, index, name, cfg = args
    df = load_csv_measurement_data(data_path)
    dt_original = df.index.diff().median()
    data = CncDataTransformations(original=df.copy(), dt_original=dt_original)
    data = apply_protection(data, cfg.protection)
    return index, name, data, cfg


if __name__ == "__main__":
    data_folder = pathlib.Path(__file__).parent.parent / "data"
    file_name = "contour_milling.csv"
    evaluation_name = "contour_milling"
    evaluate(data_folder / file_name, evaluation_name)
    file_name = "face_milling.csv"
    evaluation_name = "face_milling"
    evaluate(data_folder / file_name, evaluation_name)
