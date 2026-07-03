"""Step 3: reverse the protection for every protected configuration.

Loads ``results/<prefix>protected.pkl`` and runs the reversal filters (the grid in
``reverse_protection``) on each entry in parallel, writing the reconstructions to
``results/<prefix>reversed.pkl``.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
from typing import Optional

from reverse_engineering.classes import AttackConfig, CncDataTransformations, EvalConfig
from reverse_engineering.reversal import reverse_protection


def evaluate(evaluation_name: str = None, methods: Optional[list] = None):
    if evaluation_name is not None:
        evaluation_name = evaluation_name + "_"
    else:
        evaluation_name = ""

    if methods is None:
        methods = ["none", "kalman", "spline", "butter", "savgol"]

    with open(f"results/{evaluation_name}protected.pkl", "rb") as f:
        protected_results = pickle.load(f)

    items_args = []
    for name, (data, cfg) in protected_results.items():
        items_args.append((name, data, cfg))

    noise_estimation_window_duration_s = 0.1

    results = dict()
    try:
        tasks = list()  # list of args for worker
        total = len(items_args)

        for index, (name, data, cfg) in enumerate(items_args):
            cfg: EvalConfig
            data: CncDataTransformations

            cfg.attack = AttackConfig(
                start_pos_known=True,
                end_pos_known=False,
                noise_estimation_window_duration_s=noise_estimation_window_duration_s,
            )
            tasks.append((name, data, cfg, methods))
            print(f"[ENQUEUE] {index + 1}/{total}: {name}")

        if tasks:
            max_workers = min(len(tasks), (os.cpu_count() or 1) // 2 + 1)
            print(f"Start parallel _reverse_protection (processes), workers={max_workers}, tasks={len(tasks)}")
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_reverse_protection_worker, args): (args[0],) for args in tasks}
                completed = 0
                for fut in as_completed(futures):
                    completed += 1
                    try:
                        name, data, cfg = fut.result()
                    except Exception as exc:
                        task_info = futures.get(fut, ("?", "?"))
                        print(f"[ERROR] exception while computing {task_info}: {exc}")
                        raise
                    results[name] = (data, cfg)
                    print(f"[RUN] {name} (completed {completed}/{len(futures)})")
    finally:
        with open(f"results/{evaluation_name}reversed.pkl", "wb") as f:
            pickle.dump(results, f)


def _reverse_protection_worker(args):
    """
    Worker that loads the measurement inside the child process and runs derive_position_data.
    args: tuple (part_id: str, data_path: str, index: int, name: str, cfg: EvalConfig)
    Returns: (index, name, data, cfg)
    """

    name, data, cfg, methods = args
    data = reverse_protection(data, cfg, methods)
    return name, data, cfg


if __name__ == "__main__":
    evaluate("contour_milling", methods=["none", "kalman", "spline", "butter", "savgol"])
    evaluate("face_milling", methods=["none", "kalman"])
