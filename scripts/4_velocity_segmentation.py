"""Step 4: segment each reconstruction into motion states (G0 / G1 / standstill).

Loads ``results/<prefix>reversed.pkl``, computes velocity segments for the selected
reconstruction method in parallel, and writes
``results/<prefix>velocity_segmentation.pkl`` for the disclosure steps (5, 6_*).
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle

from reverse_engineering.classes import CncDataTransformations, EvalConfig, VelocitySegmentationConfig
from reverse_engineering.velocity_segmentation import create_velocity_segments


def evaluate(evaluation_name: str, method: str, min_segment_duration_s: float, standstill_threshold_mm_per_s: float, g0_threshold_mm_per_s: float):
    if evaluation_name is not None:
        evaluation_name = evaluation_name + "_"
    else:
        evaluation_name = ""



    with open(f"results/{evaluation_name}reversed.pkl", "rb") as f:
        reversed_results: dict[str, tuple[CncDataTransformations, EvalConfig]] = pickle.load(f)

    results_velocity_segmentation = reversed_results

    items_args = []
    for name, (data, cfg) in results_velocity_segmentation.items():
        items_args.append(
            (
                name,
                data,
                cfg,
                min_segment_duration_s,
                standstill_threshold_mm_per_s,
                g0_threshold_mm_per_s,
                method,
            )
        )

    print("Start parallel velocity segmentation (processes)")
    max_workers = min(len(items_args), (os.cpu_count() or 1) // 2 + 1)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_compute_segments_for_process, arg): arg[0] for arg in items_args}
        total = len(futures)
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            name, data = fut.result()
            cfg = results_velocity_segmentation[name][1]
            cfg.vel_segmentation = VelocitySegmentationConfig(
                g0_threshold_mm_per_s=g0_threshold_mm_per_s,
                standstill_threshold_mm_per_s=standstill_threshold_mm_per_s,
                min_segment_duration_s=min_segment_duration_s,
                smoothing_window_duration_s=data.vel_mag_est_smoothing_window,
            )
            results_velocity_segmentation[name] = (data, cfg)
            print(f"computed segment {completed} of {total}")

    print("Saving")
    with open(f"results/{evaluation_name}velocity_segmentation.pkl", "wb") as f:
        pickle.dump(results_velocity_segmentation, f)
    print("Done")


def _compute_segments_for_process(args):
    (
        name,
        data,
        cfg,
        min_segment_duration_s,
        standstill_threshold_mm_per_s,
        g0_threshold_mm_per_s,
        selected_method,
    ) = args
    min_segment_length = int(max(1, min_segment_duration_s / (cfg.protection.downsampling_rate_ms / 1e3)))
    jump = int(max(5, min_segment_length // 2))
    data_reversed = data.reversed[selected_method]
    velocity = (data_reversed["x_vel_mm_per_s"] ** 2 + data_reversed["y_vel_mm_per_s"] ** 2) ** 0.5

    data.vel_mag_est_smoothing_window = None
    data.feed_rate_est = velocity
    data.vel_segments = create_velocity_segments(
        velocity,
        standstill_threshold_mm_per_s=standstill_threshold_mm_per_s,
        g0_threshold_mm_per_s=g0_threshold_mm_per_s,
        min_segment_length=min_segment_length,
        jump=jump,
    )

    return name, data


if __name__ == "__main__":
    method="kalmen"
    g0_threshold_mm_per_s = 200
    standstill_threshold_mm_per_s = 15
    min_segment_duration_s = 0.3
    evaluate("contour_milling", method, min_segment_duration_s, standstill_threshold_mm_per_s, g0_threshold_mm_per_s)
    evaluate("face_milling", method, min_segment_duration_s, standstill_threshold_mm_per_s, g0_threshold_mm_per_s)
