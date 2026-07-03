from collections import defaultdict
import pathlib
import pickle

import figure_config as fc
from figure_config import (
    COLOR_DEFAULT_BLACK,
    COLOR_DEFAULT_BLUE,
    one_column_width,
    set_figure_config,
    two_column_width,
)
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np
from plotting_helpers import bake_alpha, gaussian_stats, plot_with_boundary_lines
import shapely.plotting

from reverse_engineering.classes import CncDataTransformations, EvalConfig
from reverse_engineering.data_loading import DataAvailabilityScenarios

set_figure_config()
default_arrow = dict(arrowstyle="-|>", linewidth=1, shrinkA=0, shrinkB=0, color=COLOR_DEFAULT_BLACK)
sampling_rate_to_marker = {2: "x", 100: "+"}
pad = 1.08

prefix = "face_milling_"
folder = pathlib.Path("../results/")
with open(folder / f"{prefix}velocity_segmentation.pkl", "rb") as f:
    results_reversal_face: dict[str, tuple[CncDataTransformations, EvalConfig]] = pickle.load(f)

fig_no = 8
method = "kalman"

s = 16
jump = 10
tool_radius = 116 / 2
lw = 0.5

data, cfg = results_reversal_face["ALL_ds_2_b_0.0_seed_0"]
union_true = shapely.geometry.box(0, 0, 200, 200)

x_pos = data.original["x_pos_mm"]
y_pos = data.original["y_pos_mm"]

fig, ax = plt.subplots(1, 1, figsize=(one_column_width(), one_column_width() / 1.5))
shapely.plotting.plot_polygon(union_true, ax, add_points=False, edgecolor=COLOR_DEFAULT_BLACK, zorder=2)
ax.plot(
    data.original["x_pos_mm"],
    data.original["y_pos_mm"],
    label="Toolpath",
    alpha=1.0,
    color=COLOR_DEFAULT_BLACK,
    linewidth=lw,
)

dx = data.original["x_pos_mm"].diff()
dy = data.original["y_pos_mm"].diff()
length = np.linalg.norm(np.array([dx, dy]).T, axis=1)
feedrate = (data.original["x_vel_mm_per_s"] ** 2 + data.original["y_vel_mm_per_s"] ** 2) ** 0.5
length[feedrate < 3] = 0
dx = dx / length
dy = dy / length
n = 500
pts = list()
dist_min = 100
dist_acc = 0
row_prev = None
for idx, row in data.original.iterrows():
    if idx == data.original.index[0]:
        row_prev = row
        continue
    dist_acc += ((row["x_pos_mm"] - row_prev["x_pos_mm"]) ** 2 + (row["y_pos_mm"] - row_prev["y_pos_mm"]) ** 2) ** 0.5
    if dist_acc >= dist_min:
        pts.append(idx)
        dist_acc = 0
    row_prev = row
ax.quiver(
    data.original.loc[pts, "x_pos_mm"],
    data.original.loc[pts, "y_pos_mm"],
    dx.loc[pts].values,
    dy[pts].values,
    angles="xy",
    scale_units="xy",
    scale=0.05,
    width=0.001,
    zorder=3,
    headlength=20,
    headaxislength=18,
    headwidth=20,
    color=COLOR_DEFAULT_BLACK,
)

ax.set_ylabel("$\it{Y}$-axis position in mm →")
ax.set_aspect("equal", "datalim")

toolpath_arrow = FancyArrowPatch(
    (0, 0), (10, 0), arrowstyle=default_arrow, color=COLOR_DEFAULT_BLACK, linewidth=lw, shrinkA=0, shrinkB=0
)
toolpath_arrow.set_label("Toolpath")
legend_handles = [
    Line2D([0], [0], color=COLOR_DEFAULT_BLACK, linestyle="-", label="Toolpath", linewidth=lw),
    plt.Rectangle(
        (0, 0),
        0,
        0,
        transform=ax.transAxes,
        label="Part geometry",
        fc=to_rgba(COLOR_DEFAULT_BLACK, 0.3),
        ec=COLOR_DEFAULT_BLACK,
    ),
]
fig.legend(handles=legend_handles, ncol=2, loc="lower center", bbox_to_anchor=(0.55, -0.07), frameon=True)

fig.suptitle("Reference part 2 – face milling")
fig.tight_layout(pad=pad, rect=[0, 0, 1, 1])
fig.savefig(f"plots/Fig{fig_no}.pdf")
fig.show()
fig_no += 1

data_500_hz, _ = results_reversal_face["ALL_ds_2_b_0.5_seed_0"]
data_10_hz, _ = results_reversal_face["ALL_ds_100_b_0.5_seed_0"]

fig, ax = plt.subplots(5, 2, figsize=(one_column_width(), two_column_width() / 1.4), sharex=True, sharey="row")

COLOR_LIGHT_BLUE = bake_alpha(COLOR_DEFAULT_BLUE, alpha=0.5)
COLOR_LIGHT_GREY = bake_alpha(COLOR_DEFAULT_BLACK, alpha=0.5)
colors_normal = (COLOR_DEFAULT_BLUE, COLOR_DEFAULT_BLACK)
colors_downsampled = (COLOR_LIGHT_BLUE, COLOR_LIGHT_GREY)
for i, data in enumerate([data_500_hz, data_10_hz]):
    ax[0, i].plot(data.noisy.index.total_seconds(), data.noisy["x_pos_mm"], color=colors_downsampled[i], linestyle="-")
    ax[2, i].plot(data.noisy.index.total_seconds(), data.noisy["y_pos_mm"], color=colors_downsampled[i], linestyle="-")
    ax[1, i].plot(
        data.noisy.index.total_seconds(), data.noisy["x_vel_mm_per_s"], color=colors_downsampled[i], linestyle="-"
    )
    ax[3, i].plot(
        data.noisy.index.total_seconds(), data.noisy["y_vel_mm_per_s"], color=colors_downsampled[i], linestyle="-"
    )

    ax[0, i].plot(
        data.aggregated.index.total_seconds(),
        data.aggregated["x_pos_mm"],
        color=colors_normal[i],
        linestyle="-",
    )
    ax[2, i].plot(
        data.aggregated.index.total_seconds(),
        data.aggregated["y_pos_mm"],
        color=colors_normal[i],
        linestyle="-",
    )
    ax[1, i].plot(
        data.aggregated.index.total_seconds(),
        data.aggregated["x_vel_mm_per_s"],
        color=colors_normal[i],
        linestyle="-",
    )
    ax[3, i].plot(
        data.aggregated.index.total_seconds(),
        data.aggregated["y_vel_mm_per_s"],
        color=colors_normal[i],
        linestyle="-",
    )
    ax[4, i].plot(
        data.aggregated.index.total_seconds(),
        data.aggregated["s_vel_deg_per_s"] / 360 * 60,
        color=colors_normal[i],
        linestyle="-",
    )

ax[4, 0].set_xlabel("Time in s" r" $\rightarrow$")
ax[4, 1].set_xlabel("Time in s" r" $\rightarrow$")
ax[4, 0].set_xlim(0, 120)
ax[4, 0].set_xticks(np.arange(0, 121, 40))
ax[0, 0].set_ylabel("$\it{X}$-position\nin mm" r" $\rightarrow$")
ax[1, 0].set_ylabel("$\it{X}$-velocity\nin mm/s" r" $\rightarrow$")
ax[2, 0].set_ylabel("$\it{Y}$-position\nin mm" r" $\rightarrow$")
ax[3, 0].set_ylabel("$\it{Y}$-velocity\nin mm/s" r" $\rightarrow$")
ax[4, 0].set_ylabel("Spindle speed\nin 1/min" r" $\rightarrow$")
ax[4, 0].set_ylim(400, 420)

freq_handles = [
    Line2D([0], [0], color=COLOR_DEFAULT_BLUE, linestyle="-", label="500 Hz"),
    Line2D([0], [0], color=COLOR_DEFAULT_BLACK, linestyle="-", label="10 Hz"),
    Line2D([0], [0], color=COLOR_LIGHT_BLUE, linestyle="-", label=r"$\ldots$ with 0.5 noise addition"),
    Line2D([0], [0], color=COLOR_LIGHT_GREY, linestyle="-", label=r"$\ldots$ with 0.5 noise addition"),
]
fig.legend(handles=freq_handles, ncol=2, loc="upper center", bbox_to_anchor=(0.55, 0.035), frameon=True)

fig.suptitle("Reference part 2 – face milling")
fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(f"plots/Fig{fig_no}.pdf")
fig.show()
fig_no += 1

seeds = list()
rates = list()
noise = list()
for name, (data, cfg) in results_reversal_face.items():
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


with open(folder / f"{prefix}state_monitoring_deviations.pkl", "rb") as f:
    results_cm_face = pickle.load(f)
with open(folder / f"{prefix}predictive_quality_deviations.pkl", "rb") as f:
    results_pq_face = pickle.load(f)

fig, ax = plt.subplots(1, 1, figsize=(one_column_width(), one_column_width() / 1.6), sharex=True)
ax = [ax]
scenario = "position"
for i, sampling_rate in enumerate(rates):
    distance_by_noise = defaultdict(list)
    for seed in seeds:
        series = results_pq_face.loc[(scenario, sampling_rate, slice(None)), (seed, "none")]
        for idx, val in series.items():
            if val is not None:
                distance_by_noise[idx[2]].append(val)
    noises, mu, sigma = gaussian_stats(distance_by_noise)
    plot_with_boundary_lines(ax[0], noises, mu, sigma, f"C{i}", marker="x")
ax[0].set_yticks(np.arange(0, 151, 50))
ax[0].set_xlim(0, 1)
ax[0].set_ylabel("Position abs. dev.\n" r"in mm $\rightarrow$")
ax[0].set_title("Utility loss in predictive quality")

ax[0].set_xlabel(r"Noise standard deviation in $\sigma_\text{n}/\sigma_\text{s} \rightarrow$")

time_handles = [
    Line2D([0], [0], color="C0", label="500 Hz", marker="x"),
    Line2D([0], [0], color="C1", label="10 Hz", marker="+"),
]
fig.legend(
    handles=time_handles,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.58, 0.02),
    frameon=True,
)
fig.suptitle("Reference part 2 – face milling")
fig.tight_layout(pad=pad, rect=[0, 0, 1, 1])
fig.savefig(f"plots/Fig{fig_no}.pdf")
fig.show()
fig_no += 1

del results_pq_face, results_cm_face

with open(folder / f"{prefix}operating_states.pkl", "rb") as f:
    results_operating_states_face = pickle.load(f)
results_operating_states_face_true = results_operating_states_face.loc[("all", 2.0, 0.0), (0, "kalman")]
with open(folder / f"{prefix}feeds.pkl", "rb") as f:
    results_feed_face = pickle.load(f)
with open(folder / f"{prefix}radial_engagements.pkl", "rb") as f:
    results_radial_engagement_face = pickle.load(f)

s = 15
feed = 0.5
radial_engagement = 50
tool_radius = 116 / 2

fig, ax = plt.subplots(3, 3, figsize=(two_column_width(), two_column_width() / 1.65), sharex=True, sharey="row")
scenario_to_title = {
    "all": "Position and velocity data available",
    "position": "Position data available",
    "velocity": "Velocity data available",
}

for j, scenario in enumerate(("all", "position", "velocity")):
    for i, sampling_rate in enumerate(rates):
        g1_by_noise = defaultdict(list)
        for seed in seeds:
            sub_df = results_operating_states_face.loc[(scenario, sampling_rate, slice(None)), ([seed], "kalman")]
            for idx, row in sub_df.iterrows():
                noise_level = idx[2]
                for _, row_2 in row.items():
                    if not isinstance(row_2, (list, tuple, np.ndarray)):
                        continue
                    g1_by_noise[noise_level].append(row_2[1] * 100)

        noises, mu, sigma = gaussian_stats(g1_by_noise)

        plot_with_boundary_lines(ax[0, j], noises, mu, sigma, f"C{i}", marker=sampling_rate_to_marker[sampling_rate])
    ax[0, j].axhline(results_operating_states_face_true[1] * 100, color=fc.COLOR_GREY_50, linestyle="-", zorder=6)
    ax[0, j].set_ylim(70, 100)
    ax[0, j].set_title(scenario_to_title[scenario])
ax[0, 0].set_ylabel("Recon. ratio of\n" r"cutting seg. in %" r" $\rightarrow$")
for j, scenario in enumerate(("all", "position", "velocity")):
    for i, sampling_rate in enumerate(rates):
        feeds_by_noise = defaultdict(list)
        for seed in seeds:
            sub_df = results_feed_face.loc[(scenario, sampling_rate, slice(None)), ([seed], method)]
            for idx, row in sub_df.iterrows():
                noise_level = idx[2]
                for _, row_2 in row.items():
                    if not isinstance(row_2, (list, tuple, np.ndarray)):
                        continue
                    for segment in row_2:
                        if segment.feed is None:
                            continue
                        feeds_by_noise[noise_level].append(segment.feed)

        noises, mu, sigma = gaussian_stats(feeds_by_noise)
        plot_with_boundary_lines(ax[1, j], noises, mu, sigma, f"C{i}", marker=sampling_rate_to_marker[sampling_rate])
    ax[1, j].axhline(feed, color=fc.COLOR_GREY_50, linestyle="--", zorder=6)
    ax[1, j].set_ylim(0.4, 1.2)
    ax[1, j].set_yticks(np.arange(0.4, 1.21, 0.2))

ax[1, 0].set_ylabel("Recon.\n" r"$f_\text{z}$ in mm" r" $\rightarrow$")
for j, scenario in enumerate(("all", "position", "velocity")):
    for i, sampling_rate in enumerate(rates):
        ae_by_noise = defaultdict(list)
        for seed in results_radial_engagement_face.columns.get_level_values(0).unique():
            series = results_radial_engagement_face.loc[(scenario, sampling_rate, slice(None)), (seed, method)]
            for idx, val in series.items():
                if isinstance(val, list):
                    ae_by_noise[idx[2]].append(abs(val[0]))
        noises, mu, sigma = gaussian_stats(ae_by_noise)
        plot_with_boundary_lines(ax[2, j], noises, mu, sigma, f"C{i}", marker=sampling_rate_to_marker[sampling_rate])
    ax[2, j].axhline(radial_engagement, color=fc.COLOR_GREY_50, linestyle="-.", zorder=6)
    ax[2, j].set_ylim(30, 70)
    ax[2, j].set_yticks(np.arange(30, 71, 10))
ax[2, 0].set_ylabel("Recon.\n" + r"$a_\text{e}$ in mm" r" $\rightarrow$")

ax[0, 0].set_xlim(0, 1)
ax[2, 1].set_xlabel(r"Noise standard deviation in $\sigma_\text{n}/\sigma_\text{s}$" r" $\rightarrow$")

time_handles = [
    Line2D([0], [0], color="C0", label="500 Hz", marker="x"),
    Line2D([0], [0], color="C1", label="10 Hz", marker="+"),
]
fig.legend(
    handles=time_handles,
    ncol=len(rates),
    title="Sampling rate",
    loc="upper left",
    bbox_to_anchor=(0.62, 0.02),
    frameon=True,
)
param_handles = [
    Line2D([0], [0], color=fc.COLOR_GREY_50, label=r"Ratio of cutting segments", linestyle="-"),
    Line2D([0], [0], color=fc.COLOR_GREY_50, label=r"Finishing $f_\text{z}$", linestyle="--"),
    Line2D([0], [0], color=fc.COLOR_GREY_50, label=r"Finishing $a_\text{e}$", linestyle="-."),
]
fig.legend(
    handles=param_handles,
    ncol=3,
    title="True values",
    loc="upper right",
    bbox_to_anchor=(0.63, 0.02),
    frameon=True,
)

fig.suptitle("Reference part 2 – face milling")
fig.tight_layout(pad=pad, rect=[0, 0, 1, 1])
fig.savefig(f"plots/Fig{fig_no}.pdf")
fig.show()
fig_no += 1


fig, ax = plt.subplots(1, 1, sharex=True, figsize=(one_column_width(), one_column_width() / 1.23))
data, cfg = results_reversal_face["VELOCITY_ds_2_b_0.5_seed_0"]
data_reversed = data.reversed[method]
toolpath = data_reversed[["x_pos_mm", "y_pos_mm"]].to_numpy()
toolpath_line = shapely.geometry.LineString(toolpath)

polygon = toolpath_line.buffer(tool_radius)

polygon_tool_entrance = shapely.LineString(
    [
        [data_reversed["x_pos_mm"].min(), data_reversed["y_pos_mm"].min() - tool_radius],
        [data_reversed["x_pos_mm"].min(), data_reversed["y_pos_mm"].max() + tool_radius],
    ]
)
polygon_tool_entrance = polygon_tool_entrance.buffer(tool_radius)
polygon_tool_exit = shapely.box(
    data_reversed["x_pos_mm"].max(),
    data_reversed["y_pos_mm"].min() - tool_radius,
    data_reversed["x_pos_mm"].max() + tool_radius * 1.1,
    data_reversed["y_pos_mm"].max() + tool_radius,
)
ae = abs(results_radial_engagement_face.loc[("velocity", 2, 0.5), (0, method)][0])
polygon_ae_limit_upper = shapely.box(
    data_reversed["x_pos_mm"].min(),
    data_reversed["y_pos_mm"].max() - tool_radius + radial_engagement,
    data_reversed["x_pos_mm"].max() + tool_radius * 1.1,
    data_reversed["y_pos_mm"].max() + 3 * tool_radius,
)
polygon_ae_limit_lower = shapely.box(
    data_reversed["x_pos_mm"].min(),
    data_reversed["y_pos_mm"].min() - radial_engagement,
    data_reversed["x_pos_mm"].max() + tool_radius * 1.1,
    data_reversed["y_pos_mm"].min() - 3 * tool_radius,
)
probable_surface_are = (
    polygon - polygon_tool_entrance - polygon_tool_exit - polygon_ae_limit_upper - polygon_ae_limit_lower
)

shapely.plotting.plot_polygon(
    union_true,
    ax,
    add_points=False,
    edgecolor=COLOR_DEFAULT_BLACK,
    facecolor=to_rgba(COLOR_DEFAULT_BLACK, 0.3),
    zorder=2,
)
shapely.plotting.plot_polygon(
    polygon,
    ax,
    add_points=False,
    edgecolor=COLOR_DEFAULT_BLUE,
    facecolor=to_rgba(COLOR_DEFAULT_BLUE, 0.3),
    zorder=2,
)
shapely.plotting.plot_polygon(
    probable_surface_are,
    ax,
    add_points=False,
    edgecolor=fc.COLOR_ACCENT_ORANGE,
    facecolor=to_rgba(fc.COLOR_ACCENT_ORANGE, 0.3),
    zorder=2,
)
ax.plot(
    data_reversed["x_pos_mm"],
    data_reversed["y_pos_mm"],
    label="Toolpath",
    alpha=1.0,
    color=COLOR_DEFAULT_BLACK,
    linewidth=lw,
)
dx = data_reversed["x_pos_mm"].diff()
dy = data_reversed["y_pos_mm"].diff()
length = np.linalg.norm(np.array([dx, dy]).T, axis=1)
feedrate = (data_reversed["x_vel_mm_per_s"] ** 2 + data_reversed["y_vel_mm_per_s"] ** 2) ** 0.5
dx = dx / length
dy = dy / length
pts = list()
dist_acc = 0
row_prev = None
for idx, row in data_reversed.iterrows():
    if idx == data_reversed.index[0]:
        row_prev = row
        continue
    dist_acc += ((row["x_pos_mm"] - row_prev["x_pos_mm"]) ** 2 + (row["y_pos_mm"] - row_prev["y_pos_mm"]) ** 2) ** 0.5
    if dist_acc >= dist_min:
        pts.append(idx)
        dist_acc = 0
    row_prev = row
ax.quiver(
    data_reversed.loc[pts, "x_pos_mm"],
    data_reversed.loc[pts, "y_pos_mm"],
    dx.loc[pts].values,
    dy[pts].values,
    angles="xy",
    scale_units="xy",
    scale=0.05,
    width=0.001,
    headlength=20,
    headaxislength=18,
    headwidth=20,
    color=COLOR_DEFAULT_BLACK,
    zorder=3,
)
ax.set_title("Velocity data available,\n500 Hz, noise addition of 0.5")

ax.set_aspect("equal", "datalim", share=True)
ax.set_xlabel(r"$\it{X}$-axis position in mm →")
ax.set_ylabel(r"$\it{Y}$-axis position in mm →")

freq_handles = [
    plt.Rectangle(
        (0, 0),
        0,
        0,
        transform=ax.transAxes,
        label="True part\ngeometry",
        fc=to_rgba(COLOR_DEFAULT_BLACK, 0.3),
        ec=COLOR_DEFAULT_BLACK,
    ),
    Line2D([0], [0], color=COLOR_DEFAULT_BLACK, linestyle="-", label="Reversed toolpath", linewidth=lw),
    plt.Rectangle(
        (0, 0),
        0,
        0,
        transform=ax.transAxes,
        label="Swept surface\narea",
        fc=to_rgba(COLOR_DEFAULT_BLUE, 0.3),
        ec=COLOR_DEFAULT_BLUE,
    ),
    plt.Rectangle(
        (0, 0),
        0,
        0,
        transform=ax.transAxes,
        label="Inferred part\ngeometry",
        fc=to_rgba(fc.COLOR_ACCENT_ORANGE, 0.3),
        ec=fc.COLOR_ACCENT_ORANGE,
    ),
]
fig.legend(
    handles=freq_handles,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.55, 0.02),
    frameon=True,
)

fig.suptitle("Reference part 2 – face milling")
fig.tight_layout(pad=pad, rect=[0, 0, 1, 1])
fig.savefig(f"plots/Fig{fig_no}.pdf")
fig.show()
fig_no += 1
