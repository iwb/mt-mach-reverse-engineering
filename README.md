# Reverse Engineering of Protected CNC Machine-Tool Data

<!--
---------------------------------------------------------
TOC
---------------------------------------------------------
-->
## Table of Contents
- [Contact](#contact)
- [Introduction](#introduction)
- [Resources](#resources)
- [Usage Instructions](#usage-instructions)
- [Citation](#citation)
- [License](#license)


## Contact
### Contact Details
Corresponding author: Daniel Piendl \
Institutional email: daniel.piendl@iwb.tum.de \
Personal profile: https://github.com/dpiendl


### Useful Links
- **[Visit our other repositories](https://iwb.github.io)**
Explore more tools and resources from our research institute.

- **[Visit our institute for more information](https://www.mec.ed.tum.de/en/iwb/homepage/)**
Learn more about our research and ongoing projects.


<!--
---------------------------------------------------------
Introduction
---------------------------------------------------------
-->
## Introduction

This repository provides tools to study the **privacy–utility trade-off of protected CNC
machine-tool measurement data**. Modern machine tools expose high-frequency axis signals
(positions, velocities, spindle speed). Sharing them enables condition monitoring and
analytics, but also leaks sensitive process know-how: tool paths, part geometry, and process 
parameters can be reconstructed from them. The tools quantify how much of
that information survives a given level of protection.

The pipeline has three stages:

1. **Protection** (data owner's defense): obfuscate a signal by *downsampling*, *adding
   noise*, and *suppressing* channels (e.g., publishing only positions or only velocities).
2. **Reversal / attack** (adversary): undo the protection with denoising/estimation filters
   (Kalman/RTS smoother, Savitzky–Golay, spline, Butterworth) and reconstruct the missing
   channels (integrate velocity → position, differentiate position → velocity).
3. **Disclosure**: extract process information from the reconstruction — operating states
   (rapid traverse `G0`, cutting `G1`, standstill), feed per tooth, radial engagement, and
   machined workpiece geometry.

The pipeline is evaluated over a grid of protection settings (sampling rate, noise level,
available channels, random seeds) to measure the residual utility of protected data.


<!--
---------------------------------------------------------
Research Article
---------------------------------------------------------
-->
### Related Research Work

These tools are part of the research published in the following article:

**"Reverse Engineering of Part Geometries and Machining Process Parameters from CNC Machining Time Series Measurements"** \
*Daniel Piendl, Charlotte Winkler, Yair Shneor, Andy Izaber-Ludwig, Moritz Goeldner, Laura Zinnel, Jannik Huellemann, Michael F. Zaeh*
Published in The International Journal of Advanced Manufacturing Technology in 2026.

For more details, please refer to the published article:
https://link.springer.com/article/10.1007/s00170-026-18585-6 \
DOI: 10.1007/s00170-026-18585-6

### Abstract
Manufacturing companies increasingly share Computerized Numerical Control (CNC) machining data via industrial data ecosystems and vendor cloud platforms.
While these approaches promise improved productivity, they also raise confidentiality risks: seemingly low-risk time series measurements can leak confidential information about part geometries and process parameters if obtained by malicious actors.
In this work, it was investigated to what extent confidential geometric and process information can be reconstructed from CNC time series measurements, and how privacy-preserving technologies can be used to mitigate this risk.
A five-step methodology was applied: (1) acquisition of CNC data, (2) protection of the axis time series by signal suppression, sampling rate reduction, and noise addition, (3) evaluation of the remaining data utility, (4) reversal of the protection using a filter-based approach, and (5) reconstruction of the confidential information using machining-domain knowledge.
The approach was evaluated on time series recorded during a peripheral milling and a face milling process of two reference parts.
The results show that suppressing only the axis positions is insufficient, as the toolpath can still be reconstructed from the axis velocities, and that reducing the sampling rate to 10 Hz combined with moderate noise addition hinders the reconstruction at the cost of a substantial loss in data utility.
At the original sampling rate of 500 Hz, the feed per tooth and certain geometric features could still be reconstructed even under high noise addition.
The degree of protection and the remaining data utility varied between the two reference parts, indicating that no general recommendation for privacy-preserving technology parameters can be given at this stage.
Instead, the study provides a methodology for assessing the leakage risk and selecting privacy-preserving technology parameters on a per-part basis when sharing CNC machining data.


### Acknowledgements
This work was supported by the German Federal Ministry of Research, Technology, and Space (BMFTR) within the research project “Secure Collaborative Machine Tool Data Utilization Leveraging Privacy-Enhancing Technologies (MINERVA)” under the grant number 16KIS1805.


<!--
---------------------------------------------------------
Resources
---------------------------------------------------------
-->
## Resources

The repository is organized as an installable library plus reproducible experiments:

```
.
├── data/                        # Example CNC measurements (CSV)
│   ├── contour_milling.csv      #   Reference part 1 – contour milling
│   └── face_milling.csv         #   Reference part 2 – face milling
├── src/reverse_engineering/     # The installable library (reusable API)
├── scripts/                     # Batch experiments that reproduce the paper results
│   └── visualize_results/       #   Scripts/notebooks that render the paper figures
├── notebooks/
│   └── single_use_case.ipynb    # Start here: end-to-end walkthrough on one signal
└── results/                     # Created by the scripts; holds intermediate .pkl files
```

### 1. `reverse_engineering` Python library
The library (`src/reverse_engineering/`) exposes the full pipeline as composable functions:
- **Protect** a measurement — `protection.apply_protection` (downsampling, noise, suppression).
- **Reverse** the protection — `reversal.reverse_protection` (Kalman/Savgol/spline/Butterworth).
- **Segment** a trajectory into motion states — `velocity_segmentation.create_velocity_segments`.
- **Disclose** process info — `reconstruction.disclose_*` (operating states, feed, radial
  engagement, machined geometry).

### 2. Example walkthrough notebook
`notebooks/single_use_case.ipynb` runs the whole pipeline on a single machining run — load →
protect → reverse → segment → disclose — with plots at every step. It is the fastest way to
understand and reuse the method.

### Data format
Each CSV in `data/` is one machining run, sampled at 500 Hz (2 ms) and indexed by elapsed time:

| Column            | Unit    | Description                                               |
| ----------------- | ------- | --------------------------------------------------------- |
| `timestamp_utc`   | —       | Elapsed time, parsed as a `TimedeltaIndex` (index column) |
| `x_pos_mm`        | mm      | X-axis position                                           |
| `y_pos_mm`        | mm      | Y-axis position                                           |
| `x_vel_mm_per_s`  | mm/s    | X-axis velocity                                           |
| `y_vel_mm_per_s`  | mm/s    | Y-axis velocity                                           |
| `s_vel_deg_per_s` | deg/s   | Spindle rotational speed                                  |

To use your own data, provide a CSV with the same columns (index column first).


<!--
---------------------------------------------------------
Usage Instructions
---------------------------------------------------------
-->
## Usage Instructions

### Installation
The project targets **Python 3.11**. Install with plain `pip`/`venv`:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -e .
```

or with [PDM](https://pdm-project.org/):

```bash
pdm install
```

A pinned `requirements.txt` (exported from the PDM lock file) is also provided for exact
reproduction.

### Quickstart (single use case)
Open the notebook for the fully visualized walkthrough:

```bash
jupyter notebook notebooks/single_use_case.ipynb
```

The minimal programmatic version:

```python
from reverse_engineering.classes import (
    AttackConfig, CncDataTransformations, EvalConfig, ProtectionConfig,
)
from reverse_engineering.data_loading import (
    DataAvailabilityScenarios, load_csv_measurement_data,
)
from reverse_engineering.protection import apply_protection
from reverse_engineering.reversal import reverse_protection

# 1. Load a measurement
df = load_csv_measurement_data("data/contour_milling.csv")
data = CncDataTransformations(original=df, dt_original=df.index.diff().median())

# 2. Configure protection (defender) and attack (adversary)
cfg = EvalConfig(
    protection=ProtectionConfig(
        data_availability_scenario=DataAvailabilityScenarios.ALL,  # publish pos + vel
        noise_standard_deviation_multiplier=0.3,                   # 0.3 * signal std
        downsampling_rate_ms=100,                                  # 500 Hz -> 10 Hz
        keep_original_index_before_downsampling=False,
        random_state=0,
    ),
    attack=AttackConfig(
        start_pos_known=True, end_pos_known=False,
        noise_estimation_window_duration_s=0.1,
    ),
)

# 3. Protect, then attempt to reverse the protection
data = apply_protection(data, cfg.protection)
data = reverse_protection(data, cfg, methods=["none", "kalman"])
reconstructed = data.reversed["kalman"]   # DataFrame with reconstructed pos + vel
```

### Reproducing the paper results
The scripts are numbered in execution order and communicate through pickle files in
`results/`. Run them from the repository root:

```bash
python scripts/1_protection_of_data.py      # apply protection over the config grid
python scripts/2_remaining_data_utility.py  # residual utility of *protected* data
python scripts/3_reversal_of_protection.py  # reverse protection (grid of filters)
python scripts/4_velocity_segmentation.py   # segment reconstructions into G0/G1/standstill
python scripts/5_operating_states.py        # disclose operating-state fractions
python scripts/6_feed.py                     # disclose feed per tooth
python scripts/6_radial_engagement.py        # disclose radial engagement
python scripts/6_workpiece_geometry.py       # disclose machined geometry
# finally, render the figures in scripts/visualize_results/
```

Steps 1, 3, and 4 parallelize over the configuration grid and can take a while.


<!--
---------------------------------------------------------
Citation
---------------------------------------------------------
-->
## Citation

If you use this repository or the tools for your research or industry projects, please cite
the following article:

```bibtex
@article{Piendl.2026,
 author = {Piendl, Daniel and Winkler, Charlotte and Shneor, Yair and Izaber-Ludwig, Andy and Goeldner, Moritz and Zinnel, Laura and Huellemann, Jannik and Zaeh, Michael F.},
 year = {2026},
 title = {Reverse engineering of part geometries and machining process parameters from CNC machining time series measurements},
 issn = {0268-3768},
 journal = {The International Journal of Advanced Manufacturing Technology},
 doi = {10.1007/s00170-026-18585-6}
}
```

<!--
---------------------------------------------------------
License
---------------------------------------------------------
-->
## License
This repository and its contents are licensed under the **MIT License**. See the
[LICENSE](./LICENSE) file for more details.


<!--
---------------------------------------------------------
Footer
---------------------------------------------------------
-->
---
For questions, suggestions, or collaboration opportunities, please contact the corresponding
author or visit our [institute website](https://www.mec.ed.tum.de/en/iwb/homepage/).
