# COMPARISON

This directory contains experimental logs, generated figures, and plotting scripts for comparing **Region-Aware GPU Push-Relabel (Region-Aware GPU PR)** with two GPU max-flow baselines, as well as experiments on sparse active-path depth settings.

## Overview

The experiments are organized into three parts:

1. **ECL-MaxFlow vs. Region-Aware GPU PR**
   - Compares end-to-end query runtime on three datasets:
     - `Bitcoin2014` (stored as `test`)
     - `PROS`
     - `IBM_LIS`
   - Includes raw logs, parsed summaries, and figures.

2. **TDPR vs. Region-Aware GPU PR**
   - Compares component-level runtime on the `Bitcoin2014` dataset.
   - Components include:
     - `InitResidualAndExcess`
     - `SourcePreflow`
     - `Height-label Init`
     - `BFS`
     - `Push-Relabel`
     - `RM`

3. **Sparse active-path depth experiments**
   - Compares different maximum path-depth settings such as `deeppath=64`, `deeppath=128`, and `deeppath=256`.
   - Includes cross-dataset runtime comparison and depth-sensitivity plots.

## Directory Structure

```text
COMPARISON/
├── ECL_RA/
│   ├── ECL-MaxFlow/
│   │   ├── IBM_LIS/
│   │   │   ├── IBM_LIS_log.txt
│   │   │   └── res_mt.txt
│   │   ├── PROS/
│   │   │   ├── PROS_log.txt
│   │   │   └── res_mt.txt
│   │   └── test/
│   │       ├── test_log.txt
│   │       └── res_mt.txt
│   ├── Region-Aware GPU PR/
│   │   ├── IBM_LIS/
│   │   │   ├── IBM_LIS_log.txt
│   │   │   └── res_mt.txt
│   │   ├── PROS/
│   │   │   ├── PROS_log.txt
│   │   │   └── res_mt.txt
│   │   └── test/
│   │       ├── test_log.txt
│   │       └── res_mt.txt
│   ├── motivation_three_datasets_figures_ecl/
│   │   ├── three_dataset_mean_runtime_comparison_ecl.png
│   │   ├── three_dataset_component_breakdown_ecl.png
│   │   ├── three_dataset_total_runtime_summary.csv
│   │   ├── three_dataset_speedup_summary.csv
│   │   ├── three_dataset_component_summary.csv
│   │   └── three_dataset_component_breakdown_ecl_for_plot.csv
│   └── plot_three_datasets_ecl_labels.py
│
├── TDPR_RA/
│   ├── TDPR_log.txt
│   ├── RA_log.txt
│   ├── bitcoin2014_figures/
│   │   ├── bitcoin2014_component_mean_comparison.png
│   │   ├── bitcoin2014_bfs_rm_overhead_by_query.png
│   │   ├── bitcoin2014_k1_time_by_query.png
│   │   ├── bitcoin2014_k7_time_by_query.png
│   │   ├── bitcoin2014_init_hl_time_by_query.png
│   │   ├── bitcoin2014_bfs_time_by_query.png
│   │   ├── bitcoin2014_pr_time_by_query.png
│   │   └── bitcoin2014_rm_time_by_query.png
│   └── plot_bitcoin2014_TDPR_RA_comparison.py
│
├── dp_compared_on_bit2014.py
├── dp_compared_on_six_datasets.py
├── sparse_active_path_depth_sensitivity.png
├── cross_dataset_runtime_active_path_variants.png
└── cross_dataset_runtime_active_path_variants.pdf
```

## Requirements

The plotting scripts require Python 3 and the following packages:

```bash
pip install pandas matplotlib numpy
```

`pandas` is required for the log-parsing comparison scripts. `numpy` is used by the active-path cross-dataset plotting script.

## How to Reproduce the Figures

### 1. ECL-MaxFlow vs. Region-Aware GPU PR

```bash
cd COMPARISON/ECL_RA
python plot_three_datasets_ecl_labels.py
```

This script reads:

```text
ECL-MaxFlow/
Region-Aware GPU PR/
```

and generates:

```text
motivation_three_datasets_figures_ecl/
├── three_dataset_mean_runtime_comparison_ecl.png
├── three_dataset_component_breakdown_ecl.png
├── three_dataset_total_runtime_summary.csv
├── three_dataset_speedup_summary.csv
├── three_dataset_component_summary.csv
└── three_dataset_component_breakdown_ecl_for_plot.csv
```

The main figures are:

- `three_dataset_mean_runtime_comparison_ecl.png`
- `three_dataset_component_breakdown_ecl.png`

### 2. TDPR vs. Region-Aware GPU PR on Bitcoin2014

```bash
cd COMPARISON/TDPR_RA
python plot_bitcoin2014_TDPR_RA_comparison.py
```

This script reads:

```text
TDPR_log.txt
RA_log.txt
```

and generates figures in:

```text
bitcoin2014_figures/
```

The most important figures are:

- `bitcoin2014_component_mean_comparison.png`
- `bitcoin2014_bfs_rm_overhead_by_query.png`

The remaining figures show per-query runtime changes for individual components.

### 3. Sparse Active-Path Depth Sensitivity

From the root `COMPARISON/` directory:

```bash
python dp_compared_on_bit2014.py
```

This generates:

```text
sparse_active_path_depth_sensitivity.png
```

The figure compares runtime under different maximum active-path depths on `Bitcoin2014` and `PROS`.

### 4. Cross-Dataset Active-Path Variants

From the root `COMPARISON/` directory:

```bash
python dp_compared_on_six_datasets.py
```

This generates:

```text
cross_dataset_runtime_active_path_variants.png
cross_dataset_runtime_active_path_variants.pdf
```

The figure compares active-path variants across six datasets.

## Data Format

### `res_mt.txt`

Each line corresponds to one source-sink query:

```text
<flow_value> <runtime_ms>
```

Example:

```text
22790000 433
5480 553
```

### Log files

The detailed logs contain query-level and component-level information, such as:

```text
Query 1: s=..., t=...
[Init] kernel_1(init residual/excess)=... ms, kernel_7(preflow)=... ms
Iteration 0: init_hl=... ms, bfs=... ms, push_relabel=... ms, remove_invalid=... ms
Flow=..., Time=... ms
```

For Region-Aware GPU PR, the parser extracts:

- `k1`: residual/excess initialization
- `k7`: source preflow
- `init_hl`: height-label initialization
- `bfs`: residual backward BFS / relabeling
- `pr`: push-relabel discharge
- `rm`: residual repair / RemoveInvalidEdges

For ECL-MaxFlow, the log exposes global relabeling and push-relabel kernel time. ECL-MaxFlow does not report a separate RM stage in the same format, so RM is not treated as a measured ECL component.

## Notes

- The `test` directory corresponds to the `Bitcoin2014` dataset.
- The Region-Aware implementation uses `typedef long long flow_t` for capacities and flow values, which avoids 32-bit integer overflow in large-capacity queries.
- Some generated figures use a logarithmic y-axis because runtimes differ substantially across datasets and methods.
- Existing `.png`, `.pdf`, and `.csv` files are included so the results can be inspected without rerunning the scripts.

## Suggested LaTeX Usage

Example for the ECL comparison figure:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.92\linewidth]{figures/three_dataset_mean_runtime_comparison_ecl.png}
    \caption{Mean end-to-end query runtime comparison between ECL-MaxFlow and Region-Aware GPU PR.}
    \label{fig:motivation_runtime}
\end{figure}
```

Example for the TDPR component comparison figure:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.92\linewidth]{figures/bitcoin2014_component_mean_comparison.png}
    \caption{Mean component-level runtime comparison on the Bitcoin2014 dataset.}
    \label{fig:bitcoin2014_component_mean}
\end{figure}
```

## Author

Winston H. Dong
