#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Project ：COMPARISON
@File    ：dp_compared_on_bit2014.py
@IDE     ：PyCharm
@Author  ：Winston·H DONG
@Date    ：2026/5/4 22:13
'''

"""
Plot ECL-MaxFlow vs Region-Aware GPU PR on three datasets.

Current expected directory layout:

.
├── ECL-MaxFlow/
│   ├── IBM_LIS/
│   │   ├── IBM_LIS_log.txt
│   │   └── res_mt.txt
│   ├── PROS/
│   │   ├── PROS_log.txt
│   │   └── res_mt.txt
│   └── test/
│       ├── test_log.txt
│       └── res_mt.txt
├── Region-Aware GPU PR/
│   ├── IBM_LIS/
│   │   ├── IBM_LIS_log.txt
│   │   └── res_mt.txt
│   ├── PROS/
│   │   ├── PROS_log.txt
│   │   └── res_mt.txt
│   └── test/
│       ├── test_log.txt
│       └── res_mt.txt
└── plot_three_datasets_ecl_labels.py

Outputs:
    motivation_three_datasets_figures_ecl/
        three_dataset_mean_runtime_comparison_ecl.png
        three_dataset_component_breakdown_ecl.png
        three_dataset_total_runtime_summary.csv
        three_dataset_speedup_summary.csv
        three_dataset_component_summary.csv
        three_dataset_component_breakdown_ecl_for_plot.csv
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


BASE = Path(".")
ECL_ROOT = BASE / "ECL-MaxFlow"
REGION_ROOT = BASE / "Region-Aware GPU PR"
OUT_DIR = BASE / "motivation_three_datasets_figures_ecl"

DATASET_MAP = {
    "test": "Bitcoin2014",
    "PROS": "PROS",
    "IBM_LIS": "IBM_LIS",
}


def parse_res_mt(path: Path) -> pd.DataFrame:
    """Parse res_mt.txt where each line is: flow time_ms."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    rows = []
    for idx, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            rows.append({
                "query": idx,
                "flow": float(parts[0]),
                "total_time_ms": float(parts[1]),
            })
        except ValueError:
            continue

    return pd.DataFrame(rows)


def parse_ecl_detailed_log(path: Path) -> pd.DataFrame:
    """
    Parse ECL-MaxFlow detailed log.

    ECL logs expose:
      - global relabeling time
      - push-relabel kernel time
      - runtime

    They do not expose k1/k7/RM with the same per-iteration format as the
    Region-Aware GPU PR logs. Therefore, for component plotting:
      k1 = 0, k7 = 0, init_hl = 0, rm = 0
    and the exposed global-relabeling time is mapped to bfs.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    text = path.read_text(errors="ignore")

    blocks = []
    current = []
    for line in text.splitlines():
        current.append(line)
        if line.startswith("runtime:"):
            blocks.append("\n".join(current))
            current = []

    rows = []
    for qid, block in enumerate(blocks, start=1):
        runtime_match = re.search(r"runtime:\s*([0-9.]+)s", block)
        runtime_ms = float(runtime_match.group(1)) * 1000 if runtime_match else None

        pr_ms = sum(
            float(x) * 1000
            for x in re.findall(r"push-relabel kernel time:\s*([0-9.]+)s", block)
        )

        bfs_ms = sum(
            float(x) * 1000
            for x in re.findall(
                r"(?:initial global relabel from (?:sink|source).*?in|"
                r"Global relabel from (?:sink|source).*?in)\s*([0-9.]+)s",
                block,
            )
        )

        rows.append({
            "query": qid,
            "total_time_ms_detailed": runtime_ms,
            "k1": 0.0,
            "k7": 0.0,
            "init_hl": 0.0,
            "bfs": bfs_ms,
            "pr": pr_ms,
            "rm": 0.0,
        })

    return pd.DataFrame(rows)


def parse_region_components(path: Path) -> pd.DataFrame:
    """Parse Region-Aware GPU PR logs with k1, k7, init_hl, bfs, pr, and rm."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    text = path.read_text(errors="ignore")

    # Make parsing robust to merged or broken log lines.
    text = re.sub(r"(?=Query \d+:)", "\n", text)
    text = re.sub(r"(?=\[Init\])", "\n", text)
    text = re.sub(r"(?=Iteration \d+:)", "\n", text)

    queries = {}
    cur = None

    for line in text.splitlines():
        qmatch = re.search(r"Query\s+(\d+):", line)
        if qmatch:
            cur = int(qmatch.group(1))
            queries.setdefault(cur, {"iters": [], "k1": None, "k7": None})

        if cur is None:
            continue

        init = re.search(
            r"kernel_1\(init residual/excess\)=([0-9.]+)\s*ms,\s*"
            r"kernel_7\(preflow\)=([0-9.]+)\s*ms",
            line,
        )
        if init:
            queries[cur]["k1"] = float(init.group(1))
            queries[cur]["k7"] = float(init.group(2))

        it = re.search(
            r"Iteration\s+(\d+):\s*"
            r"init_hl=([0-9.]+)\s*ms,\s*"
            r"bfs=([0-9.]+)\s*ms,\s*"
            r"push_relabel=([0-9.]+)\s*ms.*?"
            r"remove_invalid=([0-9.]+)\s*ms",
            line,
        )
        if it:
            queries[cur]["iters"].append({
                "init_hl": float(it.group(2)),
                "bfs": float(it.group(3)),
                "pr": float(it.group(4)),
                "rm": float(it.group(5)),
            })

    rows = []
    for qid, qdata in queries.items():
        if qdata["k1"] is None or not qdata["iters"]:
            continue

        row = {
            "query": qid,
            "k1": qdata["k1"],
            "k7": qdata["k7"],
            "iterations": len(qdata["iters"]),
        }

        for part in ["init_hl", "bfs", "pr", "rm"]:
            row[part] = sum(item[part] for item in qdata["iters"])

        rows.append(row)

    return pd.DataFrame(rows)


def collect_data():
    total_records = []
    component_records = []

    for dataset_dir, dataset_name in DATASET_MAP.items():
        log_name = "test_log.txt" if dataset_dir == "test" else f"{dataset_dir}_log.txt"

        # End-to-end runtime from res_mt.txt.
        ecl_res = parse_res_mt(ECL_ROOT / dataset_dir / "res_mt.txt")
        ecl_res["dataset"] = dataset_name
        ecl_res["method"] = "ECL-MaxFlow"
        total_records.append(ecl_res)

        region_res = parse_res_mt(REGION_ROOT / dataset_dir / "res_mt.txt")
        region_res["dataset"] = dataset_name
        region_res["method"] = "Region-Aware GPU PR"
        total_records.append(region_res)

        # Component-level measurements from logs.
        ecl_comp = parse_ecl_detailed_log(ECL_ROOT / dataset_dir / log_name)
        ecl_comp["dataset"] = dataset_name
        ecl_comp["method"] = "ECL-MaxFlow"
        component_records.append(ecl_comp)

        region_comp = parse_region_components(REGION_ROOT / dataset_dir / log_name)
        region_comp["dataset"] = dataset_name
        region_comp["method"] = "Region-Aware GPU PR"
        component_records.append(region_comp)

    total_df = pd.concat(total_records, ignore_index=True)
    comp_df = pd.concat(component_records, ignore_index=True)

    total_common = []
    comp_common = []

    for dataset_name in DATASET_MAP.values():
        common_total = (
            set(total_df[(total_df.dataset == dataset_name) &
                         (total_df.method == "ECL-MaxFlow")]["query"])
            &
            set(total_df[(total_df.dataset == dataset_name) &
                         (total_df.method == "Region-Aware GPU PR")]["query"])
        )
        total_common.append(total_df[
            (total_df.dataset == dataset_name) &
            (total_df["query"].isin(common_total))
        ])

        common_comp = (
            set(comp_df[(comp_df.dataset == dataset_name) &
                        (comp_df.method == "ECL-MaxFlow")]["query"])
            &
            set(comp_df[(comp_df.dataset == dataset_name) &
                        (comp_df.method == "Region-Aware GPU PR")]["query"])
        )
        comp_common.append(comp_df[
            (comp_df.dataset == dataset_name) &
            (comp_df["query"].isin(common_comp))
        ])

    return (
        pd.concat(total_common, ignore_index=True),
        pd.concat(comp_common, ignore_index=True),
    )


def plot_figures(total_common: pd.DataFrame, comp_common: pd.DataFrame) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    datasets = ["Bitcoin2014", "PROS", "IBM_LIS"]
    parts = ["k1", "k7", "init_hl", "bfs", "pr", "rm"]

    total_summary = (
        total_common
        .groupby(["dataset", "method"])["total_time_ms"]
        .agg(["mean", "median", "sum", "count"])
        .reset_index()
    )

    speedup_summary = total_summary.pivot(
        index="dataset", columns="method", values="mean"
    ).reset_index()
    speedup_summary["speedup"] = (
        speedup_summary["ECL-MaxFlow"] /
        speedup_summary["Region-Aware GPU PR"]
    )

    comp_summary = (
        comp_common
        .groupby(["dataset", "method"])[parts]
        .mean()
        .reset_index()
    )

    total_summary.to_csv(OUT_DIR / "three_dataset_total_runtime_summary.csv", index=False)
    speedup_summary.to_csv(OUT_DIR / "three_dataset_speedup_summary.csv", index=False)
    comp_summary.to_csv(OUT_DIR / "three_dataset_component_summary.csv", index=False)

    # Figure 1: mean end-to-end query runtime.
    ecl_means = [
        float(total_summary[
            (total_summary.dataset == d) &
            (total_summary.method == "ECL-MaxFlow")
        ]["mean"].iloc[0])
        for d in datasets
    ]

    region_means = [
        float(total_summary[
            (total_summary.dataset == d) &
            (total_summary.method == "Region-Aware GPU PR")
        ]["mean"].iloc[0])
        for d in datasets
    ]

    speedups = [ecl / region for ecl, region in zip(ecl_means, region_means)]

    plt.figure(figsize=(8.8, 4.8))
    x = range(len(datasets))
    w = 0.34

    plt.bar([i - w / 2 for i in x], ecl_means, width=w, label="ECL-MaxFlow")
    plt.bar([i + w / 2 for i in x], region_means, width=w, label="Region-Aware PR")

    plt.yscale("log")
    plt.xticks(list(x), datasets)
    plt.ylabel("Mean query runtime (ms, log scale)")
    # plt.xlabel("Dataset")
    plt.title("Mean End-to-End Query Runtime")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    ymax = max(max(ecl_means), max(region_means))
    plt.ylim(top=ymax * 2.2)

    for i, speedup in enumerate(speedups):
        plt.text(
            i,
            max(ecl_means[i], region_means[i]) * 1.08,
            f"{speedup:.1f}×",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "three_dataset_mean_runtime_comparison_ecl.png", dpi=300)
    plt.close()

    # Figure 2: component breakdown.
    plot_rows = []

    for d in datasets:
        ecl = comp_summary[
            (comp_summary.dataset == d) &
            (comp_summary.method == "ECL-MaxFlow")
        ].iloc[0]
        plot_rows.append({
            "dataset": d,
            "method": "ECL",
            "Initialization": 0.0,
            "BFS / Global relabel": ecl["bfs"],
            "Push-Relabel": ecl["pr"],
            "RM": 0.0,
        })

        region = comp_summary[
            (comp_summary.dataset == d) &
            (comp_summary.method == "Region-Aware GPU PR")
        ].iloc[0]
        plot_rows.append({
            "dataset": d,
            "method": "RA",
            "Initialization": region["k1"] + region["k7"] + region["init_hl"],
            "BFS / Global relabel": region["bfs"],
            "Push-Relabel": region["pr"],
            "RM": region["rm"],
        })

    plot_df = pd.DataFrame(plot_rows)
    plot_df.to_csv(OUT_DIR / "three_dataset_component_breakdown_ecl_for_plot.csv", index=False)

    plt.figure(figsize=(9.6, 5.2))

    xpos = []
    labels = []
    for i, d in enumerate(datasets):
        labels += ["ECL", "RA"]
        xpos += [i * 3, i * 3 + 0.9]

    components = [
        "Initialization",
        "BFS / Global relabel",
        "Push-Relabel",
        "RM",
    ]

    bottom = [0.0] * len(plot_df)
    for comp in components:
        vals = plot_df[comp].tolist()
        plt.bar(xpos, vals, bottom=bottom, label=comp, width=0.7)
        bottom = [b + v for b, v in zip(bottom, vals)]

    plt.yscale("log")
    plt.xticks(xpos, labels)
    plt.ylabel("Mean measured compared time (ms, log scale)")
    plt.title("Measured Runtime (ECL vs Region-Aware PR)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(ncol=2, fontsize=9)

    ymin, _ = plt.ylim()
    for i, d in enumerate(datasets):
        plt.text(i * 3 + 0.45, ymin * 1.3, d, ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "three_dataset_component_breakdown_ecl.png", dpi=300)
    plt.close()


def main() -> None:
    total_common, comp_common = collect_data()
    plot_figures(total_common, comp_common)
    print(f"Figures and CSV summaries written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
