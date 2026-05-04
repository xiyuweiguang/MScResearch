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
Plot Bitcoin2014/test component comparison with paper-friendly component names.

This version renames:
    k1       -> InitResidualAndExcess
    k7       -> SourcePreflow
    init_hl  -> Height-label Init
    bfs      -> BFS
    pr       -> Push-Relabel
    rm       -> RM

Expected input files:
    topoPR_test_log.txt
    region_test_log.txt

Outputs:
    bitcoin2014_figures/
        bitcoin2014_component_mean_comparison.png
        bitcoin2014_bfs_rm_overhead_by_query.png
        bitcoin2014_<part>_time_by_query.png
"""

from pathlib import Path
import re
import statistics as stats
import pandas as pd
import matplotlib.pyplot as plt


TOPO_LOG = Path("TDPR_log.txt")
REGION_LOG = Path("RA_log.txt")
OUT_DIR = Path("bitcoin2014_figures")


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    text = re.sub(r"(?=Query \d+:)", "\n", text)
    text = re.sub(r"(?=\[Init\])", "\n", text)
    text = re.sub(r"(?=Iteration \d+:)", "\n", text)

    queries = {}
    cur = None

    for line in text.splitlines():
        q = re.search(r"Query\s+(\d+):", line)
        if q:
            cur = int(q.group(1))
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

    rows = {}
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

        rows[qid] = row

    return rows


def main() -> None:
    if not TOPO_LOG.exists():
        raise FileNotFoundError(f"Missing input file: {TOPO_LOG}")
    if not REGION_LOG.exists():
        raise FileNotFoundError(f"Missing input file: {REGION_LOG}")

    OUT_DIR.mkdir(exist_ok=True)

    topo = parse_log(TOPO_LOG)
    region = parse_log(REGION_LOG)
    common_queries = sorted(set(topo) & set(region))

    records = []
    for qid in common_queries:
        for method_name, data in [
            ("Topology-Driven", topo),
            ("Region-Aware", region),
        ]:
            row = data[qid].copy()
            row["method"] = method_name
            records.append(row)

    per_query = pd.DataFrame(records)

    summary_rows = []
    for part in ["k1", "k7", "init_hl", "bfs", "pr", "rm"]:
        topo_vals = [topo[q][part] for q in common_queries]
        region_vals = [region[q][part] for q in common_queries]

        topo_mean = sum(topo_vals) / len(topo_vals)
        region_mean = sum(region_vals) / len(region_vals)

        summary_rows.append({
            "part": part,
            "topology_driven_mean_ms": topo_mean,
            "region_aware_mean_ms": region_mean,
            "topology_driven_median_ms": stats.median(topo_vals),
            "region_aware_median_ms": stats.median(region_vals),
            "topology_to_region_mean_ratio": topo_mean / region_mean if region_mean != 0 else None,
        })

    summary = pd.DataFrame(summary_rows)

    label_map = {
        "k1": "InitResidual\nAndExcess",
        "k7": "Source\nPreflow",
        "init_hl": "Height-label\nInit",
        "bfs": "BFS",
        "pr": "Push-\nRelabel",
        "rm": "RM",
    }

    # Mean component comparison.
    plt.figure(figsize=(9.2, 4.9))
    x = range(len(summary))
    bar_width = 0.35

    plt.bar(
        [i - bar_width / 2 for i in x],
        summary["topology_driven_mean_ms"],
        width=bar_width,
        label="Topology-Driven PR",
    )
    plt.bar(
        [i + bar_width / 2 for i in x],
        summary["region_aware_mean_ms"],
        width=bar_width,
        label="Region-Aware PR",
    )

    plt.xticks(list(x), [label_map[p] for p in summary["part"]])
    plt.yscale("log")
    plt.xlabel("Kernel / stage")
    plt.ylabel("Mean time per query (ms, log scale)")
    plt.title("Bitcoin2014: Mean Component Time Comparison")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bitcoin2014_component_mean_comparison.png", dpi=300)
    plt.close()

    # BFS + RM overhead by query.
    pivot = per_query.pivot(index="query", columns="method", values=["bfs", "rm"])
    queries = sorted(per_query["query"].unique())

    topo_overhead = []
    region_overhead = []
    for q in queries:
        topo_overhead.append(
            pivot.loc[q, ("bfs", "Topology-Driven")]
            + pivot.loc[q, ("rm", "Topology-Driven")]
        )
        region_overhead.append(
            pivot.loc[q, ("bfs", "Region-Aware")]
            + pivot.loc[q, ("rm", "Region-Aware")]
        )

    plt.figure(figsize=(9.2, 4.9))
    plt.plot(
        queries,
        topo_overhead,
        marker="o",
        label="Topology-Driven PR: BFS + RM",
    )
    plt.plot(
        queries,
        region_overhead,
        marker="s",
        label="Region-Aware PR: BFS + RM",
    )

    plt.yscale("log")
    plt.xlabel("Query index")
    plt.ylabel("Total BFS + RM time (ms, log scale)")
    plt.title("Bitcoin2014: Per-Query Relabeling/Repair Overhead")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bitcoin2014_bfs_rm_overhead_by_query.png", dpi=300)
    plt.close()

    # Individual per-component charts.
    component_display = {
        "k1": "InitResidualAndExcess",
        "k7": "SourcePreflow",
        "init_hl": "Height-label initialization",
        "bfs": "BFS",
        "pr": "Push-relabel discharge",
        "rm": "Residual repair",
    }

    for part, display in component_display.items():
        topo_vals = per_query[per_query["method"] == "Topology-Driven"].sort_values("query")
        region_vals = per_query[per_query["method"] == "Region-Aware"].sort_values("query")

        plt.figure(figsize=(9.2, 4.9))
        plt.plot(topo_vals["query"], topo_vals[part], marker="o", label="Topology-Driven PR")
        plt.plot(region_vals["query"], region_vals[part], marker="s", label="Region-Aware PR")
        plt.xlabel("Query index")
        plt.ylabel("Time (ms)")
        plt.title(f"Bitcoin2014: Per-query {display} time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"bitcoin2014_{part}_time_by_query.png", dpi=300)
        plt.close()

    print(f"Parsed {len(common_queries)} common queries.")
    print(f"Outputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
