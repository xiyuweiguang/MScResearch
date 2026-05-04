# -*- coding: UTF-8 -*-
'''
@Project ：COMPARISON
@File    ：dp_compared_on_six_datasets.py
@IDE     ：PyCharm 
@Author  ：Winston·H DONG
@Date    ：2026/5/4 20:43 
'''
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data: runtime in minutes
# -----------------------------
datasets = ["Bitcoin2014", "PROS", "IBM_LIS", "IBM_LIM", "IBM_HIS", "IBM_HIM"]

deeppath128 = np.array([
    0.737783,
    0.485317,
    38.600800,
    np.nan,
    26.600100,
    np.nan,
])

deeppath256 = np.array([
    0.754167,
    0.392600,
    24.911350,
    412.312283,
    17.348833,
    np.nan,
])

adaptive = np.array([
    0.589983,
    0.303700,
    15.296467,
    222.792317,
    10.412750,
    189.614933,
])

# -----------------------------
# Plot
# -----------------------------
x = np.arange(len(datasets))
width = 0.23

fig, ax = plt.subplots(figsize=(9.5, 5.2))

ax.bar(x - width, deeppath128, width, label="deeppath128")
ax.bar(x, deeppath256, width, label="deeppath256")
ax.bar(x + width, adaptive, width, label="Adaptive Scheduling")

ax.set_yscale("log")

ax.set_title("Cross-Dataset Runtime of Active-Path Variants", fontsize=17, pad=10)
ax.set_ylabel("Runtime for 100 queries (min, log scale)", fontsize=14)

ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=0, ha="center", fontsize=12)

ax.tick_params(axis="y", labelsize=12)
ax.legend(fontsize=11, loc="upper left", frameon=True)

ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()
plt.savefig("cross_dataset_runtime_active_path_variants.png", dpi=300, bbox_inches="tight")
plt.savefig("cross_dataset_runtime_active_path_variants.pdf", bbox_inches="tight")
plt.show()