# -*- coding: UTF-8 -*-
'''
@Project ：COMPARISON
@File    ：dp_compared_on_bit2014.py
@IDE     ：PyCharm 
@Author  ：Winston·H DONG
@Date    ：2026/5/4 22:13 
'''
import matplotlib.pyplot as plt

# x-axis labels
depths = ["64", "128", "256", "512", "1024", "1048576"]

# runtime in minutes for 100 queries
test_runtime = [0.70, 0.68, 0.74, 0.95, 1.55, 260.0]
pros_runtime = [0.72, 0.50, 0.40, 0.42, None, None]

fig, ax = plt.subplots(figsize=(7.5, 4.2))

# plot lines
ax.plot(depths, test_runtime, marker="o", linewidth=1.8, label="Bitcoin2014")
ax.plot(depths, pros_runtime, marker="o", linewidth=1.8, label="PROS")

# log-scale y-axis
ax.set_yscale("log")

# title and labels
ax.set_title("Sparse Active-Path Depth Sensitivity", fontsize=17)
ax.set_xlabel("Maximum path depth", fontsize=14)
ax.set_ylabel("Runtime for 100 queries (min, log scale)", fontsize=14)

# legend and grid
ax.legend(fontsize=12, loc="upper left")
ax.grid(False)

plt.tight_layout()
plt.savefig("sparse_active_path_depth_sensitivity.png", dpi=300, bbox_inches="tight")
plt.show()