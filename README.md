# Region-Aware GPU Push-Relabel Algorithm

This README describes `newest_static_regionboundaryc.cu`, a CUDA/C++ implementation of an exact maximum-flow solver based on the push-relabel algorithm. The implementation is designed for large directed graphs and repeated source-sink queries. It uses a region-ordered CSR layout, query-level pruning, region-aware push-relabel kernels, frontier-based relabeling, region-aware residual repair, and sparse active-path discharge.

## Main Features

- **CSR residual graph representation**
  - Loads a directed input graph from `graph.txt`.
  - Builds forward residual edges and reverse residual edges.
  - Stores the graph in CSR format using `row_ptr`, `col_idx`, `cap`, and `rev` arrays.

- **Capacity-degree based region partitioning**
  - Computes vertex statistics from in/out degrees and in/out capacity sums.
  - Assigns vertices to regions using a score-based load-balancing heuristic.
  - Reorders vertices so that each region occupies a contiguous CSR interval.

- **One-block-per-region push-relabel kernel**
  - Each CUDA block processes one region.
  - Active vertices are discharged for a bounded number of local cycles.
  - Cross-region residual edges remain valid and are processed through the same atomic residual/excess updates.

- **Frontier-based global relabeling**
  - Uses frontier arrays instead of full graph-level BFS scans at every BFS depth.
  - Initializes BFS from the sink and negative-excess vertices.
  - Respects the query-specific reachability mask when available.

- **Local relabeling over active regions**
  - Computes the active height range.
  - Marks regions containing active vertices.
  - Optionally runs local frontier BFS in active regions.

- **Region-aware RemoveInvalidEdges (RM)**
  - Repairs invalid reverse residual edges that violate the current height labels.
  - Runs only on active regions and relevant vertex ranges.
  - Reports the number of repaired edges through `d_fixed_edges`.

- **Sparse active-list and path-push modes**
  - Builds an active list when the active set is small.
  - Uses a path-based push kernel to move excess through short admissible paths.
  - Falls back to local push-relabel steps when path pushing cannot finish discharge.

- **Hot-region extra sweep**
  - Detects regions with concentrated active work.
  - Runs an additional region-local discharge sweep only on marked hot regions.

- **Query-specific pruning and optional boundary-aware core graph**
  - Computes a conservative reachability mask for each `(s,t)` query.
  - Can induce a smaller query-specific subgraph when pruning is effective.
  - Can build a coarse region graph and boundary-aware core candidate.

- **Static and dynamic execution modes**
  - `static`: solve each query from a freshly initialized residual graph.
  - `dynamic`: includes code paths for incremental capacity-update experiments.

## Input Directory Format

The program expects an input directory containing:

```text
<input_dir>/graph.txt
<input_dir>/test.txt
```

### `graph.txt`

The graph file must use 1-based vertex IDs:

```text
n m
u1 v1 cap1
u2 v2 cap2
...
um vm capm
```

Where:

- `n` is the number of vertices.
- `m` is the number of directed input edges.
- `u v cap` is a directed edge from `u` to `v` with capacity `cap`.
- Vertices are expected to be 1-based in the file.
- Internally, the program converts vertices to 0-based IDs.
- Self-loops, invalid vertex IDs, and non-positive capacities are ignored.

### `test.txt`

The query file also uses 1-based vertex IDs:

```text
s1 t1
s2 t2
...
```

Each line is one source-sink query.

## Output

The output directory is created if it does not already exist.

In static mode, the main output file is:

```text
<output_dir>/res_mt.txt
```

Each line contains:

```text
flow time_ms
```

Where:

- `flow` is the computed maximum flow value for the query.
- `time_ms` is the query runtime in milliseconds, including pruning time when enabled.

The program also prints detailed diagnostic logs to `stderr`, including:

- graph size after region reordering;
- query IDs and remapped source/sink IDs;
- pruning time and pruned graph size;
- initialization kernel time;
- BFS time;
- push-relabel time;
- RM time;
- active vertex counts;
- high-label active counts;
- fixed edge counts;
- current `GR_FREQ0` value.

## Build

Compile with `nvcc`:

```bash
nvcc -O3 -o newest_static_regionboundaryc newest_static_regionboundaryc.cu
```

For debugging, you may use:

```bash
nvcc -O0 -g -G -o newest_static_regionboundaryc_debug newest_static_regionboundaryc.cu
```

## Run

### Static mode

```bash
./newest_static_regionboundaryc <input_dir> <output_dir> static
```

If the third argument is omitted, static mode is used by default:

```bash
./newest_static_regionboundaryc <input_dir> <output_dir>
```

Example:

```bash
./newest_static_regionboundaryc data/test out/test static
```

### Dynamic mode

```bash
./newest_static_regionboundaryc <input_dir> <output_dir> dynamic
```

Dynamic mode is intended for experiments where residual state is reused after capacity updates. Use static mode for ordinary repeated max-flow queries.

## Important Implementation Notes

### GPU device selection

The program currently calls:

```cpp
cudaSetDevice(1);
```

If your machine has only one GPU, or if you want to run on GPU 0, change this line to:

```cpp
cudaSetDevice(0);
```

### Region count

The region count is currently set in `main` as:

```cpp
int R = 128;
```

You can change this value to tune region granularity. Larger `R` creates more regions, but also changes region sizes and scheduling behavior.

### Flow type

The implementation uses:

```cpp
typedef long long flow_t;
```

This supports large integral capacities, but the total flow and intermediate excess values must still fit into signed 64-bit integers.

### Reverse edges

For each input edge, the program creates a forward residual edge and a reverse residual edge. The reverse-edge index is stored in `rev`. This is required for constant-time residual updates during push operations.

### Query pruning

The static path enables query pruning by default:

```cpp
const bool ENABLE_QUERY_PRUNE = true;
const bool ENABLE_REGION_BOUNDARY_ON_SUBGRAPH = true;
```

The pruning pipeline may build a smaller induced graph if the retained vertex set is sufficiently smaller than the full graph. If pruning or boundary-core extraction does not shrink enough, the solver falls back to a less aggressive graph representation.

### Adaptive scheduling

The solver adapts several runtime choices:

- global relabel frequency through `GR_FREQ0`;
- RM frequency through `RM_FREQ_dyn`;
- push-relabel cycle budget `K`;
- sparse active-path mode when active vertices are few and stable;
- extra hot-region sweep when active work is concentrated in selected regions.

## Typical Workflow

1. Prepare `graph.txt` and `test.txt` in an input directory.
2. Compile the CUDA program with `nvcc`.
3. Run the program in static mode.
4. Inspect `res_mt.txt` for per-query flow values and runtimes.
5. Inspect `stderr` logs for kernel timings and debugging statistics.

Example:

```bash
mkdir -p out/test
nvcc -O3 -o newest_static_regionboundaryc newest_static_regionboundaryc.cu
./newest_static_regionboundaryc data/test out/test static 2> out/test/run.log
cat out/test/res_mt.txt
```

## Troubleshooting

### CUDA invalid device error

Change `cudaSetDevice(1)` to `cudaSetDevice(0)` if only one GPU is available.

### Very large or incorrect flow values

Check whether capacities and total flow fit in signed 64-bit integers. Also verify that all atomic updates on `flow_t` are consistent and that the graph input does not contain malformed capacities.

### Slow BFS

Global relabeling can dominate runtime on large graphs. Inspect log fields such as `bfs=... ms` and `GR_FREQ0=...` to determine whether global relabeling is being triggered too frequently.

### Slow RM

Inspect `RM_invalid=... ms` and `fixed_edges=...`. If `fixed_edges` is often zero, RM may be running too frequently. If `fixed_edges` is large, RM is actively repairing residual inconsistencies.

### No speedup from pruning

Some graphs do not shrink much under query reachability pruning. In this case, the solver may fall back to the full graph or the query-pruned graph rather than a boundary-core graph.

## File Summary

- `CSRGraph`: host-side CSR graph structure.
- `DynamicState`: per-query GPU residual and auxiliary buffers.
- `build_graph`: reads `graph.txt` and builds CSR with reverse residual edges.
- `build_capacity_degree_partition_from_stats`: computes score-based region partitioning.
- `partition_and_reorder`: creates region-ordered CSR.
- `init_global_graph_on_device`: uploads graph and region metadata to GPU.
- `staticMaxFlow_kernel_1`: initializes residual capacities and excesses.
- `staticMaxFlow_kernel_7`: performs source preflow.
- `init_height_level_frontier`: initializes frontier-based relabeling.
- `bfs_expand_frontier`: expands BFS frontier.
- `staticMaxFlow_kernel_14_region`: block-per-region push-relabel discharge.
- `staticMaxFlow_kernel_17_region`: region-aware residual repair.
- `build_active_list_and_count`: builds active list and active counters.
- `staticMaxFlow_kernel_14_active_list_with_path_push`: sparse active-path discharge.
- `pr_main_loop_static`: main iterative solver loop.
- `run_gpu_static_with_state`: initializes a query and runs the static solver.
- `main`: reads input, partitions/reorders graph, processes queries, and writes results.
