# SCC Preprocessing Utilities

This folder contains three small C++ utilities used to test SCC-based graph compression and query remapping for max-flow experiments.

## Files

### `scc_contract_english_comments.cpp`

Builds a strongly connected component (SCC) contraction of a directed capacity graph.

Input graph format:

```text
n m
u v cap
u v cap
...
```

Vertex IDs are expected to be 1-based. Self-loops, invalid vertices, and non-positive capacities are ignored. The program uses an iterative Kosaraju SCC algorithm, merges capacities between different SCCs, and writes:

- `output_graph_scc.txt`: the contracted SCC graph.
- `output_graph_scc.txt.map`: mapping from original vertex ID to SCC component ID.

Usage:

```bash
g++ -O3 -std=c++17 -o scc_contract scc_contract.cpp
./scc_contract graph.txt graph_scc.txt
```

### `convert_test_to_scc.cpp`

Converts every query endpoint in an original test/query file to its SCC component ID using `graph_scc.txt.map`.

Usage:

```bash
g++ -O3 -std=c++17 -o convert_test_to_scc convert_test_to_scc.cpp
./convert_test_to_scc test.txt graph_scc.txt.map test_scc.txt
```

If an endpoint is out of range or has no valid component ID, the program writes `0 0` for that query.

### `split_queries_by_scc.cpp`

Splits query pairs into two groups:

- `inter`: source and sink are in different SCCs. These can be mapped to component IDs and tested on the SCC-contracted graph.
- `intra`: source and sink are in the same SCC. These should be solved on the original graph with original vertex IDs.

Usage:

```bash
g++ -O3 -std=c++17 -o split_queries_by_scc split_queries_by_scc.cpp
./split_queries_by_scc test.txt graph_scc.txt.map test_inter.txt test_intra.txt
```

The program also writes an index file named `test_inter.txt.index`, which records the original query ID, the query class, and the new line ID in the corresponding output file.

## Suggested Workflow

```bash
# 1. Build SCC-contracted graph and original-to-component mapping.
g++ -O3 -std=c++17 -o scc_contract scc_contract.cpp
./scc_contract graph.txt graph_scc.txt

# 2. Split queries according to whether endpoints are in the same SCC.
g++ -O3 -std=c++17 -o split_queries_by_scc split_queries_by_scc.cpp
./split_queries_by_scc test.txt graph_scc.txt.map test_inter.txt test_intra.txt

# 3. Run inter-component queries on graph_scc.txt with test_inter.txt.
# 4. Run intra-component queries on the original graph.txt with test_intra.txt.
# 5. Merge the two result streams using the generated index file if needed.
```

## Notes

- These utilities do not change the max-flow solver itself. They only generate SCC-based graph/query preprocessing outputs.
- SCC contraction is a static graph transformation. It may reduce graph size for some datasets, but it does not necessarily reduce GPU max-flow runtime because push-relabel operates on a changing residual graph.
- The contracted graph stores only component-level edges in the output file. If the GPU solver builds explicit reverse residual edges internally, the CSR edge count used by the solver may be approximately twice the component-level edge count.
