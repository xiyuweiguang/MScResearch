#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <string>
#include <queue>
#include <climits>
#include <cmath>
#include <numeric>

using namespace std;

// -------------------- Basic types and graph structure --------------------

typedef long long flow_t;

struct CSRGraph {
    int n;
    int m;
    int*    row_ptr;
    int*    col_idx;
    flow_t* cap;
    int*    rev;
};

// -------------------- Global graph structure shared by all (s,t) queries --------------------

static bool   g_graph_inited   = false;
static int   *g_d_meta         = nullptr;
static int   *g_d_data         = nullptr;
static flow_t*g_d_weight       = nullptr;
static int   *g_d_parallel     = nullptr;
static int    g_V              = 0;
static int    g_E              = 0;

static int   *g_d_region_start = nullptr;
static int   *g_d_region_end   = nullptr;
static int   *g_d_region_id    = nullptr;
static int    g_R              = 0;


// -------------------- Dynamic residual state, one copy per (s,t) query --------------------

struct DynamicState {
    int V = 0, E = 0, R = 0;
    bool inited = false;

    // Residual network and push-relabel state
    flow_t *d_residual_capacity     = nullptr;
    flow_t *d_rev_residual_capacity = nullptr;
    flow_t *d_excess                = nullptr;
    int    *d_height                = nullptr;
    int    *d_level                 = nullptr;

    // BFS, active-list, RM, and path-push related buffers
    int *d_frontier           = nullptr;
    int *d_next_frontier      = nullptr;
    int *d_frontier_size      = nullptr;
    int *d_next_frontier_size = nullptr;
    int *d_active_count       = nullptr;
    int *d_active_high        = nullptr;
    int *d_fixed_edges        = nullptr;
    int *d_hmin_active        = nullptr;
    int *d_hmax_active        = nullptr;
    int *d_region_active      = nullptr;
    long long *d_pos_excess_sum = nullptr;

    // active-list
    int *d_active_list = nullptr;
    int *d_active_size = nullptr;
    int *d_is_reachable_to_t = nullptr;  // Reachability marker from t
    int *d_region_hot = nullptr;   // Added: hot-region marker

    int *d_reach_from_s = nullptr;
    int *d_reach_to_t   = nullptr;
    int *d_tmp_frontier = nullptr;
    int *d_tmp_next     = nullptr;
    int *d_tmp_fsize    = nullptr;
    int *d_tmp_nsize    = nullptr;
};

// Release GPU memory owned by one DynamicState
void free_state_for_st(DynamicState& st) {
    if (!st.inited) return;

    cudaFree(st.d_residual_capacity);
    cudaFree(st.d_rev_residual_capacity);
    cudaFree(st.d_excess);
    cudaFree(st.d_height);
    cudaFree(st.d_level);

    cudaFree(st.d_frontier);
    cudaFree(st.d_next_frontier);
    cudaFree(st.d_frontier_size);
    cudaFree(st.d_next_frontier_size);
    cudaFree(st.d_active_count);
    cudaFree(st.d_active_high);
    cudaFree(st.d_fixed_edges);
    cudaFree(st.d_hmin_active);
    cudaFree(st.d_hmax_active);
    cudaFree(st.d_region_active);
    cudaFree(st.d_pos_excess_sum);
    cudaFree(st.d_active_list);
    cudaFree(st.d_active_size);
    cudaFree(st.d_is_reachable_to_t);
    cudaFree(st.d_region_hot);   // Added

    st.inited = false;
    st.V = st.E = st.R = 0;
}


// ============ Update structure, only capacity increases are implemented ============

struct Update {
    char type;        // 'a' = add/modify; only capacity increases are implemented
    int  source;      // Vertex ID after new_id remapping
    int  destination; // Vertex ID after new_id remapping
    long long weight; // New capacity; must be >= original capacity
};

// -------------------- 64-bit atomic operations --------------------

__device__ __forceinline__ long long atomicAddLongLong(long long* address, long long val) {
    unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *addr_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        long long new_val = static_cast<long long>(assumed) + val;
        unsigned long long new_val_ull = static_cast<unsigned long long>(new_val);
        old = atomicCAS(addr_ull, assumed, new_val_ull);
    } while (assumed != old);
    return static_cast<long long>(old);
}

__device__ __forceinline__ void atomicAddFlow(flow_t* addr, flow_t val) {
    atomicAddLongLong(reinterpret_cast<long long*>(addr), (long long)val);
}

__device__ __forceinline__ void atomicSubFlow(flow_t* addr, flow_t val) {
    atomicAddLongLong(reinterpret_cast<long long*>(addr), -(long long)val);
}

// -------------------- Kernels --------------------

// Initialize residual capacities and set excess to 0
__global__ void staticMaxFlow_kernel_1(
    int V,
    int*    d_meta,
    int*    d_data,
    flow_t* d_weight,
    flow_t* d_residual_capacity,
    flow_t* d_rev_residual_capacity,
    int*    d_parallel_edge,
    flow_t* d_excess
) {
    unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= (unsigned)V) return;
    for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) {
        int e1 = edge;
        d_residual_capacity[e1]     = d_weight[e1];
        int p                       = d_parallel_edge[e1];
        d_rev_residual_capacity[e1] = d_weight[p];
    }
    d_excess[v] = 0;
}

// Source preflow
__global__ void staticMaxFlow_kernel_7(
    int source0,
    int V,
    int E,
    int*    d_meta,
    int*    d_data,
    flow_t* d_residual_capacity,
    flow_t* d_excess,
    int*    d_parallel_edge,
    flow_t* d_rev_residual_capacity
) {
    unsigned v = source0;
    if (v >= (unsigned)V) return;
    for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) {
        int    vv           = d_data[edge];
        int    forward_edge = edge;
        flow_t d            = d_residual_capacity[forward_edge];
        if (d > 0) {
            d_excess[vv]  += d;
            d_excess[v]   -= d;
            d_residual_capacity[forward_edge]     -= d;
            d_rev_residual_capacity[forward_edge] += d;
            int p = d_parallel_edge[forward_edge];
            d_residual_capacity[p]    += d;
            d_rev_residual_capacity[p] -= d;
        }
    }
}

// Global BFS seed initialization for the whole graph with pruning constraints
__global__ void init_height_level_frontier(
    int     source0,
    int     sink0,
    int     V,
    const flow_t* d_excess,
    int*    d_level,
    int*    d_height,
    int*    d_frontier,
    int*    d_frontier_size,
    const int* __restrict__ d_is_reachable_to_t   // Added: pruning marker
) {
    unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= (unsigned)V) return;

    // Vertices unreachable from t, except t itself, are fixed at height V and ignored by later BFS
    if ((int)v != sink0 && d_is_reachable_to_t[v] == 0) {
        d_height[v] = V;
        d_level[v]  = -1;
        return;
    }

    d_height[v] = V;
    d_level[v]  = -1;
    if ((int)v != source0) {
        if (d_excess[v] < 0 || (int)v == sink0) {
            d_level[v] = 0;
            int pos = atomicAdd(d_frontier_size, 1);
            d_frontier[pos] = v;
        }
    }
}

// Local BFS seed initialization with height band, active-region, and pruning constraints
__global__ void init_local_frontier_in_band(
    int     source0,
    int     sink0,
    int     V,
    const flow_t* d_excess,
    int*    d_level,
    int*    d_height,
    int*    d_frontier,
    int*    d_frontier_size,
    int     h_min_band,
    int     h_max_band,
    const int* __restrict__ d_region_start,
    const int* __restrict__ d_region_end,
    const int* __restrict__ d_region_active,
    const int* __restrict__ d_region_id,
    const int* __restrict__ d_is_reachable_to_t   // Added
) {
    unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= (unsigned)V) return;

    // Pruning: vertices unreachable from t, except t, keep height V and do not join local BFS
    if ((int)v != sink0 && d_is_reachable_to_t[v] == 0) {
        d_height[v] = V;
        d_level[v]  = -1;
        return;
    }

    int r = d_region_id[v];   // O(1) direct region lookup
    if (r < 0) return;
    if (d_region_active[r] == 0) return;

    int hv = d_height[v];
    if (hv < h_min_band || hv > h_max_band) return;

    d_level[v] = -1;

    if ((int)v != source0) {
        if (d_excess[v] < 0 || (int)v == sink0) {
            d_level[v] = 0;
            int pos = atomicAdd(d_frontier_size, 1);
            d_frontier[pos] = v;
        }
    }
}


// Count active vertices with excess > 0 in each region, one block per region
__global__ void count_region_active_kernel(
    int R,
    const int* __restrict__ d_region_start,
    const int* __restrict__ d_region_end,
    const flow_t* __restrict__ d_excess,
    int* __restrict__ d_region_active_cnt
) {
    int r = blockIdx.x;
    if (r >= R) return;

    int start = d_region_start[r];
    int end   = d_region_end[r];

    int local_cnt = 0;
    for (int v = start + threadIdx.x; v < end; v += blockDim.x) {
        if (d_excess[v] > 0) {
            local_cnt++;
        }
    }

    // In-block reduction
    __shared__ int smem[256];   // Assume blockDim.x <= 256
    int tid = threadIdx.x;
    smem[tid] = local_cnt;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            smem[tid] += smem[tid + offset];
        __syncthreads();
    }

    if (tid == 0) {
        d_region_active_cnt[r] = smem[0];
    }
}


// One top-down BFS level
__global__ void bfs_expand_frontier(
    int V,
    const int*    __restrict__ d_meta,
    const int*    __restrict__ d_data,
    const flow_t* __restrict__ d_rev_residual_capacity,
    int*          __restrict__ d_level,
    int*          __restrict__ d_height,
    const int*    __restrict__ frontier,
    int frontier_size,
    int cur_level,
    int*          __restrict__ next_frontier,
    int*          __restrict__ d_next_frontier_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;
    int v = frontier[idx];
    d_height[v] = cur_level;
    for (int e = d_meta[v]; e < d_meta[v+1]; ++e) {
        int w = d_data[e];
        if (d_rev_residual_capacity[e] > 0 && d_level[w] == -1) {
            if (atomicCAS(&d_level[w], -1, cur_level + 1) == -1) {
                int pos = atomicAdd(d_next_frontier_size, 1);
                next_frontier[pos] = w;
            }
        }
    }
}


__global__ void bfs_forward_residual(
    int V, int source,
    const int* __restrict__ d_meta,
    const int* __restrict__ d_data,
    const flow_t* __restrict__ d_residual,
    int* __restrict__ d_reach,
    const int* __restrict__ frontier,
    int frontier_size,
    int* __restrict__ next_frontier,
    int* __restrict__ d_next_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;
    int v = frontier[idx];

    for (int e = d_meta[v]; e < d_meta[v+1]; ++e) {
        int u = d_data[e];
        if (d_residual[e] > 0 && atomicCAS(&d_reach[u], 0, 1) == 0) {
            int pos = atomicAdd(d_next_size, 1);
            next_frontier[pos] = u;
        }
    }
}


__global__ void bfs_backward_residual(
    int V, int sink,
    const int* __restrict__ d_meta,
    const int* __restrict__ d_data,
    const flow_t* __restrict__ d_rev_residual,
    const int* __restrict__ d_parallel,
    int* __restrict__ d_reach,
    const int* __restrict__ frontier,
    int frontier_size,
    int* __restrict__ next_frontier,
    int* __restrict__ d_next_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;
    int v = frontier[idx];

    // Traverse edges u->v for which v is the destination:
    // For each outgoing edge e of v, the reverse edge p = d_parallel[e] is u->v
    for (int e = d_meta[v]; e < d_meta[v+1]; ++e) {
        int p = d_parallel[e];   // p: u->v
        int u = d_data[p];       // u is the source endpoint of reverse edge p

        if (d_rev_residual[e] > 0 && atomicCAS(&d_reach[u], 0, 1) == 0) {
            int pos = atomicAdd(d_next_size, 1);
            next_frontier[pos] = u;
        }
    }
}


__global__ void refine_mask_and_height(
    int V,
    int source, int sink,
    const int* __restrict__ d_reach_from_s,
    const int* __restrict__ d_reach_to_t,
    int*       __restrict__ d_is_reach,   // Reuse this buffer
    int*       __restrict__ d_height
    // flow_t*    __restrict__ d_excess     // Optional: whether to clear it
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;

    int keep = (d_reach_from_s[v] && d_reach_to_t[v]) || (v == source) || (v == sink);
    d_is_reach[v] = keep;

    if (!keep && v != source && v != sink) {
        d_height[v] = V;
        // Optionally clamp the excess of these vertices:
        // d_excess[v] = 0;  // Caution: this breaks flow conservation; only use as a heuristic after validation
    }
}




// Region-aware push-relabel
__global__ void staticMaxFlow_kernel_14_region(
    int     source0,
    int     sink0,
    int     kernel_cycles0,
    int     V,
    int     E,
    const int* __restrict__ d_region_start,
    const int* __restrict__ d_region_end,
    int*    d_meta,
    int*    d_data,
    flow_t* d_excess,
    int*    d_parallel_edge,
    flow_t* d_rev_residual_capacity,
    flow_t* d_residual_capacity,
    int*    d_height,
    int*    d_active_count,
    int*    d_active_high,
    int     H_cut,
    const int* __restrict__ d_region_hot  // Added: may be nullptr or a 0/1 mask
) {
    int r = blockIdx.x;  // One block per region
    if (d_region_hot != nullptr && d_region_hot[r] == 0) {
        // Skip this region in this round
        return;
    }

    int start = d_region_start[r];
    int end   = d_region_end[r];

    float num_nodes = (float)V;

    for (int v = start + threadIdx.x; v < end; v += blockDim.x) {
        int hv = d_height[v];

        if (d_excess[v] > 0 && v != source0 && v != sink0 && hv < num_nodes) {
            if (d_active_count != nullptr) {
                atomicAdd(d_active_count, 1);
                if (d_active_high != nullptr && hv > H_cut) {
                    atomicAdd(d_active_high, 1);
                }
            }

            int cycle = kernel_cycles0;
            do {
                if (d_excess[v] > 0 && d_height[v] < V) {
                    flow_t ex1 = d_excess[v];
                    int    hh  = INT_MAX;
                    int    v_0 = -1;
                    int    forward_edge = -1;

                    for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) {
                        int vv = d_data[edge];
                        int e  = edge;
                        int h1 = d_height[vv];
                        if (h1 < hh && d_residual_capacity[e] > 0) {
                            v_0 = vv;
                            hh  = h1;
                            forward_edge = e;
                        }
                    }

                    if (d_height[v] > hh && v_0 != -1) {
                        flow_t fec = d_residual_capacity[forward_edge];
                        int    p   = d_parallel_edge[forward_edge];
                        flow_t d   = (ex1 < fec ? ex1 : fec);
                        atomicSubFlow(&d_excess[v], d);
                        atomicAddFlow(&d_excess[v_0], d);
                        atomicSubFlow(&d_residual_capacity[forward_edge], d);
                        atomicAddFlow(&d_rev_residual_capacity[forward_edge], d);
                        atomicAddFlow(&d_residual_capacity[p], d);
                        atomicSubFlow(&d_rev_residual_capacity[p], d);
                    } else {
                        if (v_0 != -1) {
                            d_height[v] = hh + 1;
                        }
                    }
                } else {
                    break;
                }
                cycle--;
            } while (cycle > 0);
        }
    }
}


// Compute active-vertex height range (min/max)
__global__ void compute_active_height_range(
    int     V,
    int     source0,
    int     sink0,
    const flow_t* d_excess,
    const int*    d_height,
    int*          d_hmin,
    int*          d_hmax
) {
    unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= (unsigned)V) return;
    if ((int)v == source0 || (int)v == sink0) return;
    if (d_excess[v] > 0) {
        int h = d_height[v];
        atomicMin(d_hmin, h);
        atomicMax(d_hmax, h);
    }
}

// Determine whether each region has active vertices; one block per region
__global__ void compute_region_active(
    int     R,
    const int* __restrict__ d_region_start,
    const int* __restrict__ d_region_end,
    const flow_t* __restrict__ d_excess,
    int*          __restrict__ d_region_active
) {
    int r = blockIdx.x;
    if (r >= R) return;
    int start = d_region_start[r];
    int end   = d_region_end[r];
    for (int v = start + threadIdx.x; v < end; v += blockDim.x) {
        if (d_excess[v] > 0) {
            atomicExch(&d_region_active[r], 1);
            break;
        }
    }
}

// Sum positive excess over all non-terminal vertices using a 64-bit accumulator
__global__ void sum_positive_excess_kernel(
    int V,
    int source0,
    int sink0,
    const flow_t* __restrict__ d_excess,
    long long* __restrict__ d_sum  // device-side accumulator
) {
    unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= (unsigned)V) return;
    if ((int)v == source0 || (int)v == sink0) return;
    flow_t ex = d_excess[v];
    if (ex > 0) {
        atomicAddLongLong(d_sum, ex);
    }
}



// Build the active list only for hot regions
__global__ void build_hot_active_list(
    int V,
    int source0,
    int sink0,
    const flow_t* __restrict__ d_excess,
    const int*   __restrict__ d_height,
    const int*   __restrict__ d_region_id,
    const int*   __restrict__ d_region_hot,
    int*         __restrict__ d_active_list,
    int*         __restrict__ d_active_size
) {
    unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= (unsigned)V) return;
    if ((int)v == source0 || (int)v == sink0) return;

    int r = d_region_id[v];
    if (r < 0 || d_region_hot[r] == 0) return;   // Keep only vertices in hot regions

    flow_t ex = d_excess[v];
    if (ex > 0 && d_height[v] < V) {
        int pos = atomicAdd(d_active_size, 1);
        d_active_list[pos] = v;
    }
}


// Region-aware RM: one block per region, process only regions with region_active[r] == 1
__global__ void staticMaxFlow_kernel_17_region(
    int     R,
    int     source0,
    int     sink0,
    int     V,
    int     E,
    const int* __restrict__ d_region_start,
    const int* __restrict__ d_region_end,
    const int* __restrict__ d_region_active,
    int*    d_meta,
    int*    d_data,
    flow_t* d_rev_residual_capacity,
    int*    d_parallel_edge,
    flow_t* d_residual_capacity,
    flow_t* d_excess,
    int*    d_height,
    int*    d_fixed_edges,
    int     h_min_rm,
    int     h_max_rm
) {
    int r = blockIdx.x;
    if (r >= R) return;
    if (d_region_active[r] == 0) return;

    int start = d_region_start[r];
    int end   = d_region_end[r];

    for (int v = start + threadIdx.x; v < end; v += blockDim.x) {
        int h = d_height[v];
        // Clean only high-level vertices inside the RM band
        if (h < h_min_rm || h > h_max_rm) continue;
        if (v == source0 || v == sink0)   continue;

        for (int edge = d_meta[v]; edge < d_meta[v+1]; ++edge) {
            int vv = d_data[edge];
            int e  = edge;

            if (d_height[vv] > h + 1 && d_excess[vv] <= 0) {
                flow_t d = d_rev_residual_capacity[e];
                d_rev_residual_capacity[e] -= d;
                d_residual_capacity[e]     += d;

                int p = d_parallel_edge[e];
                d_rev_residual_capacity[p] += d;
                d_residual_capacity[p]     -= d;

                atomicSubFlow(&d_excess[vv], d);
                atomicAddFlow(&d_excess[v],  d);
                atomicAdd(d_fixed_edges, 1);
            }
        }
    }
}




// -------------------- Count active vertices and build the active list in one scan --------------------

__global__ void build_active_list_and_count(
    int V,
    int source0,
    int sink0,
    const flow_t* __restrict__ d_excess,
    const int* __restrict__ d_height,
    int* __restrict__ d_active_list,
    int* __restrict__ d_active_size,
    int* __restrict__ d_active_count,
    int* __restrict__ d_active_high,
    int H_cut
) {
    unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= (unsigned)V) return;
    if ((int)v == source0 || (int)v == sink0) return;
    
    flow_t ex = d_excess[v];
    if (ex > 0) {
        int h = d_height[v];
        if (h < V) {
            // Count active vertices
            atomicAdd(d_active_count, 1);
            if (h > H_cut) {
                atomicAdd(d_active_high, 1);
            }

            // Build active list
            int pos = atomicAdd(d_active_size, 1);
            d_active_list[pos] = v;
        }
    }
}



// Small-active phase: run push-relabel only on the active list,
// each thread tries to fully discharge its assigned vertex before exiting.
__global__ void staticMaxFlow_kernel_14_active_list(
    int     active_size,
    const int* __restrict__ d_active_list,
    int     source0,
    int     sink0,
    int     V,
    int*    d_meta,
    int*    d_data,
    flow_t* d_excess,
    int*    d_parallel_edge,
    flow_t* d_rev_residual_capacity,
    flow_t* d_residual_capacity,
    int*    d_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_size) return;

    int v = d_active_list[idx];
    if (v == source0 || v == sink0) return;

    // Limit the maximum loop count to avoid pathological infinite loops
    const int MAX_CYCLES = 1000000;
    int cycle = 0;

    while (d_excess[v] > 0 && d_height[v] < V && cycle < MAX_CYCLES) {
        flow_t ex1 = d_excess[v];
        if (ex1 <= 0) break;

        int    hh  = INT_MAX;
        int    v_0 = -1;
        int    forward_edge = -1;

        // Find a pushable neighbor with the minimum height
        for (int edge = d_meta[v]; edge < d_meta[v+1]; ++edge) {
            int vv = d_data[edge];
            int h1 = d_height[vv];
            if (d_residual_capacity[edge] > 0 && h1 < hh) {
                hh  = h1;
                v_0 = vv;
                forward_edge = edge;
            }
        }

        if (v_0 != -1 && d_height[v] > hh) {
            // push
            flow_t fec = d_residual_capacity[forward_edge];
            int    p   = d_parallel_edge[forward_edge];
            flow_t d   = (ex1 < fec ? ex1 : fec);

            atomicSubFlow(&d_excess[v],    d);
            atomicAddFlow(&d_excess[v_0],  d);
            atomicSubFlow(&d_residual_capacity[forward_edge],     d);
            atomicAddFlow(&d_rev_residual_capacity[forward_edge], d);
            atomicAddFlow(&d_residual_capacity[p],                d);
            atomicSubFlow(&d_rev_residual_capacity[p],            d);
        } else {
            // relabel
            if (hh < INT_MAX) {
                d_height[v] = hh + 1;
            } else {
                // No outgoing edge has residual capacity; no further push is possible
                break;
            }
        }

        ++cycle;
    }
}

// -------------------- Added: Active-list push-relabel with path pushing --------------------

// Try to push flow from vertex v along the height gradient; return whether excess was discharged
__device__ bool try_push_along_path(
    int v,
    int source,
    int sink,
    int V,
    const int* __restrict__ d_meta,
    const int* __restrict__ d_data,
    flow_t* d_excess,
    int* d_parallel_edge,
    flow_t* d_rev_residual_capacity,
    flow_t* d_residual_capacity,
    int* d_height,
    int max_path_length
) {
    int current = v;
    int path_len = 0;
    
    while (path_len < max_path_length) {
        // Check whether the current vertex still has excess
        flow_t ex = d_excess[current];
        if (ex <= 0) return true; // Successfully discharged all excess
        
        if (current == sink) return true; // Reached the sink
        
        int h_current = d_height[current];
        if (h_current >= V) return false; // Height is too high; cannot push flow
        
        // Find the best push target: minimum height satisfying h[current] = h[next] + 1
        int best_next = -1;
        int best_edge = -1;
        int min_height = INT_MAX;
        flow_t max_flow_can_push = 0;
        
        for (int edge = d_meta[current]; edge < d_meta[current + 1]; ++edge) {
            int next = d_data[edge];
            flow_t cap = d_residual_capacity[edge];
            
            if (cap > 0) {
                int h_next = d_height[next];
                
                // Prefer a neighbor whose height is exactly h_current - 1
                if (h_current == h_next + 1 && h_next < min_height) {
                    min_height = h_next;
                    best_next = next;
                    best_edge = edge;
                    max_flow_can_push = cap;
                }
            }
        }
        
        // If a suitable push target is found
        if (best_next != -1 && best_edge != -1) {
            flow_t delta = min(ex, max_flow_can_push);
            
            if (delta > 0) {
                // Perform the push
                atomicSubFlow(&d_excess[current], delta);
                atomicAddFlow(&d_excess[best_next], delta);
                atomicSubFlow(&d_residual_capacity[best_edge], delta);
                atomicAddFlow(&d_rev_residual_capacity[best_edge], delta);
                
                int rev_edge = d_parallel_edge[best_edge];
                atomicAddFlow(&d_residual_capacity[rev_edge], delta);
                atomicSubFlow(&d_rev_residual_capacity[rev_edge], delta);
                
                // Continue pushing along this path
                current = best_next;
                path_len++;
                continue;
            }
        }
        
        // Cannot push; relabel is needed
        // Find the minimum height among all residual neighbors
        int new_height = INT_MAX;
        for (int edge = d_meta[current]; edge < d_meta[current + 1]; ++edge) {
            int next = d_data[edge];
            flow_t cap = d_residual_capacity[edge];
            
            if (cap > 0) {
                int h_next = d_height[next];
                new_height = min(new_height, h_next);
            }
        }
        
        if (new_height != INT_MAX && new_height < V - 1) {
            d_height[current] = new_height + 1;
            // Retry after relabel without increasing path_len, giving it one more chance
        } else {
            // Cannot relabel further or has reached the maximum height
            d_height[current] = V;
            return false;
        }
    }
    
    // Reached the maximum path-length limit
    return d_excess[v] <= 0;
}

// Active-list push-relabel kernel with path pushing
__global__ void staticMaxFlow_kernel_14_active_list_with_path_push(
    int     active_size,
    const int* __restrict__ d_active_list,
    int     source,
    int     sink,
    int     V,
    int*    d_meta,
    int*    d_data,
    flow_t* d_excess,
    int*    d_parallel_edge,
    flow_t* d_rev_residual_capacity,
    flow_t* d_residual_capacity,
    int*    d_height,
    int     max_path_length  // Maximum path-push length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_size) return;

    int v = d_active_list[idx];
    if (v == source || v == sink) return;

    // Limit the maximum number of attempts to avoid pathological infinite loops
    const int MAX_ATTEMPTS = 4;
    int attempts = 0;

    while (d_excess[v] > 0 && d_height[v] < V && attempts < MAX_ATTEMPTS) {
        // Try path-based pushing
        bool success = try_push_along_path(
            v, source, sink, V,
            d_meta, d_data,
            d_excess, d_parallel_edge,
            d_rev_residual_capacity,
            d_residual_capacity,
            d_height,
            max_path_length
        );
        
        if (!success) {
            // Path push failed; try a conventional single-step push-relabel fallback
            flow_t ex1 = d_excess[v];
            if (ex1 <= 0) break;
            
            int hh = INT_MAX;
            int v_0 = -1;
            int forward_edge = -1;

            // Find a pushable neighbor with the minimum height
            for (int edge = d_meta[v]; edge < d_meta[v + 1]; ++edge) {
                int vv = d_data[edge];
                int h1 = d_height[vv];
                if (d_residual_capacity[edge] > 0 && h1 < hh) {
                    hh = h1;
                    v_0 = vv;
                    forward_edge = edge;
                }
            }

            if (v_0 != -1 && d_height[v] > hh) {
                // push
                flow_t fec = d_residual_capacity[forward_edge];
                int p = d_parallel_edge[forward_edge];
                flow_t d = (ex1 < fec ? ex1 : fec);

                atomicSubFlow(&d_excess[v], d);
                atomicAddFlow(&d_excess[v_0], d);
                atomicSubFlow(&d_residual_capacity[forward_edge], d);
                atomicAddFlow(&d_rev_residual_capacity[forward_edge], d);
                atomicAddFlow(&d_residual_capacity[p], d);
                atomicSubFlow(&d_rev_residual_capacity[p], d);
            } else {
                // relabel
                if (hh < INT_MAX) {
                    d_height[v] = hh + 1;
                } else {
                    d_height[v] = V;
                    break;
                }
            }
        }
        
        ++attempts;
    }
}

// ============ Added: OnAdd_kernel, only capacity increases are supported ============

__global__ void OnAdd_kernel(
    const Update* __restrict__ d_updateBatch,
    int batch_size,
    const int* __restrict__ d_meta,
    const int* __restrict__ d_data,
    flow_t* __restrict__ d_weight,              // Original graph capacity
    flow_t* __restrict__ d_residual_capacity,   // Forward residual capacity
    const int* __restrict__ d_parallel
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    Update u = d_updateBatch[tid];
    if (u.type != 'a') return;

    int src  = u.source;
    int dest = u.destination;
    if (src == dest) return;

    flow_t new_cap = (flow_t)u.weight;

    // Find the edge (src->dest)
    int e_forward = -1;
    for (int e = d_meta[src]; e < d_meta[src + 1]; ++e) {
        if (d_data[e] == dest) {
            e_forward = e;
            break;
        }
    }
    if (e_forward < 0) return;  // Inserting new edges is not currently supported

    flow_t old_cap = d_weight[e_forward];
    if (new_cap <= old_cap) {
        // Only capacity increases are implemented; decreases are not handled yet
        return;
    }

    flow_t delta = new_cap - old_cap;
    d_weight[e_forward] = new_cap;
    // Increase forward residual by delta without changing existing flow or reverse residual
    atomicAddFlow(&d_residual_capacity[e_forward], delta);
}


// Check reverse connectivity from t in the initial residual network
__global__ void prune_unreachable_nodes_bfs(
    int V,
    const int* __restrict__ d_meta,
    const int* __restrict__ d_data,
    const flow_t* __restrict__ d_weight, // Initial capacity, or current residual capacity
    const int* __restrict__ d_rev,       // Reverse-edge index
    int* __restrict__ d_reachable,
    const int* __restrict__ d_frontier,
    int frontier_size,
    int* __restrict__ d_next_frontier,
    int* __restrict__ d_next_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;

    int v = d_frontier[idx];

    // Traverse edges u->v for which v is the destination:
    // For each outgoing edge e of v, the reverse edge p = d_rev[e] is u->v
    for (int e = d_meta[v]; e < d_meta[v+1]; ++e) {
        int p = d_rev[e];       // p: u->v
        int u = d_data[p];      // u is the source endpoint of reverse edge p

        // If u->v has positive capacity and u is unmarked, mark and enqueue u
        if (d_weight[p] > 0 && atomicCAS(&d_reachable[u], 0, 1) == 0) {
            int pos = atomicAdd(d_next_size, 1);
            d_next_frontier[pos] = u;
        }
    }
}

// -------------------- Host side: build CSR, partition, and region_id --------------------
// -------------------- Host side: build CSR --------------------

struct EdgeTmp {
    int    to;
    flow_t cap;
    int    rev;
};

void build_graph(const char* filename, CSRGraph& g, long long& build_time_ms,
                 vector<long long>& sum_cap_out,
                 vector<long long>& sum_cap_in,
                 vector<int>& deg_out,
                 vector<int>& deg_in) {
    auto b_start = chrono::high_resolution_clock::now();

    FILE* f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }

    int n, m;
    if (fscanf(f, "%d %d", &n, &m) != 2) {
        fprintf(stderr, "Invalid graph header in %s\n", filename);
        exit(1);
    }

    vector<vector<EdgeTmp>> adj(n);

    // Initialize statistic arrays
    sum_cap_out.assign(n, 0);
    sum_cap_in.assign(n, 0);
    deg_out.assign(n, 0);
    deg_in.assign(n, 0);

    for (int i = 0; i < m; ++i) {
        int u, v;
        long long c_ll;
        if (fscanf(f, "%d %d %lld", &u, &v, &c_ll) != 3) break;
        --u; --v;
        if (u < 0 || u >= n || v < 0 || v >= n) continue;
        if (u == v) continue;
        if (c_ll <= 0) continue;
        flow_t c = static_cast<flow_t>(c_ll);

        // Forward edge u->v and reverse edge v->u
        EdgeTmp a{v, c, (int)adj[v].size()};
        EdgeTmp b{u, 0, (int)adj[u].size()};
        adj[u].push_back(a);
        adj[v].push_back(b);

        // In/out degree and capacity statistics; only original forward edges matter because reverse capacity is 0
        sum_cap_out[u] += c;
        sum_cap_in[v]  += c;
        deg_out[u]++;           // Out-degree in the original graph
        deg_in[v]++;            // In-degree in the original graph
    }
    fclose(f);

    // Build CSR graph g
    g.n = n;
    g.m = 0;
    for (int u = 0; u < n; ++u) g.m += (int)adj[u].size();

    g.row_ptr = new int[n + 1];
    g.col_idx = new int[g.m];
    g.cap     = new flow_t[g.m];
    g.rev     = new int[g.m];

    int ptr = 0;
    for (int u = 0; u < n; ++u) {
        g.row_ptr[u] = ptr;
        for (auto &e : adj[u]) {
            g.col_idx[ptr] = e.to;
            g.cap[ptr]     = e.cap;
            ptr++;
        }
    }
    g.row_ptr[n] = ptr;

    for (int u = 0; u < n; ++u) {
        int base_u = g.row_ptr[u];
        for (int k = 0; k < (int)adj[u].size(); ++k) {
            EdgeTmp &e = adj[u][k];
            int v      = e.to;
            int idx_u  = base_u + k;
            int idx_v  = g.row_ptr[v] + e.rev;
            g.rev[idx_u] = idx_v;
        }
    }

    auto b_end = chrono::high_resolution_clock::now();
    build_time_ms = chrono::duration_cast<chrono::milliseconds>(b_end - b_start).count();
}


// Compute each vertex score from CSR graph g and generate part[v]
void build_capacity_degree_partition_from_stats(
    int n, int R,
    const vector<long long>& sum_cap_out,
    const vector<long long>& sum_cap_in,
    const vector<int>& deg_out,
    const vector<int>& deg_in,
    vector<int>& part)
{
    struct VInfo { int v; double score; };
    vector<VInfo> vinfos(n);

    for (int u = 0; u < n; ++u) {
        double so = (double)sum_cap_out[u];
        double si = (double)sum_cap_in[u];
        double do_ = (double)deg_out[u];
        double di  = (double)deg_in[u];

        double num = std::log(so * do_ + 1.0) + std::log(si * di + 1.0);

        double prod_deg = do_ * di;
        double denom = std::log(std::sqrt(prod_deg) + 1.0);

        double score;
        if (denom > 0.0) {
            score = num / denom;
        } else {
            score = 1.0;  
        }

        vinfos[u] = {u, score};
    }

    sort(vinfos.begin(), vinfos.end(),
         [](const VInfo& a, const VInfo& b){ return a.score > b.score; });

    part.assign(n, 0);
    vector<double> load(R, 0.0);

    for (auto &info : vinfos) {
        int v = info.v;
        double s = info.score;
        int best_r = 0;
        double best_load = load[0];
        for (int r = 1; r < R; ++r)
            if (load[r] < best_load) { best_load = load[r]; best_r = r; }
        part[v] = best_r;
        load[best_r] += s;
    }
}


// Reorder the graph into region-ordered CSR using the given part[v] partition
void partition_and_reorder(const CSRGraph& g_in,
                           CSRGraph& g_out,
                           int R,
                           const vector<int>& part,
                           vector<int>& new_id,
                           vector<int>& old_id,
                           vector<int>& region_vertex_start,
                           vector<int>& region_vertex_end) {
    int n = g_in.n;
    int m = g_in.m;

    vector<vector<int>> vertices_in_region(R);
    for (int v = 0; v < n; ++v) {
        int r = part[v];
        if (r < 0 || r >= R) r = 0;
        vertices_in_region[r].push_back(v);
    }

    new_id.assign(n, -1);
    old_id.assign(n, -1);

    int cur = 0;
    region_vertex_start.resize(R);
    region_vertex_end.resize(R);
    for (int r = 0; r < R; ++r) {
        region_vertex_start[r] = cur;
        for (int v : vertices_in_region[r]) {
            new_id[v]   = cur;
            old_id[cur] = v;
            cur++;
        }
        region_vertex_end[r] = cur;  // [start, end)
    }

    g_out.n = n;
    g_out.m = m;
    g_out.row_ptr = new int[n + 1];
    g_out.col_idx = new int[m];
    g_out.cap     = new flow_t[m];
    g_out.rev     = new int[m];

    vector<int> deg(n, 0);
    for (int old_u = 0; old_u < n; ++old_u) {
        int new_u = new_id[old_u];
        for (int e = g_in.row_ptr[old_u]; e < g_in.row_ptr[old_u+1]; ++e) {
            deg[new_u]++;
        }
    }

    g_out.row_ptr[0] = 0;
    for (int u = 0; u < n; ++u) {
        g_out.row_ptr[u+1] = g_out.row_ptr[u] + deg[u];
    }
    vector<int> cursor(n);
    for (int u = 0; u < n; ++u) cursor[u] = g_out.row_ptr[u];

    vector<int> old_to_new_edge(m, -1);

    for (int old_u = 0; old_u < n; ++old_u) {
        int new_u = new_id[old_u];
        for (int e = g_in.row_ptr[old_u]; e < g_in.row_ptr[old_u+1]; ++e) {
            int old_v = g_in.col_idx[e];
            int new_v = new_id[old_v];
            int pos   = cursor[new_u]++;
            g_out.col_idx[pos] = new_v;
            g_out.cap[pos]     = g_in.cap[e];
            old_to_new_edge[e] = pos;
        }
    }

    for (int old_u = 0; old_u < n; ++old_u) {
        for (int e = g_in.row_ptr[old_u]; e < g_in.row_ptr[old_u+1]; ++e) {
            int new_e     = old_to_new_edge[e];
            int old_rev_e = g_in.rev[e];
            int new_rev_e = old_to_new_edge[old_rev_e];
            g_out.rev[new_e] = new_rev_e;
        }
    }
}


void build_region_id(int n,
                     const vector<int>& region_vertex_start,
                     const vector<int>& region_vertex_end,
                     vector<int>& region_id) {
    int R = (int)region_vertex_start.size();
    region_id.assign(n, -1);
    for (int r = 0; r < R; ++r) {
        int s = region_vertex_start[r];
        int e = region_vertex_end[r];
        for (int v = s; v < e; ++v) {
            region_id[v] = r;
        }
    }
}

// -------------------- Initialize the global graph structure on the GPU --------------------

void free_global_graph_on_device() {
    if (!g_graph_inited) return;
    cudaFree(g_d_meta); g_d_meta = nullptr;
    cudaFree(g_d_data); g_d_data = nullptr;
    cudaFree(g_d_weight); g_d_weight = nullptr;
    cudaFree(g_d_parallel); g_d_parallel = nullptr;
    cudaFree(g_d_region_start); g_d_region_start = nullptr;
    cudaFree(g_d_region_end); g_d_region_end = nullptr;
    cudaFree(g_d_region_id); g_d_region_id = nullptr;
    g_graph_inited = false;
    g_V = g_E = g_R = 0;
}

void init_global_graph_on_device(
    CSRGraph& g,
    int R,
    const vector<int>& region_vertex_start,
    const vector<int>& region_vertex_end,
    const vector<int>& region_id
) {
    int V = g.n;
    int E = g.m;

    bool need_rebuild = (!g_graph_inited) || (V != g_V) || (E != g_E) || (R != g_R);
    if (need_rebuild) {
        free_global_graph_on_device();

        cudaMalloc(&g_d_meta,     sizeof(int)    * (V + 1));
        cudaMalloc(&g_d_data,     sizeof(int)    * E);
        cudaMalloc(&g_d_weight,   sizeof(flow_t) * E);
        cudaMalloc(&g_d_parallel, sizeof(int)    * E);

        cudaMemcpy(g_d_meta,     g.row_ptr, sizeof(int)    * (V + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(g_d_data,     g.col_idx, sizeof(int)    * E,       cudaMemcpyHostToDevice);
        cudaMemcpy(g_d_weight,   g.cap,     sizeof(flow_t) * E,       cudaMemcpyHostToDevice);
        cudaMemcpy(g_d_parallel, g.rev,     sizeof(int)    * E,       cudaMemcpyHostToDevice);

        cudaMalloc(&g_d_region_start, sizeof(int) * R);
        cudaMalloc(&g_d_region_end,   sizeof(int) * R);
        cudaMemcpy(g_d_region_start, region_vertex_start.data(), sizeof(int) * R, cudaMemcpyHostToDevice);
        cudaMemcpy(g_d_region_end,   region_vertex_end.data(),   sizeof(int) * R, cudaMemcpyHostToDevice);

        cudaMalloc(&g_d_region_id, sizeof(int) * V);
        cudaMemcpy(g_d_region_id, region_id.data(), sizeof(int) * V, cudaMemcpyHostToDevice);

        g_graph_inited = true;
        g_V = V;
        g_E = E;
        g_R = R;
    } else {
        cudaMemcpy(g_d_weight, g.cap, sizeof(flow_t) * E, cudaMemcpyHostToDevice);
    }
}

// -------------------- Initialize the DynamicState for a specific (s,t) query --------------------

void init_state_for_st(DynamicState& st, int V, int E, int R) {
    bool need_rebuild = (!st.inited) || (st.V != V) || (st.E != E) || (st.R != R);
    if (need_rebuild) {
        free_state_for_st(st);
        st.V = V; st.E = E; st.R = R;
        cudaMalloc(&st.d_residual_capacity,     sizeof(flow_t) * E);
        cudaMalloc(&st.d_rev_residual_capacity, sizeof(flow_t) * E);
        cudaMalloc(&st.d_excess,                sizeof(flow_t) * V);
        cudaMalloc(&st.d_height,                sizeof(int)    * V);
        cudaMalloc(&st.d_level,                 sizeof(int)    * V);

        cudaMalloc(&st.d_frontier,          sizeof(int) * V);
        cudaMalloc(&st.d_next_frontier,     sizeof(int) * V);
        cudaMalloc(&st.d_frontier_size,     sizeof(int));
        cudaMalloc(&st.d_next_frontier_size,sizeof(int));
        cudaMalloc(&st.d_active_count,      sizeof(int));
        cudaMalloc(&st.d_active_high,       sizeof(int));
        cudaMalloc(&st.d_fixed_edges,       sizeof(int));
        cudaMalloc(&st.d_hmin_active,       sizeof(int));
        cudaMalloc(&st.d_hmax_active,       sizeof(int));
        cudaMalloc(&st.d_region_active,     sizeof(int) * R);
        cudaMalloc(&st.d_pos_excess_sum,    sizeof(long long));
        cudaMalloc(&st.d_active_list,       sizeof(int) * V);
        cudaMalloc(&st.d_active_size,       sizeof(int));
        cudaMalloc(&st.d_is_reachable_to_t, sizeof(int) * V);
        cudaMalloc(&st.d_region_hot,        sizeof(int) * R);

        cudaMalloc(&st.d_reach_from_s, sizeof(int) * V);
        cudaMalloc(&st.d_reach_to_t,   sizeof(int) * V);
        cudaMalloc(&st.d_tmp_frontier, sizeof(int) * V);
        cudaMalloc(&st.d_tmp_next,     sizeof(int) * V);
        cudaMalloc(&st.d_tmp_fsize,    sizeof(int));
        cudaMalloc(&st.d_tmp_nsize,    sizeof(int));

        st.inited = true;
    }
}

// -------------------- Pruning: only compute reachability; do not directly modify heights --------------------

// void preprocess_pruning(int sink, DynamicState& st) {
//     int V = st.V;

//     // Disable pruning completely: mark all vertices as reachable
//     int one = 1;
//     // cudaMemset could also be used here, but write 1 explicitly for clarity
//     std::vector<int> h_reach(V, 1);
//     cudaMemcpy(st.d_is_reachable_to_t, h_reach.data(), sizeof(int) * V, cudaMemcpyHostToDevice);
// }

// -------------------- CPU side: semi-directed pruning --------------------
// For (s, t):
// forward: start from s and follow directed edges with cap > 0
// reverse: start from t and follow reverse directed edges with cap > 0 using the rev array

void build_reachable_refined_from_st(
    const CSRGraph& g,
    int s,
    int t,
    std::vector<int>& h_reach   // Output a 0/1 mask of length g.n
) {
    int V = g.n;
    h_reach.assign(V, 0);
    if (s < 0 || s >= V || t < 0 || t >= V) return;

    // forward: start from s and follow forward edges with cap > 0
    std::vector<char> fwd(V, 0), und(V, 0);
    {
        std::queue<int> q;
        fwd[s] = 1;
        q.push(s);
        while (!q.empty()) {
            int v = q.front(); q.pop();
            for (int e = g.row_ptr[v]; e < g.row_ptr[v+1]; ++e) {
                int u = g.col_idx[e];
                flow_t c = g.cap[e];
                if (c > 0 && !fwd[u]) {
                    fwd[u] = 1;
                    q.push(u);
                }
            }
        }
    }

    // undirected: start from t and treat the graph as undirected because zero-capacity reverse edges already exist
    {
        std::queue<int> q;
        und[t] = 1;
        q.push(t);
        while (!q.empty()) {
            int v = q.front(); q.pop();
            for (int e = g.row_ptr[v]; e < g.row_ptr[v+1]; ++e) {
                int u = g.col_idx[e];
                if (!und[u]) {
                    und[u] = 1;
                    q.push(u);
                }
            }
        }
    }

    // Intersection: directed-reachable from s and in the same undirected component as t
    for (int v = 0; v < V; ++v) {
        if (fwd[v] && und[v]) h_reach[v] = 1;
    }

    // Ensure s and t are retained
    h_reach[s] = 1;
    h_reach[t] = 1;
}


// -------------------- Main loop: reuse graph structure plus per-(s,t) residual state --------------------

void pr_main_loop_static(
    int source,
    int sink,
    DynamicState& st
) {
    int V = st.V;
    int E = st.E;
    int R = st.R;

    int  RM_FREQ_dyn = 64;
    long long h_pos_excess_sum = 0;

    unsigned numThreads = (V < 1024) ? V : 1024;
    unsigned numBlocks  = (V + numThreads - 1) / numThreads;

    int  avg_deg   = (V > 0) ? (E / V) : 0;
    int GR_FREQ0 = 128;
    int kernel_cycles_base = max(4, avg_deg);
    const int RM_FREQ0  = 64;
    const float tau_h   = 0.8f;
    const int H_cut     = 100;
    
    int STAGNATE_ROUNDS = 128;
    int stagnation_count = 0;
    long long last_pos_excess_sum = LLONG_MAX;
    bool force_global_relabel = false;
    const int ITER_MAX = 20000000;

    int no_stag_window_rounds = 0;

    const int PATH_PUSH_SWITCH = 8;
    int MAX_PATH_LENGTH = 256;
    int path_push_stable_counter = 0;
    int p_count = 64;
    int push_count_iter = 0;

    bool h_flag = true;
    int  iter   = 0;

    do {
        float ms_init_hl = 0.0f;
        float ms_bfs     = 0.0f;
        float ms_pr      = 0.0f;
        float ms_rm      = 0.0f;
        float hot_pr      = 0.0f;

        bool do_global_relabel = (iter % GR_FREQ0 == 0) || force_global_relabel;
        force_global_relabel = false;

        if (do_global_relabel) {
            cudaMemset(st.d_frontier_size, 0, sizeof(int));

            cudaEvent_t ev_init_hl_start, ev_init_hl_end;
            cudaEvent_t ev_bfs_start, ev_bfs_end;
            cudaEventCreate(&ev_init_hl_start);
            cudaEventCreate(&ev_init_hl_end);
            cudaEventCreate(&ev_bfs_start);
            cudaEventCreate(&ev_bfs_end);

            // ---------- Global relabel: initialize height/level with pruning ----------
            cudaEventRecord(ev_init_hl_start, 0);
            init_height_level_frontier<<<numBlocks, numThreads>>>(
                source, sink, V,
                st.d_excess,
                st.d_level,
                st.d_height,
                st.d_frontier,
                st.d_frontier_size,
                st.d_is_reachable_to_t   // Added: pruning constraint
            );
            cudaError_t err_init_height_level_frontier = cudaGetLastError();
            if (err_init_height_level_frontier != cudaSuccess) {
                fprintf(stderr, "CUDA error after init_height_level_frontier: %s\n",
                        cudaGetErrorString(err_init_height_level_frontier));
                abort();
            }
            cudaEventRecord(ev_init_hl_end, 0);
            cudaEventSynchronize(ev_init_hl_end);
            cudaEventElapsedTime(&ms_init_hl, ev_init_hl_start, ev_init_hl_end);

            cudaEventRecord(ev_bfs_start, 0);

            int h_frontier_size = 0;
            int h_next_frontier_size = 0;
            int level = 0;

            cudaMemcpy(&h_frontier_size, st.d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);

            while (h_frontier_size > 0) {
                cudaMemset(st.d_next_frontier_size, 0, sizeof(int));

                int threads = 256;
                int blocks  = (h_frontier_size + threads - 1) / threads;

                bfs_expand_frontier<<<blocks, threads>>>(
                    V,
                    g_d_meta,
                    g_d_data,
                    st.d_rev_residual_capacity,
                    st.d_level,
                    st.d_height,
                    st.d_frontier,
                    h_frontier_size,
                    level,
                    st.d_next_frontier,
                    st.d_next_frontier_size
                );
                cudaError_t err_bfs_expand_frontier = cudaGetLastError();
                if (err_bfs_expand_frontier != cudaSuccess) {
                    fprintf(stderr, "CUDA error after bfs_expand_frontier: %s\n",
                            cudaGetErrorString(err_bfs_expand_frontier));
                    abort();
                }
                cudaDeviceSynchronize();

                cudaMemcpy(&h_next_frontier_size, st.d_next_frontier_size,
                           sizeof(int), cudaMemcpyDeviceToHost);
                if (h_next_frontier_size == 0) break;

                std::swap(st.d_frontier, st.d_next_frontier);
                h_frontier_size = h_next_frontier_size;
                level++;
            }

            cudaEventRecord(ev_bfs_end, 0);
            cudaEventSynchronize(ev_bfs_end);
            cudaEventElapsedTime(&ms_bfs, ev_bfs_start, ev_bfs_end);

            cudaEventDestroy(ev_init_hl_start);
            cudaEventDestroy(ev_init_hl_end);
            cudaEventDestroy(ev_bfs_start);
            cudaEventDestroy(ev_bfs_end);
        } else {
            if (h_flag) {
                int hmin_init = V;
                int hmax_init = 0;
                cudaMemcpy(st.d_hmin_active, &hmin_init, sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(st.d_hmax_active, &hmax_init, sizeof(int), cudaMemcpyHostToDevice);

                compute_active_height_range<<<numBlocks, numThreads>>>(
                    V,
                    source,
                    sink,
                    st.d_excess,
                    st.d_height,
                    st.d_hmin_active,
                    st.d_hmax_active
                );
                cudaError_t err_compute_active_height_range = cudaGetLastError();
                if (err_compute_active_height_range != cudaSuccess) {
                    fprintf(stderr, "CUDA error after compute_active_height_range_1: %s\n",
                            cudaGetErrorString(err_compute_active_height_range));
                    abort();
                }

                int h_min_a = 0, h_max_a = 0;
                cudaMemcpy(&h_min_a, st.d_hmin_active, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_max_a, st.d_hmax_active, sizeof(int), cudaMemcpyDeviceToHost);

                if (!(h_min_a > h_max_a || h_min_a == V)) {
                    int band_width = h_max_a - h_min_a + 1;
                    if (band_width <= 128) {
                        int margin = V;
                        int h_min_band = max(0,     h_min_a - margin);
                        int h_max_band = min(V - 1, h_max_a + margin);

                        cudaMemset(st.d_region_active, 0, sizeof(int) * R);
                        int threads_region = 256;
                        compute_region_active<<<R, threads_region>>>(
                            R,
                            g_d_region_start,
                            g_d_region_end,
                            st.d_excess,
                            st.d_region_active
                        );
                        cudaError_t err_compute_region_active = cudaGetLastError();
                        if (err_compute_region_active != cudaSuccess) {
                            fprintf(stderr, "CUDA error after compute_region_active: %s\n",
                                    cudaGetErrorString(err_compute_region_active));
                            abort();
                        }

                        cudaMemset(st.d_frontier_size, 0, sizeof(int));

                        cudaEvent_t ev_init_hl_start, ev_init_hl_end;
                        cudaEvent_t ev_bfs_start, ev_bfs_end;
                        cudaEventCreate(&ev_init_hl_start);
                        cudaEventCreate(&ev_init_hl_end);
                        cudaEventCreate(&ev_bfs_start);
                        cudaEventCreate(&ev_bfs_end);

                        // ---------- Local relabel: only in height band, active regions, and vertices reachable to t ----------
                        cudaEventRecord(ev_init_hl_start, 0);
                        init_local_frontier_in_band<<<numBlocks, numThreads>>>(
                            source, sink, V,
                            st.d_excess,
                            st.d_level,
                            st.d_height,
                            st.d_frontier,
                            st.d_frontier_size,
                            h_min_band,
                            h_max_band,
                            g_d_region_start,
                            g_d_region_end,
                            st.d_region_active,
                            g_d_region_id,
                            st.d_is_reachable_to_t   // Added: pruning constraint
                        );
                        cudaError_t err_init_local_frontier_in_band = cudaGetLastError();
                        if (err_init_local_frontier_in_band != cudaSuccess) {
                            fprintf(stderr, "CUDA error after init_local_frontier_in_band: %s\n",
                                    cudaGetErrorString(err_init_local_frontier_in_band));
                            abort();
                        }
                        cudaEventRecord(ev_init_hl_end, 0);
                        cudaEventSynchronize(ev_init_hl_end);
                        cudaEventElapsedTime(&ms_init_hl, ev_init_hl_start, ev_init_hl_end);

                        cudaEventRecord(ev_bfs_start, 0);
                        int h_frontier_size2 = 0, h_next_frontier_size2 = 0;
                        int level2 = 0;

                        cudaMemcpy(&h_frontier_size2, st.d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);

                        while (h_frontier_size2 > 0 && level2 <= band_width + 2) {
                            cudaMemset(st.d_next_frontier_size, 0, sizeof(int));

                            int threads = 256;
                            int blocks  = (h_frontier_size2 + threads - 1) / threads;

                            bfs_expand_frontier<<<blocks, threads>>>(
                                V,
                                g_d_meta,
                                g_d_data,
                                st.d_rev_residual_capacity,
                                st.d_level,
                                st.d_height,
                                st.d_frontier,
                                h_frontier_size2,
                                level2,
                                st.d_next_frontier,
                                st.d_next_frontier_size
                            );
                            cudaError_t err_bfs_expand_frontier = cudaGetLastError();
                            if (err_bfs_expand_frontier != cudaSuccess) {
                                fprintf(stderr, "CUDA error after bfs_expand_frontier: %s\n",
                                        cudaGetErrorString(err_bfs_expand_frontier));
                                abort();
                            }
                            cudaDeviceSynchronize();

                            cudaMemcpy(&h_next_frontier_size2, st.d_next_frontier_size,
                                       sizeof(int), cudaMemcpyDeviceToHost);
                            if (h_next_frontier_size2 == 0) break;

                            std::swap(st.d_frontier, st.d_next_frontier);
                            h_frontier_size2 = h_next_frontier_size2;
                            level2++;
                        }

                        cudaEventRecord(ev_bfs_end, 0);
                        cudaEventSynchronize(ev_bfs_end);
                        cudaEventElapsedTime(&ms_bfs, ev_bfs_start, ev_bfs_end);

                        cudaEventDestroy(ev_init_hl_start);
                        cudaEventDestroy(ev_init_hl_end);
                        cudaEventDestroy(ev_bfs_start);
                        cudaEventDestroy(ev_bfs_end);
                    }
                }
            }
        }

        int kernel_cycles_this_iter = kernel_cycles_base;
        int blocks_region = R;
        int threads_region = 512;

        cudaEvent_t ev_pr_start, ev_pr_end;
        cudaEventCreate(&ev_pr_start);
        cudaEventCreate(&ev_pr_end);

        int zero_active = 0;
        int zero_high   = 0;
        int zero_size   = 0;
        cudaMemcpy(st.d_active_count, &zero_active, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(st.d_active_high,  &zero_high,   sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(st.d_active_size,  &zero_size,   sizeof(int), cudaMemcpyHostToDevice);

        build_active_list_and_count<<<numBlocks, numThreads>>>(
            V,
            source,
            sink,
            st.d_excess,
            st.d_height,
            st.d_active_list,
            st.d_active_size,
            st.d_active_count,
            st.d_active_high,
            H_cut
        );
        cudaError_t err_build_active_list_and_count = cudaGetLastError();
        if (err_build_active_list_and_count != cudaSuccess) {
            fprintf(stderr, "CUDA error after build_active_list_and_count: %s\n",
                    cudaGetErrorString(err_build_active_list_and_count));
            abort();
        }

        int h_active      = 0;
        int h_active_high = 0;
        int h_active_size = 0;

        cudaMemcpy(&h_active,      st.d_active_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_active_high, st.d_active_high,  sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_active_size, st.d_active_size,  sizeof(int), cudaMemcpyDeviceToHost);

        if (h_active > 0 && h_active <= PATH_PUSH_SWITCH) { 
            path_push_stable_counter++;
        } else {
            path_push_stable_counter = 0;
            p_count = 64;
        }

        bool use_path_push = (h_active > 0 && h_active <= PATH_PUSH_SWITCH && path_push_stable_counter >= p_count);

        cudaEventRecord(ev_pr_start, 0);
        if(!use_path_push) {
            cudaMemcpy(st.d_active_count, &zero_active, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(st.d_active_high,  &zero_high,   sizeof(int), cudaMemcpyHostToDevice);

            staticMaxFlow_kernel_14_region<<<blocks_region, threads_region>>>(
                source, sink, kernel_cycles_this_iter,
                V, E,
                g_d_region_start,
                g_d_region_end,
                g_d_meta, g_d_data,
                st.d_excess,
                g_d_parallel,
                st.d_rev_residual_capacity,
                st.d_residual_capacity,
                st.d_height,
                st.d_active_count,
                st.d_active_high,
                H_cut,
                nullptr   // Normal round: process all regions
            );
            cudaError_t err1 = cudaGetLastError();
            if (err1 != cudaSuccess) {
                fprintf(stderr, "CUDA error after kernel_14_region: %s\n", cudaGetErrorString(err1));
                abort();
            }
        } else {
            int threads = 256;
            int blocks  = 4 * (h_active_size + threads - 1) / threads;

            staticMaxFlow_kernel_14_active_list_with_path_push<<<blocks, threads>>>(
                h_active_size,
                st.d_active_list,
                source,
                sink,
                V,
                g_d_meta,
                g_d_data,
                st.d_excess,
                g_d_parallel,
                st.d_rev_residual_capacity,
                st.d_residual_capacity,
                st.d_height,
                MAX_PATH_LENGTH
            );
            cudaError_t err2 = cudaGetLastError();
            if (err2 != cudaSuccess) {
                fprintf(stderr, "CUDA error after kernel_14_active_list_with_path_push: %s\n", cudaGetErrorString(err2));
                abort();
            }

            path_push_stable_counter = 0;
            push_count_iter++;
            if(push_count_iter >= 4){
                push_count_iter = 0;
                p_count=2;
            }
        }


        cudaEventRecord(ev_pr_end, 0);
        cudaEventSynchronize(ev_pr_end);
        cudaEventElapsedTime(&ms_pr, ev_pr_start, ev_pr_end);
        cudaEventDestroy(ev_pr_start);
        cudaEventDestroy(ev_pr_end);

        if (!use_path_push) {
            cudaMemcpy(&h_active,      st.d_active_count, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_active_high, st.d_active_high,  sizeof(int), cudaMemcpyDeviceToHost);
        }

        float frac_high = (h_active > 0) ? (float)h_active_high / (float)h_active : 0.0f;

        if (frac_high > tau_h && (iter % 2048 == 0)) {
            force_global_relabel = true;
        }

        long long zero_ll = 0;
        cudaMemcpy(st.d_pos_excess_sum, &zero_ll, sizeof(long long), cudaMemcpyHostToDevice);

        sum_positive_excess_kernel<<<numBlocks, numThreads>>>(
            V, source, sink, st.d_excess, st.d_pos_excess_sum
        );

        cudaMemcpy(&h_pos_excess_sum, st.d_pos_excess_sum, sizeof(long long), cudaMemcpyDeviceToHost);

        if (last_pos_excess_sum == LLONG_MAX) {
            stagnation_count = 0;
        } else {
            if (h_pos_excess_sum >= last_pos_excess_sum) {
                stagnation_count++;
            } else {
                stagnation_count = 0;
            }
        }
        last_pos_excess_sum = h_pos_excess_sum;
        bool hotspot_triggered = (stagnation_count >= STAGNATE_ROUNDS && (!use_path_push));


        if (hotspot_triggered) {
            force_global_relabel = true;
            stagnation_count = 0;

            GR_FREQ0 = min(256, GR_FREQ0 * 2);
            kernel_cycles_base = min(16, kernel_cycles_base * 2);
            kernel_cycles_this_iter = kernel_cycles_base;
            no_stag_window_rounds = 0;
            

        } else {
            no_stag_window_rounds++;
            if (no_stag_window_rounds >= STAGNATE_ROUNDS) {
                GR_FREQ0 = max(128, GR_FREQ0 / 2);
                kernel_cycles_base = max(4, kernel_cycles_base / 2);
                kernel_cycles_this_iter = kernel_cycles_base;
                no_stag_window_rounds = 0;
            }
        }

        if (iter > ITER_MAX) {
            fprintf(stderr, "Exceeded ITER_MAX=%d, aborting loop.\n", ITER_MAX);
            break;
        }

        // Extra hotspot acceleration: enable only when active count is high and path-push mode is not used
        bool region_hotspot = (h_active >= 64 && (iter % 2 == 0) && !hotspot_triggered);

        if (region_hotspot) {
            // 1) Count active vertices in each region
            int *d_region_active_cnt = nullptr;
            cudaMalloc(&d_region_active_cnt, sizeof(int) * R);
            cudaMemset(d_region_active_cnt, 0, sizeof(int) * R);

            int threads_cnt = 256;
            int blocks_cnt  = R;
            count_region_active_kernel<<<blocks_cnt, threads_cnt>>>(
                R,
                g_d_region_start,
                g_d_region_end,
                st.d_excess,
                d_region_active_cnt
            );
            cudaError_t err_count_region_active_kernel = cudaGetLastError();
            if (err_count_region_active_kernel != cudaSuccess) {
                fprintf(stderr, "CUDA error after count_region_active_kernel: %s\n",
                        cudaGetErrorString(err_count_region_active_kernel));
                abort();
            }

            // 2) Copy to host and select the top-K hottest regions
            std::vector<int> h_cnt(R);
            cudaMemcpy(h_cnt.data(), d_region_active_cnt,
                       sizeof(int) * R, cudaMemcpyDeviceToHost);

            // Free immediately after use
            cudaFree(d_region_active_cnt);
            
            std::vector<int> idx(R);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b){ return h_cnt[a] > h_cnt[b]; });

            const int K = 32;  // Select the top 32 hottest regions
            std::vector<int> h_hot(R, 0);
            for (int i = 0; i < K && i < R; ++i) {
                if (h_cnt[idx[i]] > 0) {
                    h_hot[idx[i]] = 1;
                }
            }

            // 3) Copy the hot mask to the GPU
            cudaMemcpy(st.d_region_hot, h_hot.data(),
                       sizeof(int) * R, cudaMemcpyHostToDevice);

            // 4) Rescan the top-K hot regions; other regions are skipped inside the kernel
            {
                int blocks_region_hot  = R;     // Still launch R blocks
                int threads_region_hot = 1024;   // Use 1024 threads for hot regions
                cudaEvent_t hot_pr_start, hot_pr_end;
                cudaEventCreate(&hot_pr_start);
                cudaEventCreate(&hot_pr_end);
                cudaEventRecord(hot_pr_start, 0);
                staticMaxFlow_kernel_14_region<<<blocks_region_hot, threads_region_hot>>>(
                    source, sink, kernel_cycles_this_iter,
                    V, E,
                    g_d_region_start,
                    g_d_region_end,
                    g_d_meta, g_d_data,
                    st.d_excess,
                    g_d_parallel,
                    st.d_rev_residual_capacity,
                    st.d_residual_capacity,
                    st.d_height,
                    nullptr,
                    nullptr,
                    H_cut,
                    st.d_region_hot   // Process only regions with hot=1
                );
                cudaError_t err_staticMaxFlow_kernel_14_region_hot = cudaGetLastError();
                if (err_staticMaxFlow_kernel_14_region_hot != cudaSuccess) {
                    fprintf(stderr, "CUDA error after staticMaxFlow_kernel_14_region (hot sweep): %s\n",
                            cudaGetErrorString(err_staticMaxFlow_kernel_14_region_hot));
                    abort();
                }
                cudaEventRecord(hot_pr_end, 0);
                cudaEventSynchronize(hot_pr_end);
                cudaEventElapsedTime(&hot_pr, hot_pr_start, hot_pr_end);
                cudaEventDestroy(hot_pr_start);
                cudaEventDestroy(hot_pr_end);
            }
        } 



        h_flag = (h_active > 0);

        // Run RM only when the high-level active ratio is large and the iteration reaches the RM frequency
        bool do_rm = (frac_high>=0.5f && (iter % RM_FREQ_dyn == 0) && h_flag && iter != 0);

        int h_fixed_edges = 0;
        if (do_rm) {
            int hmin_init = V;
            int hmax_init = 0;
            cudaMemcpy(st.d_hmin_active, &hmin_init, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(st.d_hmax_active, &hmax_init, sizeof(int), cudaMemcpyHostToDevice);

            compute_active_height_range<<<numBlocks, numThreads>>>(
                V,
                source,
                sink,
                st.d_excess,
                st.d_height,
                st.d_hmin_active,
                st.d_hmax_active
            );
            cudaError_t err_hr = cudaGetLastError();
            if (err_hr != cudaSuccess) {
                fprintf(stderr, "CUDA error after compute_active_height_range_2: %s\n",
                        cudaGetErrorString(err_hr));
                abort();
            }

            int h_min_a = 0, h_max_a = 0;
            cudaMemcpy(&h_min_a, st.d_hmin_active, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_max_a, st.d_hmax_active, sizeof(int), cudaMemcpyDeviceToHost);

            int h_min_rm = 0;
            int h_max_rm = V - 1;
            if (!(h_min_a > h_max_a || h_min_a == V)) {
                int margin = 4;
                h_min_rm = max(0, h_min_a - margin);
                h_max_rm = min(V - 1, h_max_a + margin);
            }


            int zero_fixed = 0;
            cudaMemcpy(st.d_fixed_edges, &zero_fixed, sizeof(int), cudaMemcpyHostToDevice);

            cudaMemset(st.d_region_active, 0, sizeof(int) * R);
            int threads_region_active = 256;
            compute_region_active<<<R, threads_region_active>>>(
                R,
                g_d_region_start,
                g_d_region_end,
                st.d_excess,
                st.d_region_active
            );
            cudaError_t err_ra = cudaGetLastError();
            if (err_ra != cudaSuccess) {
                fprintf(stderr, "CUDA error after compute_region_active: %s\n",
                        cudaGetErrorString(err_ra));
                abort();
            }

            cudaEvent_t ev_rm_start, ev_rm_end;
            cudaEventCreate(&ev_rm_start);
            cudaEventCreate(&ev_rm_end);

            cudaEventRecord(ev_rm_start, 0);
            staticMaxFlow_kernel_17_region<<<R, threads_region_active>>>(
                R,
                source, sink,
                V, E,
                g_d_region_start,
                g_d_region_end,
                st.d_region_active,
                g_d_meta, g_d_data,
                st.d_rev_residual_capacity,
                g_d_parallel,
                st.d_residual_capacity,
                st.d_excess,
                st.d_height,
                st.d_fixed_edges,
                h_min_rm,
                h_max_rm
            );
            cudaError_t err3 = cudaGetLastError();
            if (err3 != cudaSuccess) {
                fprintf(stderr, "CUDA error after kernel_17_region: %s\n", cudaGetErrorString(err3));
                abort();
            }

            cudaEventRecord(ev_rm_end, 0);
            cudaEventSynchronize(ev_rm_end);
            cudaEventElapsedTime(&ms_rm, ev_rm_start, ev_rm_end);

            cudaEventDestroy(ev_rm_start);
            cudaEventDestroy(ev_rm_end);

            cudaMemcpy(&h_fixed_edges, st.d_fixed_edges, sizeof(int), cudaMemcpyDeviceToHost);
            

            double frac_fix = (double)h_fixed_edges / (double)E;
            const double tau_rm = 1e-5;

            if (frac_fix < tau_rm) {
                RM_FREQ_dyn = min(RM_FREQ_dyn * 2, 256);
            } else {
                RM_FREQ_dyn = max(RM_FREQ0, RM_FREQ_dyn / 2);
            }
        } else {
            ms_rm = 0.0f;
            h_fixed_edges = 0;
        }

        const char* mode_str; 
        if (!use_path_push) {
            mode_str = "region-aware";
        } else {
            mode_str = "active-path-aware";
        } 

        fprintf(stderr, "Iter %d: init_hl=%.3f ms, bfs=%.3f ms, "
               "pr=%.3f ms (K=%d, mode=%s, D_path=%d, S_detected: %s, R_hotspot: %s), RM_invalid=%.3f ms, "
               "active=%d, active_high=%d, fixed_edges=%d, GR_FREQ0=%d\n\n",
               iter, ms_init_hl, ms_bfs, (ms_pr+hot_pr), kernel_cycles_this_iter,
               mode_str, MAX_PATH_LENGTH, hotspot_triggered ? "Yes" : "No", region_hotspot ? "Yes" : "No",
               ms_rm, h_active, h_active_high, h_fixed_edges, GR_FREQ0);

        iter++;
    } while (h_flag);
}

// -------------------- Static (s,t) solve using DynamicState --------------------

flow_t run_gpu_static_with_state(
    int source,
    int sink,
    DynamicState& st
) {
    int V = st.V, E = st.E;

    // Create CUDA events for timing
    cudaEvent_t ev_k1_start, ev_k1_end;
    cudaEvent_t ev_k7_start, ev_k7_end;
    cudaEventCreate(&ev_k1_start);
    cudaEventCreate(&ev_k1_end);
    cudaEventCreate(&ev_k7_start);
    cudaEventCreate(&ev_k7_end);

    unsigned numThreads = (V < 1024) ? V : 1024;
    unsigned numBlocks  = (V + numThreads - 1) / numThreads;

    // Pruning: compute vertices reachable from the sink under initial capacities and store in st.d_is_reachable_to_t
    // preprocess_pruning(sink, st); 

    // Initialize the residual graph
    cudaEventRecord(ev_k1_start, 0);
    staticMaxFlow_kernel_1<<<numBlocks, numThreads>>>(
        V,
        g_d_meta,
        g_d_data,
        g_d_weight,
        st.d_residual_capacity,
        st.d_rev_residual_capacity,
        g_d_parallel,
        st.d_excess
    );
    cudaError_t err_staticMaxFlow_kernel_1 = cudaGetLastError();
    if (err_staticMaxFlow_kernel_1 != cudaSuccess) {
        fprintf(stderr, "CUDA error after staticMaxFlow_kernel_1: %s\n",
                cudaGetErrorString(err_staticMaxFlow_kernel_1));
        abort();
    }
    cudaEventRecord(ev_k1_end, 0);
    cudaEventSynchronize(ev_k1_end);

    float ms_k1 = 0.0f;
    cudaEventElapsedTime(&ms_k1, ev_k1_start, ev_k1_end);

    // Source preflow
    cudaEventRecord(ev_k7_start, 0);
    staticMaxFlow_kernel_7<<<1,1>>>(
        source, V, E,
        g_d_meta,
        g_d_data,
        st.d_residual_capacity,
        st.d_excess,
        g_d_parallel,
        st.d_rev_residual_capacity
    );
    cudaError_t err_staticMaxFlow_kernel_7 = cudaGetLastError();
    if (err_staticMaxFlow_kernel_7 != cudaSuccess) {
        fprintf(stderr, "CUDA error after staticMaxFlow_kernel_7: %s\n",
                cudaGetErrorString(err_staticMaxFlow_kernel_7));
        abort();
    }

    cudaEventRecord(ev_k7_end, 0);
    cudaEventSynchronize(ev_k7_end);

    float ms_k7 = 0.0f;
    cudaEventElapsedTime(&ms_k7, ev_k7_start, ev_k7_end);

    fprintf(stderr, "[Init] kernel_1(init residual/excess)=%.3f ms, kernel_7(preflow)=%.3f ms\n",
           ms_k1, ms_k7);
    
    // Destroy events after use to avoid leaks
    cudaEventDestroy(ev_k1_start);
    cudaEventDestroy(ev_k1_end);
    cudaEventDestroy(ev_k7_start);
    cudaEventDestroy(ev_k7_end);

    cudaDeviceSynchronize();

    // Main loop; internally masks unreachable vertices permanently using d_is_reachable_to_t
    pr_main_loop_static(source, sink, st);

    flow_t max_flow = 0;
    cudaMemcpy(&max_flow, st.d_excess + sink, sizeof(flow_t), cudaMemcpyDeviceToHost);
    return max_flow;
}

// External static run_gpu interface: wrap the call with DynamicState internally
flow_t run_gpu(CSRGraph& h_g, int source, int sink,
               long long& query_time_ms,
               int R,
               const vector<int>& region_vertex_start,
               const vector<int>& region_vertex_end,
               const vector<int>& region_id) {
    auto q_start = chrono::high_resolution_clock::now();

    init_global_graph_on_device(h_g, R, region_vertex_start, region_vertex_end, region_id);

    DynamicState st;
    init_state_for_st(st, g_V, g_E, g_R);

    flow_t flow = run_gpu_static_with_state(source, sink, st);

    auto q_end = chrono::high_resolution_clock::now();
    query_time_ms = chrono::duration_cast<chrono::milliseconds>(q_end - q_start).count();

    free_state_for_st(st);
    return flow;
}

// -------------------- Dynamic mode: initialize state for a specific (s,t) query --------------------

flow_t init_max_flow_dynamic(
    CSRGraph& g,
    int source,
    int sink,
    int R,
    const vector<int>& region_vertex_start,
    const vector<int>& region_vertex_end,
    const vector<int>& region_id,
    DynamicState& st,
    long long& query_time_ms
) {
    auto t0 = chrono::high_resolution_clock::now();

    init_global_graph_on_device(g, R, region_vertex_start, region_vertex_end, region_id);
    init_state_for_st(st, g_V, g_E, g_R);

    // ===== Added: CPU-side directed pruning =====
    auto t_prune_start = chrono::high_resolution_clock::now();

    std::vector<int> h_reach;
    build_reachable_refined_from_st(g, source, sink, h_reach);
    cudaMemcpy(st.d_is_reachable_to_t,
            h_reach.data(),
            sizeof(int) * g_V,
            cudaMemcpyHostToDevice);
    
    auto t_prune_end = chrono::high_resolution_clock::now();
    long long prune_ms = chrono::duration_cast<chrono::milliseconds>(
                            t_prune_end - t_prune_start).count();
    fprintf(stderr, "[Prune-Dyn] prune_time=%lld ms\n", prune_ms);


    flow_t flow = run_gpu_static_with_state(source, sink, st);

    auto t1 = chrono::high_resolution_clock::now();
    query_time_ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    return flow;
}

// -------------------- Dynamic mode: incremental update on existing residual graph, capacity increase only --------------------

flow_t incremental_update_max_flow_with_state(
    int source,
    int sink,
    const vector<Update>& batch,
    DynamicState& st,
    long long& update_time_ms
) {
    auto t0 = chrono::high_resolution_clock::now();
    int V = st.V, E = st.E;

    // 1) Increase capacities on the GPU residual graph
    Update* d_batch = nullptr;
    int batch_size = (int)batch.size();
    cudaMalloc(&d_batch, sizeof(Update) * batch_size);
    cudaMemcpy(d_batch, batch.data(), sizeof(Update) * batch_size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (batch_size + threads - 1) / threads;
    OnAdd_kernel<<<blocks, threads>>>(
        d_batch, batch_size,
        g_d_meta,
        g_d_data,
        g_d_weight,             // Original capacity
        st.d_residual_capacity, // Forward residual capacity
        g_d_parallel
    );
    cudaError_t err4 = cudaGetLastError();
    if (err4 != cudaSuccess) {
        fprintf(stderr, "CUDA error after OnAdd_kernel: %s\n", cudaGetErrorString(err4));
        abort();
    }

    cudaDeviceSynchronize();
    cudaFree(d_batch);

    // 2) Run source preflow again
    staticMaxFlow_kernel_7<<<1,1>>>(
        source, V, E,
        g_d_meta,
        g_d_data,
        st.d_residual_capacity,
        st.d_excess,
        g_d_parallel,
        st.d_rev_residual_capacity
    );
    cudaError_t err5 = cudaGetLastError();
    if (err5 != cudaSuccess) {
        fprintf(stderr, "CUDA error after staticMaxFlow_kernel_7: %s\n", cudaGetErrorString(err5));
        abort();
    }

    cudaDeviceSynchronize();

    // 3) Run the main loop again on the existing residual graph
    pr_main_loop_static(source, sink, st);

    flow_t max_flow = 0;
    cudaMemcpy(&max_flow, st.d_excess + sink, sizeof(flow_t), cudaMemcpyDeviceToHost);

    auto t1 = chrono::high_resolution_clock::now();
    update_time_ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    return max_flow;
}


// -------------------- Query prune + Region/Boundary helpers --------------------

void build_region_coarse_graph(
    const CSRGraph& g,
    int R,
    const vector<int>& region_id,
    vector<vector<int>>& rg_adj,
    vector<vector<int>>& rg_radj
) {
    rg_adj.assign(R, {});
    rg_radj.assign(R, {});
    for (int v = 0; v < g.n; ++v) {
        int rv = region_id[v];
        for (int e = g.row_ptr[v]; e < g.row_ptr[v+1]; ++e) {
            int u = g.col_idx[e];
            int ru = region_id[u];
            if (rv != ru) {
                rg_adj[rv].push_back(ru);
                rg_radj[ru].push_back(rv);
            }
        }
    }
    for (int r = 0; r < R; ++r) {
        sort(rg_adj[r].begin(), rg_adj[r].end());
        rg_adj[r].erase(unique(rg_adj[r].begin(), rg_adj[r].end()), rg_adj[r].end());
        sort(rg_radj[r].begin(), rg_radj[r].end());
        rg_radj[r].erase(unique(rg_radj[r].begin(), rg_radj[r].end()), rg_radj[r].end());
    }
}

void mark_boundary_vertices(
    const CSRGraph& g,
    const vector<int>& region_id,
    vector<char>& is_boundary
) {
    is_boundary.assign(g.n, 0);
    for (int v = 0; v < g.n; ++v) {
        int rv = region_id[v];
        for (int e = g.row_ptr[v]; e < g.row_ptr[v+1]; ++e) {
            int u = g.col_idx[e];
            if (region_id[u] != rv) {
                is_boundary[v] = 1;
                break;
            }
        }
    }
}

long long mark_regions_on_any_st_path(
    int s_region,
    int t_region,
    const vector<vector<int>>& rg_adj,
    const vector<vector<int>>& rg_radj,
    vector<char>& keep_region
) {
    int R = (int)rg_adj.size();
    vector<char> from_s(R, 0), to_t(R, 0);
    queue<int> q;

    from_s[s_region] = 1;
    q.push(s_region);
    while (!q.empty()) {
        int r = q.front(); q.pop();
        for (int nr : rg_adj[r]) if (!from_s[nr]) {
            from_s[nr] = 1;
            q.push(nr);
        }
    }

    to_t[t_region] = 1;
    q.push(t_region);
    while (!q.empty()) {
        int r = q.front(); q.pop();
        for (int pr : rg_radj[r]) if (!to_t[pr]) {
            to_t[pr] = 1;
            q.push(pr);
        }
    }

    keep_region.assign(R, 0);
    long long kept = 0;
    for (int r = 0; r < R; ++r) {
        if (from_s[r] && to_t[r]) {
            keep_region[r] = 1;
            ++kept;
        }
    }
    return kept;
}

void build_boundary_aware_core_mask(
    const CSRGraph& g,
    const vector<int>& region_id,
    const vector<char>& keep_region,
    const vector<char>& is_boundary,
    int s,
    int t,
    int hop_limit,
    vector<int>& mask
) {
    int n = g.n;
    mask.assign(n, 0);
    if (s < 0 || s >= n || t < 0 || t >= n) return;

    int rs = region_id[s];
    int rt = region_id[t];
    vector<int> dist(n, -1);
    queue<int> q;

    for (int v = 0; v < n; ++v) {
        int r = region_id[v];
        if (!keep_region[r]) continue;
        if (r == rs || r == rt || is_boundary[v]) {
            mask[v] = 1;
            dist[v] = 0;
            q.push(v);
        }
    }

    while (!q.empty()) {
        int v = q.front(); q.pop();
        if (dist[v] >= hop_limit) continue;
        int rv = region_id[v];
        for (int e = g.row_ptr[v]; e < g.row_ptr[v+1]; ++e) {
            int u = g.col_idx[e];
            if (region_id[u] != rv) continue;
            if (!keep_region[rv]) continue;
            if (dist[u] == -1) {
                dist[u] = dist[v] + 1;
                mask[u] = 1;
                q.push(u);
            }
        }
    }

    mask[s] = 1;
    mask[t] = 1;
}

void build_induced_subgraph_by_mask(
    const CSRGraph& g_in,
    const vector<int>& mask,
    CSRGraph& g_out,
    vector<int>& old2sub,
    vector<int>& sub2old
) {
    int n = g_in.n;
    old2sub.assign(n, -1);
    sub2old.clear();

    for (int v = 0; v < n; ++v) {
        if (mask[v]) {
            old2sub[v] = (int)sub2old.size();
            sub2old.push_back(v);
        }
    }

    int sub_n = (int)sub2old.size();
    if (sub_n == 0) {
        g_out.n = 0;
        g_out.m = 0;
        g_out.row_ptr = nullptr;
        g_out.col_idx = nullptr;
        g_out.cap = nullptr;
        g_out.rev = nullptr;
        return;
    }

    vector<int> deg(sub_n, 0);
    for (int i = 0; i < sub_n; ++i) {
        int v_old = sub2old[i];
        for (int e = g_in.row_ptr[v_old]; e < g_in.row_ptr[v_old+1]; ++e) {
            int u_old = g_in.col_idx[e];
            if (old2sub[u_old] != -1) deg[i]++;
        }
    }

    g_out.n = sub_n;
    g_out.row_ptr = new int[sub_n + 1];
    g_out.row_ptr[0] = 0;
    for (int i = 0; i < sub_n; ++i) g_out.row_ptr[i+1] = g_out.row_ptr[i] + deg[i];
    g_out.m = g_out.row_ptr[sub_n];
    g_out.col_idx = new int[g_out.m];
    g_out.cap     = new flow_t[g_out.m];
    g_out.rev     = new int[g_out.m];

    vector<int> cursor(sub_n);
    for (int i = 0; i < sub_n; ++i) cursor[i] = g_out.row_ptr[i];
    vector<int> old_edge_to_sub(g_in.m, -1);

    for (int i = 0; i < sub_n; ++i) {
        int v_old = sub2old[i];
        for (int e = g_in.row_ptr[v_old]; e < g_in.row_ptr[v_old+1]; ++e) {
            int u_old = g_in.col_idx[e];
            int u_sub = old2sub[u_old];
            if (u_sub != -1) {
                int pos = cursor[i]++;
                g_out.col_idx[pos] = u_sub;
                g_out.cap[pos] = g_in.cap[e];
                old_edge_to_sub[e] = pos;
            }
        }
    }

    for (int i = 0; i < sub_n; ++i) {
        int v_old = sub2old[i];
        for (int e = g_in.row_ptr[v_old]; e < g_in.row_ptr[v_old+1]; ++e) {
            int se = old_edge_to_sub[e];
            if (se == -1) continue;
            int old_rev = g_in.rev[e];
            g_out.rev[se] = (old_rev >= 0 && old_rev < g_in.m) ? old_edge_to_sub[old_rev] : -1;
        }
    }
}

// -------------------- Utility function: create directory --------------------

void create_dir(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

// -------------------- Main function --------------------

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_dir> <output_dir> [mode]\n", argv[0]);
        fprintf(stderr, "  mode = static (default) | dynamic\n");
        return 1;
    }

    std::string input_dir  = argv[1];
    std::string output_dir = argv[2];
    std::string mode       = (argc >= 4 ? argv[3] : "static");

    create_dir(output_dir);
    cudaSetDevice(1);

    CSRGraph g_orig;
    long long build_time_ms = 0;
    vector<long long> sum_cap_out, sum_cap_in;
    vector<int> deg_out, deg_in;

    build_graph((input_dir + "/graph.txt").c_str(), g_orig, build_time_ms,
                sum_cap_out, sum_cap_in, deg_out, deg_in);

    int R = 128;

    vector<int> part;
    build_capacity_degree_partition_from_stats(
        g_orig.n, R,
        sum_cap_out, sum_cap_in,
        deg_out, deg_in,
        part
    );

    CSRGraph g;
    vector<int> new_id, old_id;
    vector<int> region_vertex_start, region_vertex_end;
    partition_and_reorder(g_orig, g, R, part,
                          new_id, old_id,
                          region_vertex_start, region_vertex_end);

    vector<int> region_id;
    build_region_id(g.n, region_vertex_start, region_vertex_end, region_id);

    FILE* tf = fopen((input_dir + "/test.txt").c_str(), "r");
    if (!tf) {
        fprintf(stderr, "Cannot open %s\n", (input_dir + "/test.txt").c_str());
        return 1;
    }

    vector<pair<int, int>> qs;
    int u, v;
    while (fscanf(tf, "%d %d", &u, &v) == 2) {
        int old_s = u - 1;
        int old_t = v - 1;
        if (old_s >= 0 && old_s < g.n && old_t >= 0 && old_t < g.n) {
            int new_s = new_id[old_s];
            int new_t = new_id[old_t];
            qs.push_back({new_s, new_t});
        } else {
            qs.push_back({-1, -1});
        }
    }
    fclose(tf);

    fprintf(stderr, "Graph (reordered by Region): n=%d, m=%d, Regions=%d, Queries=%zu\n",
        g.n, g.m, R, qs.size());


    if (mode == "static") {
        init_global_graph_on_device(
            g, R,
            region_vertex_start,
            region_vertex_end,
            region_id
        );

        const size_t Q = qs.size();
        long long total_query_time = 0;
        vector<flow_t> flows(Q);
        vector<long long> times(Q);

        DynamicState st;
        init_state_for_st(st, g_V, g_E, g_R);

        const bool ENABLE_QUERY_PRUNE = true;
        const bool ENABLE_REGION_BOUNDARY_ON_SUBGRAPH = true;
        const double MIN_PRUNE_SHRINK_RATIO = 0.995;   // Build a subgraph only if at least 0.5% of vertices are pruned
        const double MIN_CORE_SHRINK_RATIO  = 0.98;    // Continue to a boundary/core graph only if it shrinks by at least another 2%
        const int CORE_HOP_LIMIT = 2;

        for (size_t qi = 0; qi < Q; ++qi) {
            int s = qs[qi].first;
            int t = qs[qi].second;
            long long query_time = 0;

            int old_s = (s >= 0 && s < g.n) ? old_id[s] : -1;
            int old_t = (t >= 0 && t < g.n) ? old_id[t] : -1;
            fprintf(stderr, "[Static] Query %zu: s=%d, t=%d... ", qi + 1, old_s + 1, old_t + 1);

            if (s < 0 || s >= g.n || t < 0 || t >= g.n || s == t) {
                flows[qi] = 0;
                times[qi] = 0;
                fprintf(stderr, "Invalid.\n");
                continue;
            }

            auto t_prune_start = chrono::high_resolution_clock::now();
            vector<int> h_reach(g.n, 1);
            int pruned_n = g.n;
            long long pruned_m = g.m;
            bool use_pruned_subgraph = false;
            bool use_core_subgraph = false;
            CSRGraph g_sub{};
            CSRGraph g_core{};
            vector<int> old2sub, sub2old;
            vector<int> sub2core_old, sub_old2core;
            int run_s = s, run_t = t;

            if (ENABLE_QUERY_PRUNE) {
                build_reachable_refined_from_st(g, s, t, h_reach);
                pruned_n = 0;
                pruned_m = 0;
                for (int v = 0; v < g.n; ++v) {
                    if (h_reach[v]) {
                        ++pruned_n;
                        for (int e = g.row_ptr[v]; e < g.row_ptr[v+1]; ++e) {
                            if (h_reach[g.col_idx[e]]) ++pruned_m;
                        }
                    }
                }
                if (pruned_n > 0 && (double)pruned_n < MIN_PRUNE_SHRINK_RATIO * (double)g.n) {
                    build_induced_subgraph_by_mask(g, h_reach, g_sub, old2sub, sub2old);
                    if (g_sub.n > 0 && old2sub[s] >= 0 && old2sub[t] >= 0) {
                        use_pruned_subgraph = true;
                        run_s = old2sub[s];
                        run_t = old2sub[t];
                    }
                }
            }

            long long coarse_regions = 0;
            long long kept_vertices = use_pruned_subgraph ? g_sub.n : g.n;
            int run_n = use_pruned_subgraph ? g_sub.n : g.n;
            int run_m = use_pruned_subgraph ? g_sub.m : g.m;

            CSRGraph* run_graph = use_pruned_subgraph ? &g_sub : &g;
            int run_R = std::min(R, std::max(1, run_graph->n));
            vector<long long> sc_out(run_graph->n, 0), sc_in(run_graph->n, 0);
            vector<int> dg_out(run_graph->n, 0), dg_in(run_graph->n, 0);
            for (int v = 0; v < run_graph->n; ++v) {
                for (int e = run_graph->row_ptr[v]; e < run_graph->row_ptr[v+1]; ++e) {
                    flow_t c = run_graph->cap[e];
                    if (c > 0) {
                        int u = run_graph->col_idx[e];
                        sc_out[v] += c;
                        sc_in[u] += c;
                        dg_out[v]++;
                        dg_in[u]++;
                    }
                }
            }
            vector<int> part_run;
            build_capacity_degree_partition_from_stats(run_graph->n, run_R, sc_out, sc_in, dg_out, dg_in, part_run);

            CSRGraph g_run_re{};
            vector<int> run_new_id, run_old_id;
            vector<int> run_region_start, run_region_end;
            partition_and_reorder(*run_graph, g_run_re, run_R, part_run, run_new_id, run_old_id, run_region_start, run_region_end);
            vector<int> run_region_id;
            build_region_id(g_run_re.n, run_region_start, run_region_end, run_region_id);
            run_s = run_new_id[run_s];
            run_t = run_new_id[run_t];

            if (ENABLE_REGION_BOUNDARY_ON_SUBGRAPH && use_pruned_subgraph) {
                vector<vector<int>> rg_adj, rg_radj;
                build_region_coarse_graph(g_run_re, run_R, run_region_id, rg_adj, rg_radj);
                vector<char> keep_region, is_boundary;
                mark_boundary_vertices(g_run_re, run_region_id, is_boundary);
                coarse_regions = mark_regions_on_any_st_path(run_region_id[run_s], run_region_id[run_t], rg_adj, rg_radj, keep_region);
                vector<int> core_mask;
                build_boundary_aware_core_mask(g_run_re, run_region_id, keep_region, is_boundary, run_s, run_t, CORE_HOP_LIMIT, core_mask);
                kept_vertices = 0;
                for (int x : core_mask) kept_vertices += (x != 0);
                if (kept_vertices > 0 && (double)kept_vertices < MIN_CORE_SHRINK_RATIO * (double)g_run_re.n) {
                    build_induced_subgraph_by_mask(g_run_re, core_mask, g_core, sub_old2core, sub2core_old);
                    if (g_core.n > 0 && sub_old2core[run_s] >= 0 && sub_old2core[run_t] >= 0) {
                        use_core_subgraph = true;
                        run_s = sub_old2core[run_s];
                        run_t = sub_old2core[run_t];
                        run_n = g_core.n;
                        run_m = g_core.m;
                    }
                }
            }

            auto t_prune_end = chrono::high_resolution_clock::now();
            long long prune_ms = chrono::duration_cast<chrono::milliseconds>(t_prune_end - t_prune_start).count();

            CSRGraph* final_graph = &g_run_re;
            int final_R = run_R;
            vector<int>* final_region_start = &run_region_start;
            vector<int>* final_region_end = &run_region_end;
            vector<int>* final_region_id = &run_region_id;

            CSRGraph g_core_re{};
            vector<int> core_region_start, core_region_end, core_region_id, core_new_id, core_old_id;
            if (use_core_subgraph) {
                int core_R = std::min(R, std::max(1, g_core.n));
                vector<long long> core_sc_out(g_core.n, 0), core_sc_in(g_core.n, 0);
                vector<int> core_dg_out(g_core.n, 0), core_dg_in(g_core.n, 0);
                for (int v = 0; v < g_core.n; ++v) {
                    for (int e = g_core.row_ptr[v]; e < g_core.row_ptr[v+1]; ++e) {
                        flow_t c = g_core.cap[e];
                        if (c > 0) {
                            int u = g_core.col_idx[e];
                            core_sc_out[v] += c;
                            core_sc_in[u] += c;
                            core_dg_out[v]++;
                            core_dg_in[u]++;
                        }
                    }
                }
                vector<int> core_part;
                build_capacity_degree_partition_from_stats(g_core.n, core_R, core_sc_out, core_sc_in, core_dg_out, core_dg_in, core_part);
                partition_and_reorder(g_core, g_core_re, core_R, core_part, core_new_id, core_old_id, core_region_start, core_region_end);
                build_region_id(g_core_re.n, core_region_start, core_region_end, core_region_id);
                run_s = core_new_id[run_s];
                run_t = core_new_id[run_t];
                final_graph = &g_core_re;
                final_R = core_R;
                final_region_start = &core_region_start;
                final_region_end = &core_region_end;
                final_region_id = &core_region_id;
                run_n = g_core_re.n;
                run_m = g_core_re.m;
            } else {
                run_n = g_run_re.n;
                run_m = g_run_re.m;
            }

            init_global_graph_on_device(*final_graph, final_R, *final_region_start, *final_region_end, *final_region_id);
            init_state_for_st(st, final_graph->n, final_graph->m, final_R);
            std::vector<int> h_reach_gpu(final_graph->n, 1);
            cudaMemcpy(st.d_is_reachable_to_t, h_reach_gpu.data(), sizeof(int) * final_graph->n, cudaMemcpyHostToDevice);

            auto t0 = chrono::high_resolution_clock::now();
            flow_t flow = run_gpu_static_with_state(run_s, run_t, st);
            auto t1 = chrono::high_resolution_clock::now();
            query_time = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() + prune_ms;

            flows[qi] = flow;
            times[qi] = query_time;
            total_query_time += query_time;

            if (use_pruned_subgraph) {
                fprintf(stderr, "[QueryPrune+RegionBoundary] prune_time=%lld ms, pruned_n=%d/%d, pruned_m=%lld, coarse_regions=%lld/%d, kept_n=%lld/%d, run_n=%d, run_m=%d, Flow=%lld, Time=%lld ms\n",
                        prune_ms, pruned_n, g.n, pruned_m, coarse_regions, run_R, kept_vertices, g_run_re.n, run_n, run_m, (long long)flow, query_time);
            } else {
                fprintf(stderr, "[QueryPrune+RegionBoundary] prune_time=%lld ms, pruned_n=%d/%d, pruned_m=%lld, fallback=full, run_n=%d, run_m=%d, Flow=%lld, Time=%lld ms\n",
                        prune_ms, pruned_n, g.n, pruned_m, run_n, run_m, (long long)flow, query_time);
            }

            if (g_sub.row_ptr) { delete[] g_sub.row_ptr; delete[] g_sub.col_idx; delete[] g_sub.cap; delete[] g_sub.rev; }
            if (g_run_re.row_ptr) { delete[] g_run_re.row_ptr; delete[] g_run_re.col_idx; delete[] g_run_re.cap; delete[] g_run_re.rev; }
            if (g_core.row_ptr) { delete[] g_core.row_ptr; delete[] g_core.col_idx; delete[] g_core.cap; delete[] g_core.rev; }
            if (g_core_re.row_ptr) { delete[] g_core_re.row_ptr; delete[] g_core_re.col_idx; delete[] g_core_re.cap; delete[] g_core_re.rev; }
        }

        free_state_for_st(st);
        free_global_graph_on_device();

        std::string res_file = output_dir + "/res_mt.txt";
        FILE* out = fopen(res_file.c_str(), "w");
        if (!out) {
            fprintf(stderr, "Cannot open output file %s\n", res_file.c_str());
            return 1;
        }
        for (size_t qi = 0; qi < flows.size(); ++qi) {
            fprintf(out, "%lld %lld\n", (long long)flows[qi], (long long)times[qi]);
        }
        fclose(out);

        fprintf(stderr, "Total time: %lld ms\n", total_query_time);
    }
    else if (mode == "dynamic") {
        if (qs.empty()) {
            fprintf(stderr, "[Dynamic] No queries found.\n");
        } else {
            // Initialize the global graph structure on the GPU
            init_global_graph_on_device(g, R, region_vertex_start, region_vertex_end, region_id);

            const size_t Q = qs.size();
            const size_t WINDOW = 10; 
            long long total_dyn_time = 0; // Accumulate total runtime

            // Result file: res_dynamic.txt
            std::string dyn_file = output_dir + "/res_dynamic.txt";
            FILE* dout = fopen(dyn_file.c_str(), "w");
            if (!dout) {
                fprintf(stderr, "Cannot open dynamic output file %s\n", dyn_file.c_str());
                return 1;
            }

            // Process queries window by window
            for (size_t start_q = 0; start_q < Q; start_q += WINDOW) {
                size_t end_q  = std::min(start_q + WINDOW, Q);
                size_t cur_cnt = end_q - start_q;

                fprintf(stderr, "\n[Dynamic] ===== Processing queries [%zu, %zu) =====\n", start_q, end_q);

                vector<DynamicState> states(cur_cnt);
                vector<flow_t>       base_flows(cur_cnt);
                vector<long long>    init_times(cur_cnt);

                // 1) Initialize: compute the initial max-flow and record time
                for (size_t idx = 0; idx < cur_cnt; ++idx) {
                    size_t qi = start_q + idx;
                    int s = qs[qi].first;
                    int t = qs[qi].second;

                    if (s < 0 || s >= g.n || t < 0 || t >= g.n || s == t) {
                        base_flows[idx] = 0;
                        init_times[idx] = 0;
                        continue;
                    }

                    long long init_time = 0;
                    flow_t f0 = init_max_flow_dynamic(
                        g, s, t, R,
                        region_vertex_start, region_vertex_end, region_id,
                        states[idx], init_time
                    );
                    
                    init_times[idx] = init_time;
                    total_dyn_time += init_time;
                    base_flows[idx] = f0;

                    fprintf(stderr, "[Dynamic] Query %zu Init: flow=%lld, time=%lld ms\n",
                            qi + 1, (long long)f0, init_time);
                }

                // 2) Write results to file, one line as [flow time]
                for (size_t idx = 0; idx < cur_cnt; ++idx) {
                    fprintf(dout, "%lld %lld\n", (long long)base_flows[idx], init_times[idx]);
                }

                // 3) Release GPU memory for the current window
                for (size_t idx = 0; idx < cur_cnt; ++idx) {
                    free_state_for_st(states[idx]);
                }
            }

            fclose(dout);
            fprintf(stderr, "Total time: %lld ms\n", total_dyn_time);
            fprintf(stderr, "[Dynamic] Summary results written to %s\n", dyn_file.c_str());
        }
    }
    else {
        fprintf(stderr, "Unknown mode: %s (use static or dynamic)\n", mode.c_str());
        return 1;
    }

    delete[] g_orig.row_ptr;
    delete[] g_orig.col_idx;
    delete[] g_orig.cap;
    delete[] g_orig.rev;

    delete[] g.row_ptr;
    delete[] g.col_idx;
    delete[] g.cap;
    delete[] g.rev;

    return 0;
}
