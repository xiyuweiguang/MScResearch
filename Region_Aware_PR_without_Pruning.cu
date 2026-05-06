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

// -------------------- 基本类型与图结构 --------------------

typedef long long flow_t;

struct CSRGraph {
    int n;
    int m;
    int*    row_ptr;
    int*    col_idx;
    flow_t* cap;
    int*    rev;
};

// -------------------- 64 位原子操作（基于 atomicCAS，自定义有符号加法） --------------------

__device__ __forceinline__ long long atomicAddLongLong(long long* address, long long val) {
    // 把 long long* 当作 unsigned long long* 做 CAS
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


// -------------------- GPU kernel --------------------

// 初始化残余容量与 excess = 0
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

// 源点预流
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

// 全局 BFS 起点初始化（全图）
__global__ void init_height_level_frontier(
    int     source0,
    int     sink0,
    int     V,
    const flow_t* d_excess,
    int*    d_level,
    int*    d_height,
    int*    d_frontier,
    int*    d_frontier_size
) {
    unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= (unsigned)V) return;
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

// 局部 BFS 起点初始化：限定高度带 + region_active
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
    const int* __restrict__ d_region_id
) {
    unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= (unsigned)V) return;

    int r = d_region_id[v];   // O(1) 直接查 region
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

__global__ void compute_region_score(
    int R,
    int V,
    const int* __restrict__ d_region_start,
    const int* __restrict__ d_region_end,
    const flow_t* __restrict__ d_excess,
    long long* __restrict__ d_region_score,
    long long total_pos_excess
) {
    int r = blockIdx.x;
    if (r >= R) return;

    __shared__ long long sum_excess;
    __shared__ int count_active;

    if (threadIdx.x == 0) {
        sum_excess = 0;
        count_active = 0;
    }
    __syncthreads();

    for (int v = d_region_start[r] + threadIdx.x;
         v < d_region_end[r];
         v += blockDim.x) {

        flow_t ex = d_excess[v];
        if (ex > 0) {
            atomicAddLongLong(&sum_excess, (long long)ex);
            atomicAdd(&count_active, 1);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double ratio = (double)count_active / (double)V;

        long long score =
            sum_excess +
            (long long)(ratio * (double)total_pos_excess);

        d_region_score[r] = score;
    }
}

// BFS 一层 top-down
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

// push-relabel：region 版本
__global__ void staticMaxFlow_kernel_14_region(
    int     source0,
    int     sink0,
    int     kernel_cycles0,
    int     V,
    int     E,
    int     R,
    const int* __restrict__ d_region_start,
    const int* __restrict__ d_region_end,
    const int* __restrict__ d_region_block_offset,
    const int* __restrict__ d_region_blocks,
    const int* __restrict__ d_block_to_region,
    int*    d_meta,
    int*    d_data,
    flow_t* d_excess,
    int*    d_parallel_edge,
    flow_t* d_rev_residual_capacity,
    flow_t* d_residual_capacity,
    int*    d_height,
    int*    d_active_count,
    int*    d_active_high,
    int     H_cut
) {
    int bid = blockIdx.x;

    // 找 region
    int r = d_block_to_region[bid];   // O(1)

    // local block id
    int local_bid = bid - d_region_block_offset[r];

    // blocks per region
    int blocks_in_region = d_region_blocks[r];

    int start = d_region_start[r];
    int end   = d_region_end[r];

    float num_nodes = (float)V;

    int total_threads = blockDim.x * blocks_in_region;

    for (int v = start + local_bid * blockDim.x + threadIdx.x;
        v < end;
        v += total_threads) {
        int hv = d_height[v];

        if (d_excess[v] > 0 && v != source0 && v != sink0 && hv < num_nodes) {
            atomicAdd(d_active_count, 1);
            if (hv > H_cut) atomicAdd(d_active_high, 1);

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

// 计算活跃点高度范围（min/max）
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

// 统计每个 region 是否有活跃点：一块(block)一个 region
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

// 统计所有顶点（除 source/sink）正的 excess 总和（64-bit）
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


// region-aware rm：每个 block 一个 region，只处理 region_active[r] == 1 的
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
        if (h < h_min_rm || h > h_max_rm) continue;
        for (int edge = d_meta[v]; edge < d_meta[v+1]; edge++) {
            int vv = d_data[edge];
            int e  = edge;
            if (d_rev_residual_capacity[e] > 0) {
                if (d_height[vv] > h + 1) {
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
}


// -------------------- 一次扫描同时统计 active 和构建 active list --------------------

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
            // 统计 active
            atomicAdd(d_active_count, 1);
            if (h > H_cut) {
                atomicAdd(d_active_high, 1);
            }
            
            // 构建 active list
            int pos = atomicAdd(d_active_size, 1);
            d_active_list[pos] = v;
        }
    }
}


// 小 active 阶段：只对 active list 做 push-relabel，
// 每个线程尽量把自己的点 discharge 完再退出。
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

    // 为了防止极端死循环，限定一个最大循环次数
    const int MAX_CYCLES = 1000000;
    int cycle = 0;

    while (d_excess[v] > 0 && d_height[v] < V && cycle < MAX_CYCLES) {
        flow_t ex1 = d_excess[v];
        if (ex1 <= 0) break;

        int    hh  = INT_MAX;
        int    v_0 = -1;
        int    forward_edge = -1;

        // 寻找最小高度的可推进邻居
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
                // 所有出边都无残余容量，无法再推进，结束
                break;
            }
        }

        ++cycle;
    }
}

// -------------------- 新增：基于路径推流的 Active-List Push-Relabel --------------------

// 尝试从顶点 v 开始沿着高度梯度推流，返回是否成功推出 excess
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
        // 检查当前顶点是否还有 excess
        flow_t ex = d_excess[current];
        if (ex <= 0) return true; // 成功推空
        
        if (current == sink) return true; // 到达汇点
        
        int h_current = d_height[current];
        if (h_current >= V) return false; // 高度过高，无法推流
        
        // 寻找最佳的推流目标：高度最小且满足 h[current] = h[next] + 1
        int best_next = -1;
        int best_edge = -1;
        int min_height = INT_MAX;
        flow_t max_flow_can_push = 0;
        
        for (int edge = d_meta[current]; edge < d_meta[current + 1]; ++edge) {
            int next = d_data[edge];
            flow_t cap = d_residual_capacity[edge];
            
            if (cap > 0) {
                int h_next = d_height[next];
                
                // 优先选择高度正好是 h_current - 1 的邻居
                if (h_current == h_next + 1 && h_next < min_height) {
                    min_height = h_next;
                    best_next = next;
                    best_edge = edge;
                    max_flow_can_push = cap;
                }
            }
        }
        
        // 如果找到了合适的推流目标
        if (best_next != -1 && best_edge != -1) {
            flow_t delta = min(ex, max_flow_can_push);
            
            if (delta > 0) {
                // 执行推流
                atomicSubFlow(&d_excess[current], delta);
                atomicAddFlow(&d_excess[best_next], delta);
                atomicSubFlow(&d_residual_capacity[best_edge], delta);
                atomicAddFlow(&d_rev_residual_capacity[best_edge], delta);
                
                int rev_edge = d_parallel_edge[best_edge];
                atomicAddFlow(&d_residual_capacity[rev_edge], delta);
                atomicSubFlow(&d_rev_residual_capacity[rev_edge], delta);
                
                // 继续沿着这条路径推流
                current = best_next;
                path_len++;
                continue;
            }
        }
        
        // 无法推流，需要 relabel
        // 找到所有可推流邻居中的最小高度
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
            // relabel 后重试，但不增加 path_len（给它一次机会）
        } else {
            // 无法再 relabel 或者已经达到最大高度
            d_height[current] = V;
            return false;
        }
    }
    
    // 达到最大路径长度限制
    return d_excess[v] <= 0;
}

// 基于路径推流的 Active-List Push-Relabel kernel
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
    int     max_path_length  // 最大路径推流长度
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_size) return;

    int v = d_active_list[idx];
    if (v == source || v == sink) return;

    // 为了防止极端死循环，限定一个最大尝试次数
    const int MAX_ATTEMPTS = 2;
    int attempts = 0;

    while (d_excess[v] > 0 && d_height[v] < V && attempts < MAX_ATTEMPTS) {
        // 尝试沿路径推流
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
            // 路径推流失败，尝试传统的单步 push-relabel
            flow_t ex1 = d_excess[v];
            if (ex1 <= 0) break;
            
            int hh = INT_MAX;
            int v_0 = -1;
            int forward_edge = -1;

            // 寻找最小高度的可推进邻居
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


                

// -------------------- 主机端：构建 CSR --------------------

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

    // 初始化统计数组
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

        // 正向边 u->v，反向边 v->u
        EdgeTmp a{v, c, (int)adj[v].size()};
        EdgeTmp b{u, 0, (int)adj[u].size()};
        adj[u].push_back(a);
        adj[v].push_back(b);

        // 出入度 & 容量统计（只算原始正向边即可，反向边容量是 0）
        sum_cap_out[u] += c;
        sum_cap_in[v]  += c;
        deg_out[u]++;           // 原始图的出度
        deg_in[v]++;            // 原始图的入度
    }
    fclose(f);

    // 构建 CSR g
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


// 根据 CSR (g) 计算每个顶点的 score，并生成 part[v]
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


// 使用给定的 part[v] 分区结果，对图做 region-ordered CSR 重排
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

// -------------------- run_gpu：region-ordered CSR + region-aware rm + 局部 BFS + active-list --------------------

flow_t run_gpu(CSRGraph& h_g, int source, int sink,
               long long& query_time_ms,
               int R,
               const vector<int>& region_vertex_start,
               const vector<int>& region_vertex_end,
               const vector<int>& region_id) {
    auto q_start = chrono::high_resolution_clock::now();

    int V = h_g.n;
    int E = h_g.m;
    int RM_FREQ_dyn = 128;

    static bool inited = false;
    static int     *d_meta     = nullptr;
    static int     *d_data     = nullptr;
    static flow_t  *d_weight   = nullptr;
    static int     *d_parallel = nullptr;
    static int total_blocks_pr = 0;

    static int* d_block_to_region = nullptr;

    static long long* d_region_score      = nullptr;
    static int*       d_hot_region_flag   = nullptr;
    static int*       d_region_block_offset = nullptr;

    static flow_t* d_residual_capacity     = nullptr;
    static flow_t* d_rev_residual_capacity = nullptr;
    static flow_t* d_excess                = nullptr;
    static int*    d_height                = nullptr;
    static int*    d_level                 = nullptr;
    static int*    d_region_blocks         = nullptr;

    static int* d_frontier            = nullptr;
    static int* d_next_frontier       = nullptr;
    static int* d_frontier_size       = nullptr;
    static int* d_next_frontier_size  = nullptr;
    static int* d_active_count        = nullptr;
    static int* d_active_high         = nullptr;
    static int* d_fixed_edges         = nullptr;
    static int* d_hmin_active         = nullptr;
    static int* d_hmax_active         = nullptr;
    static int* d_region_active       = nullptr;

    static long long* d_pos_excess_sum = nullptr; 

    // active-list 相关
    static int* d_active_list = nullptr;
    static int* d_active_size = nullptr;

    static int cap_V = 0;
    static int cap_E = 0;

    static int *d_region_start = nullptr;
    static int *d_region_end   = nullptr;
    static int *d_region_id    = nullptr;

    cudaEvent_t ev_k1_start, ev_k1_end;
    cudaEvent_t ev_k7_start, ev_k7_end;
    cudaEvent_t ev_init_hl_start, ev_init_hl_end;
    cudaEvent_t ev_bfs_start,    ev_bfs_end;
    cudaEvent_t ev_pr_start,     ev_pr_end;
    cudaEvent_t ev_rm_start,     ev_rm_end;

    cudaEventCreate(&ev_k1_start);
    cudaEventCreate(&ev_k1_end);
    cudaEventCreate(&ev_k7_start);
    cudaEventCreate(&ev_k7_end);
    cudaEventCreate(&ev_init_hl_start);
    cudaEventCreate(&ev_init_hl_end);
    cudaEventCreate(&ev_bfs_start);
    cudaEventCreate(&ev_bfs_end);
    cudaEventCreate(&ev_pr_start);
    cudaEventCreate(&ev_pr_end);
    cudaEventCreate(&ev_rm_start);
    cudaEventCreate(&ev_rm_end);

    if (!inited) {
        cudaMalloc(&d_meta,     sizeof(int)    * (V + 1));
        cudaMalloc(&d_data,     sizeof(int)    * E);
        cudaMalloc(&d_weight,   sizeof(flow_t) * E);
        cudaMalloc(&d_parallel, sizeof(int)    * E);
        if (!d_region_blocks) {
            cudaMalloc(&d_region_blocks, sizeof(int) * R);
        }

        cudaMemcpy(d_meta,     h_g.row_ptr, sizeof(int)    * (V + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_data,     h_g.col_idx, sizeof(int)    * E,       cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight,   h_g.cap,     sizeof(flow_t) * E,       cudaMemcpyHostToDevice);
        cudaMemcpy(d_parallel, h_g.rev,     sizeof(int)    * E,       cudaMemcpyHostToDevice);

        cudaMalloc(&d_region_start, sizeof(int) * R);
        cudaMalloc(&d_region_end,   sizeof(int) * R);
        cudaMemcpy(d_region_start, region_vertex_start.data(), sizeof(int) * R, cudaMemcpyHostToDevice);
        cudaMemcpy(d_region_end,   region_vertex_end.data(),   sizeof(int) * R, cudaMemcpyHostToDevice);

        cudaMalloc(&d_region_id, sizeof(int) * V);
        cudaMemcpy(d_region_id, region_id.data(), sizeof(int) * V, cudaMemcpyHostToDevice);

        inited = true;
    } else {
        cudaMemcpy(d_weight, h_g.cap, sizeof(flow_t) * E, cudaMemcpyHostToDevice);
    }

    if (!d_residual_capacity || V > cap_V || E > cap_E) {
        if (d_residual_capacity) {
            cudaFree(d_residual_capacity);
            cudaFree(d_rev_residual_capacity);
            cudaFree(d_excess);
            cudaFree(d_height);
            cudaFree(d_level);

            cudaFree(d_frontier);
            cudaFree(d_next_frontier);
            cudaFree(d_frontier_size);
            cudaFree(d_next_frontier_size);
            cudaFree(d_active_count);
            cudaFree(d_active_high);
            cudaFree(d_fixed_edges);
            cudaFree(d_hmin_active);
            cudaFree(d_hmax_active);
            cudaFree(d_region_active);

            if (d_pos_excess_sum) cudaFree(d_pos_excess_sum);

            if (d_active_list) cudaFree(d_active_list);
            if (d_active_size) cudaFree(d_active_size);
        }

        cudaMalloc(&d_residual_capacity,     sizeof(flow_t) * E);
        cudaMalloc(&d_rev_residual_capacity, sizeof(flow_t) * E);
        cudaMalloc(&d_excess,                sizeof(flow_t) * V);
        cudaMalloc(&d_height,                sizeof(int)    * V);
        cudaMalloc(&d_level,                 sizeof(int)    * V);

        cudaMalloc(&d_frontier,          sizeof(int) * V);
        cudaMalloc(&d_next_frontier,     sizeof(int) * V);
        cudaMalloc(&d_frontier_size,     sizeof(int));
        cudaMalloc(&d_next_frontier_size,sizeof(int));
        cudaMalloc(&d_active_count,      sizeof(int));
        cudaMalloc(&d_active_high,       sizeof(int));
        cudaMalloc(&d_fixed_edges,       sizeof(int));
        cudaMalloc(&d_hmin_active,       sizeof(int));
        cudaMalloc(&d_hmax_active,       sizeof(int));
        cudaMalloc(&d_region_active,     sizeof(int) * R);

        cudaMalloc(&d_region_score,        sizeof(long long) * R);
        cudaMalloc(&d_hot_region_flag,     sizeof(int) * R);
        cudaMalloc(&d_region_block_offset, sizeof(int) * R);

        cudaMalloc(&d_pos_excess_sum,    sizeof(long long));  

        // active list
        cudaMalloc(&d_active_list, sizeof(int) * V);
        cudaMalloc(&d_active_size, sizeof(int));

        cap_V = V;
        cap_E = E;

        // ========= 默认每个 region 1 个 block，并构造 block_to_region =========
        {
            std::vector<int> h_blocks(R, 1);          // d_region_blocks
            std::vector<int> h_offset(R);             // d_region_block_offset
            std::vector<int> h_block_to_region(R);    // d_block_to_region

            int total_blocks = 0;
            for (int r = 0; r < R; ++r) {
                h_offset[r] = total_blocks;
                h_block_to_region[total_blocks] = r;
                total_blocks += 1;
            }
            total_blocks_pr = total_blocks;   // = R

            if (d_block_to_region) cudaFree(d_block_to_region);
            cudaMalloc(&d_block_to_region, sizeof(int) * total_blocks);

            cudaMemcpy(d_region_blocks, h_blocks.data(),
                       sizeof(int) * R, cudaMemcpyHostToDevice);
            cudaMemcpy(d_region_block_offset, h_offset.data(),
                       sizeof(int) * R, cudaMemcpyHostToDevice);
            cudaMemcpy(d_block_to_region, h_block_to_region.data(),
                       sizeof(int) * total_blocks, cudaMemcpyHostToDevice);

            std::vector<int> h_hot(R, 0);
            cudaMemcpy(d_hot_region_flag, h_hot.data(),
                       sizeof(int) * R, cudaMemcpyHostToDevice);
        }
    }


    long long h_pos_excess_sum = 0;
    unsigned numThreads = (V < 1024) ? V : 1024;
    unsigned numBlocks  = (V + numThreads - 1) / numThreads;

    // 1. 初始化残余容量和 excess
    cudaEventRecord(ev_k1_start, 0);
    staticMaxFlow_kernel_1<<<numBlocks, numThreads>>>(
        V, d_meta, d_data, d_weight,
        d_residual_capacity, d_rev_residual_capacity,
        d_parallel, d_excess
    );
    cudaEventRecord(ev_k1_end, 0);
    cudaEventSynchronize(ev_k1_end);

    float ms_k1 = 0.0f;
    cudaEventElapsedTime(&ms_k1, ev_k1_start, ev_k1_end);

    // 2. 源点预流
    cudaEventRecord(ev_k7_start, 0);
    staticMaxFlow_kernel_7<<<1, 1>>>(
        source, V, E,
        d_meta, d_data,
        d_residual_capacity,
        d_excess,
        d_parallel,
        d_rev_residual_capacity
    );
    cudaEventRecord(ev_k7_end, 0);
    cudaEventSynchronize(ev_k7_end);

    float ms_k7 = 0.0f;
    cudaEventElapsedTime(&ms_k7, ev_k7_start, ev_k7_end);

    printf("[Init] kernel_1(init residual/excess)=%.3f ms, kernel_7(preflow)=%.3f ms\n",
           ms_k1, ms_k7);

    cudaDeviceSynchronize();

    int  avg_deg   = (V > 0) ? (E / V) : 0;
    int GR_FREQ0 = 64;              // 可变的全局重标周期
    int kernel_cycles_base = max(4, avg_deg);  // 基于平均度的基本 push-relabel 循环次数
    const int RM_FREQ0  = 64;
    const float tau_h   = 0.8f;
    const int H_cut     = 1000;
    
    const int STAGNATE_ROUNDS = 128;
    int stagnation_count = 0;
    long long last_pos_excess_sum = LLONG_MAX;
    bool force_global_relabel = false;
    const int ITER_MAX = 20000000;

    int no_stag_window_rounds = 0;

    // active-list 模式的开关阈值
    const int ACTIVE_LIST_SWITCH = 16;  // 当活跃点 <= 16 时使用 active-list
    const int PATH_PUSH_SWITCH = 8;  // 当活跃点 <= 8 时使用路径推流
    int MAX_PATH_LENGTH = 256;   // 路径推流的最大深度
    int path_push_stable_counter = 0; // 记录连续满足活跃点 <= PATH_PUSH_SWITCH 的轮数
    int p_count = 32;
    int push_count_iter = 0;

    bool h_flag = true;
    int  iter   = 0;

    do {
        float ms_init_hl = 0.0f;
        float ms_bfs     = 0.0f;
        float ms_pr      = 0.0f;
        float ms_rm      = 0.0f;

        bool do_global_relabel = (iter % GR_FREQ0 == 0) || force_global_relabel;
        force_global_relabel = false;

        if (do_global_relabel) {
            cudaMemset(d_frontier_size, 0, sizeof(int));

            cudaEventRecord(ev_init_hl_start, 0);
            init_height_level_frontier<<<numBlocks, numThreads>>>(
                source, sink, V,
                d_excess,
                d_level,
                d_height,
                d_frontier,
                d_frontier_size
            );
            cudaEventRecord(ev_init_hl_end, 0);
            cudaEventSynchronize(ev_init_hl_end);
            cudaEventElapsedTime(&ms_init_hl, ev_init_hl_start, ev_init_hl_end);

            cudaEventRecord(ev_bfs_start, 0);

            int h_frontier_size = 0;
            int h_next_frontier_size = 0;
            int level = 0;

            cudaMemcpy(&h_frontier_size, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);

            while (h_frontier_size > 0) {
                cudaMemset(d_next_frontier_size, 0, sizeof(int));

                int threads = 256;
                int blocks  = (h_frontier_size + threads - 1) / threads;

                bfs_expand_frontier<<<blocks, threads>>>(
                    V,
                    d_meta,
                    d_data,
                    d_rev_residual_capacity,
                    d_level,
                    d_height,
                    d_frontier,
                    h_frontier_size,
                    level,
                    d_next_frontier,
                    d_next_frontier_size
                );
                cudaDeviceSynchronize();

                cudaMemcpy(&h_next_frontier_size, d_next_frontier_size,
                           sizeof(int), cudaMemcpyDeviceToHost);
                if (h_next_frontier_size == 0) break;

                std::swap(d_frontier, d_next_frontier);
                h_frontier_size = h_next_frontier_size;
                level++;
            }

            cudaEventRecord(ev_bfs_end, 0);
            cudaEventSynchronize(ev_bfs_end);
            cudaEventElapsedTime(&ms_bfs, ev_bfs_start, ev_bfs_end);
        } else {
            // 非 global 轮：局部 BFS（region + 高度带）
            if (h_flag) {
                int hmin_init = V;
                int hmax_init = 0;
                cudaMemcpy(d_hmin_active, &hmin_init, sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_hmax_active, &hmax_init, sizeof(int), cudaMemcpyHostToDevice);

                compute_active_height_range<<<numBlocks, numThreads>>>(
                    V,
                    source,
                    sink,
                    d_excess,
                    d_height,
                    d_hmin_active,
                    d_hmax_active
                );
                cudaDeviceSynchronize();

                int h_min_a = 0, h_max_a = 0;
                cudaMemcpy(&h_min_a, d_hmin_active, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_max_a, d_hmax_active, sizeof(int), cudaMemcpyDeviceToHost);

                if (!(h_min_a > h_max_a || h_min_a == V)) {
                    int band_width = h_max_a - h_min_a + 1;
                    if (band_width <= 128) { // 只在高度带较窄时做局部 BFS
                        int margin = 2;
                        int h_min_band = max(0,     h_min_a - margin);
                        int h_max_band = min(V - 1, h_max_a + margin);

                        // 先算 region_active
                        cudaMemset(d_region_active, 0, sizeof(int) * R);
                        int threads_region = 256;
                        compute_region_active<<<R, threads_region>>>(
                            R,
                            d_region_start,
                            d_region_end,
                            d_excess,
                            d_region_active
                        );
                        cudaDeviceSynchronize();

                        cudaMemset(d_frontier_size, 0, sizeof(int));

                        cudaEventRecord(ev_init_hl_start, 0);
                        init_local_frontier_in_band<<<numBlocks, numThreads>>>(
                            source, sink, V,
                            d_excess,
                            d_level,
                            d_height,
                            d_frontier,
                            d_frontier_size,
                            h_min_band,
                            h_max_band,
                            d_region_start,
                            d_region_end,
                            d_region_active,
                            d_region_id
                        );
                        cudaEventRecord(ev_init_hl_end, 0);
                        cudaEventSynchronize(ev_init_hl_end);
                        cudaEventElapsedTime(&ms_init_hl, ev_init_hl_start, ev_init_hl_end);

                        cudaEventRecord(ev_bfs_start, 0);
                        int h_frontier_size = 0, h_next_frontier_size = 0;
                        int level = 0;

                        cudaMemcpy(&h_frontier_size, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);

                        while (h_frontier_size > 0 && level <= band_width + 2) {
                            cudaMemset(d_next_frontier_size, 0, sizeof(int));

                            int threads = 256;
                            int blocks  = (h_frontier_size + threads - 1) / threads;

                            bfs_expand_frontier<<<blocks, threads>>>(
                                V,
                                d_meta,
                                d_data,
                                d_rev_residual_capacity,
                                d_level,
                                d_height,
                                d_frontier,
                                h_frontier_size,
                                level,
                                d_next_frontier,
                                d_next_frontier_size
                            );
                            cudaDeviceSynchronize();

                            cudaMemcpy(&h_next_frontier_size, d_next_frontier_size,
                                       sizeof(int), cudaMemcpyDeviceToHost);
                            if (h_next_frontier_size == 0) break;

                            std::swap(d_frontier, d_next_frontier);
                            h_frontier_size = h_next_frontier_size;
                            level++;
                        }

                        cudaEventRecord(ev_bfs_end, 0);
                        cudaEventSynchronize(ev_bfs_end);
                        cudaEventElapsedTime(&ms_bfs, ev_bfs_start, ev_bfs_end);
                    }
                }
            }
        }

        // 2) push‑relabel
        int kernel_cycles_this_iter = kernel_cycles_base;

        int blocks_region = (total_blocks_pr > 0 ? total_blocks_pr : R);
        int threads_region = 256;    // 可调

        cudaEventRecord(ev_pr_start, 0);
        int zero_active = 0;
        int zero_high   = 0;
        int zero_size   = 0;
        cudaMemcpy(d_active_count, &zero_active, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_active_high,  &zero_high,   sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_active_size,  &zero_size,   sizeof(int), cudaMemcpyHostToDevice);

        // ========== 关键修改：一次扫描同时统计和构建列表 ==========
        build_active_list_and_count<<<numBlocks, numThreads>>>(
            V,
            source,
            sink,
            d_excess,
            d_height,
            d_active_list,
            d_active_size,
            d_active_count,
            d_active_high,
            H_cut
        );
        cudaDeviceSynchronize();

        // 获取统计结果
        int h_active      = 0;
        int h_active_high = 0;
        int h_active_size = 0;

        cudaMemcpy(&h_active,      d_active_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_active_high, d_active_high,  sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_active_size, d_active_size,  sizeof(int), cudaMemcpyDeviceToHost);

        // ====================================================
        // 增加 path_push_stable_counter 的更新逻辑
        // ====================================================
        if (h_active > 0 && h_active <= PATH_PUSH_SWITCH) { 
            path_push_stable_counter++;
        } else {
            path_push_stable_counter = 0; // 不满足条件时重置计数器
            p_count = 64; // 重置稳定计数器上限
        }


        // 判断是否应该使用 active-aware 模式 (在没有路径推流时)
        bool use_active_list = (h_active > 0 && h_active <= ACTIVE_LIST_SWITCH);
        // 判断是否应该使用路径推流 path-push 模式
        bool use_path_push = (h_active > 0 && h_active <= PATH_PUSH_SWITCH && path_push_stable_counter >= p_count);
        // 判断是否应该使用 region-aware 模式
        // bool use_very_aggressive = (h_active > 0 && h_active > ACTIVE_LIST_SWITCH);



        if(!use_active_list) {
            // 正常 region 模式
            cudaMemcpy(d_active_count, &zero_active, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_active_high,  &zero_high,   sizeof(int), cudaMemcpyHostToDevice);

            staticMaxFlow_kernel_14_region<<<blocks_region, threads_region>>>(
                source, sink, kernel_cycles_this_iter,
                V, E, R,
                d_region_start,
                d_region_end,
                d_region_block_offset,
                d_region_blocks,
                d_block_to_region,  
                d_meta,
                d_data,
                d_excess,
                d_parallel,
                d_rev_residual_capacity,
                d_residual_capacity,
                d_height,
                d_active_count,
                d_active_high,
                H_cut
            );
        } else {
            if(use_active_list) {
                int threads = 128;
                int blocks  = (h_active_size + threads - 1) / threads;

                if(use_path_push) {
                // 小 active 模式：active-list
                // 使用路径推流版本
                staticMaxFlow_kernel_14_active_list_with_path_push<<<blocks, threads>>>(
                    h_active_size,
                    d_active_list,
                    source,
                    sink,
                    V,
                    d_meta,
                    d_data,
                    d_excess,
                    d_parallel,
                    d_rev_residual_capacity,
                    d_residual_capacity,
                    d_height,
                    MAX_PATH_LENGTH
                );

                path_push_stable_counter = 0;
                push_count_iter++;
                if(push_count_iter >= 64){
                    push_count_iter = 0;
                    p_count=2;
                }

            }
            else {
                // 使用普通 active-list 版本
                staticMaxFlow_kernel_14_active_list<<<blocks, threads>>>(
                    h_active_size,
                    d_active_list,
                    source,
                    sink,
                    V,
                    d_meta,
                    d_data,
                    d_excess,
                    d_parallel,
                    d_rev_residual_capacity,
                    d_residual_capacity,
                    d_height
                );
            }
            }
        }       


        cudaEventRecord(ev_pr_end, 0);
        cudaEventSynchronize(ev_pr_end);
        cudaEventElapsedTime(&ms_pr, ev_pr_start, ev_pr_end);

        // 如果使用了 region 模式，需要重新获取 active 统计
        if (!use_active_list) {
            cudaMemcpy(&h_active,      d_active_count, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_active_high, d_active_high,  sizeof(int), cudaMemcpyDeviceToHost);
        }


        float frac_high = (h_active > 0) ? (float)h_active_high / (float)h_active : 0.0f;

        if (frac_high > tau_h && (iter % 1024 == 0)) {
            force_global_relabel = true;
        }

        // --- 统计正的 excess 总和（用于停滞检测） ---
        long long zero_ll = 0;
        cudaMemcpy(d_pos_excess_sum, &zero_ll, sizeof(long long), cudaMemcpyHostToDevice);

        sum_positive_excess_kernel<<<numBlocks, numThreads>>>(
            V, source, sink, d_excess, d_pos_excess_sum
        );
        cudaDeviceSynchronize();

        cudaMemcpy(&h_pos_excess_sum, d_pos_excess_sum, sizeof(long long), cudaMemcpyDeviceToHost);

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

        if (stagnation_count >= STAGNATE_ROUNDS) {
            fprintf(stderr,
                "Stagnation detected: pos_excess_sum=%lld\n",
                (long long)h_pos_excess_sum);

            force_global_relabel = true;
            stagnation_count = 0;
            GR_FREQ0 = min(512, GR_FREQ0 * 2);
            kernel_cycles_base = min(32, kernel_cycles_base * 2);
            kernel_cycles_this_iter = kernel_cycles_base;
            no_stag_window_rounds = 0;

            // =========================
            // 计算 hotspot region
            // =========================
            cudaMemset(d_region_score, 0, sizeof(long long) * R);

            compute_region_score<<<R, 256>>>(
                R,
                V,
                d_region_start,
                d_region_end,
                d_excess,
                d_region_score,
                h_pos_excess_sum
            );
            cudaDeviceSynchronize();

            vector<long long> h_score(R);
            cudaMemcpy(h_score.data(), d_region_score,
                       sizeof(long long) * R,
                       cudaMemcpyDeviceToHost);

            vector<int> idx(R);
            iota(idx.begin(), idx.end(), 0);

            sort(idx.begin(), idx.end(),
                [&](int a, int b) {
                    return h_score[a] > h_score[b];
                });

            vector<int> h_hot_flag(R, 0);

            for (int i = 0; i < 16 && i < R; i++) {
                if (h_score[idx[i]] > 0)
                    h_hot_flag[idx[i]] = 1;
            }

            double avg_score = 0;
            for (int r = 0; r < R; r++) avg_score += h_score[r];
            avg_score /= R;

            const int base_block = 4;
            const int MAX_BLOCK_PER_REGION = 16;

            vector<int> h_blocks(R);

            for (int r = 0; r < R; r++) {
                double ratio = h_score[r] / (avg_score + 1e-9);

                int b = (int)(ratio * base_block);

                if (b < 1) b = 1;
                if (b > MAX_BLOCK_PER_REGION) b = MAX_BLOCK_PER_REGION;

                h_blocks[r] = b;
            }

            vector<int> h_offset(R);
            int total_blocks = 0;

            for (int r = 0; r < R; r++) {
                h_offset[r] = total_blocks;
                total_blocks += h_blocks[r];
            }

            vector<int> h_block_to_region(total_blocks);
            for (int r = 0; r < R; r++) {
                int start = h_offset[r];
                int end   = start + h_blocks[r];
                for (int b = start; b < end; b++) {
                    h_block_to_region[b] = r;
                }
            }

            cudaMemcpy(d_region_blocks, h_blocks.data(),
                       sizeof(int) * R, cudaMemcpyHostToDevice);

            cudaMemcpy(d_hot_region_flag, h_hot_flag.data(),
                       sizeof(int) * R, cudaMemcpyHostToDevice);

            cudaMemcpy(d_region_block_offset, h_offset.data(),
                       sizeof(int) * R, cudaMemcpyHostToDevice);

            cudaFree(d_block_to_region);
            cudaMalloc(&d_block_to_region, sizeof(int) * total_blocks);

            cudaMemcpy(d_block_to_region,
                       h_block_to_region.data(),
                       sizeof(int) * total_blocks,
                       cudaMemcpyHostToDevice);

            total_blocks_pr = total_blocks;
        } else {
            no_stag_window_rounds++;
            if (no_stag_window_rounds >= STAGNATE_ROUNDS) {
                GR_FREQ0 = max(32, GR_FREQ0 / 2);
                kernel_cycles_base = max(4, kernel_cycles_base / 2);
                kernel_cycles_this_iter = kernel_cycles_base;
                no_stag_window_rounds = 0;
            }
        }

        if (iter > ITER_MAX) {
            fprintf(stderr, "Exceeded ITER_MAX=%d, aborting loop.\n", ITER_MAX);
            break;
        }

        h_flag = (h_active > 0);

        // 3) 懒 + region-aware rm：每 RM_FREQ_dyn 轮执行一次，只在活跃 region 上
        int h_fixed_edges = 0;
        if (h_flag && (iter % RM_FREQ_dyn == 0)) {
            int hmin_init = V;
            int hmax_init = 0;
            cudaMemcpy(d_hmin_active, &hmin_init, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_hmax_active, &hmax_init, sizeof(int), cudaMemcpyHostToDevice);

            compute_active_height_range<<<numBlocks, numThreads>>>(
                V,
                source,
                sink,
                d_excess,
                d_height,
                d_hmin_active,
                d_hmax_active
            );
            cudaDeviceSynchronize();

            int h_min_a = 0, h_max_a = 0;
            cudaMemcpy(&h_min_a, d_hmin_active, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_max_a, d_hmax_active, sizeof(int), cudaMemcpyDeviceToHost);

            int h_min_rm = 0;
            int h_max_rm = V - 1;
            if (!(h_min_a > h_max_a || h_min_a == V)) {
                int margin = 2;
                h_min_rm = max(0, h_min_a - margin);
                h_max_rm = min(V - 1, h_max_a + margin);
            }

            int zero_fixed = 0;
            cudaMemcpy(d_fixed_edges, &zero_fixed, sizeof(int), cudaMemcpyHostToDevice);

            // region_active 清零
            cudaMemset(d_region_active, 0, sizeof(int) * R);
            // 标记每个 region 是否有活跃点
            int threads_region_active = 256; // 不同的线程块大小，避免与主PR kernel混淆
            compute_region_active<<<R, threads_region_active>>>(
                R,
                d_region_start,
                d_region_end,
                d_excess,
                d_region_active
            );
            cudaDeviceSynchronize();

            cudaEventRecord(ev_rm_start, 0);
            // 每个 block 一条 region
            staticMaxFlow_kernel_17_region<<<R, threads_region_active>>>(
                R,
                source, sink,
                V, E,
                d_region_start,
                d_region_end,
                d_region_active,
                d_meta, d_data,
                d_rev_residual_capacity,
                d_parallel,
                d_residual_capacity,
                d_excess,
                d_height,
                d_fixed_edges,
                h_min_rm,
                h_max_rm
            );
            cudaEventRecord(ev_rm_end, 0);
            cudaEventSynchronize(ev_rm_end);
            cudaEventElapsedTime(&ms_rm, ev_rm_start, ev_rm_end);

            cudaMemcpy(&h_fixed_edges, d_fixed_edges, sizeof(int), cudaMemcpyDeviceToHost);
            double frac_fix = (double)h_fixed_edges / (double)E;
            const double tau_rm = 1e-5;

            if (frac_fix < tau_rm) {
                RM_FREQ_dyn = min(RM_FREQ_dyn * 2, 1024);
            } else {
                RM_FREQ_dyn = max(RM_FREQ0, RM_FREQ_dyn / 2);
            }
        } 

        const char* mode_str;
        if (!use_active_list) {
            mode_str = "region-aware";
        } else if (use_path_push) {
            mode_str = "path-aware";
        } else {
            mode_str = "active-aware";
        }

        printf("Iteration %d: init_hl=%.3f ms, bfs=%.3f ms, "
               "push_relabel=%.3f ms (K=%d, mode=%s, deeppath=%d), remove_invalid=%.3f ms, "
               "active=%d, active_high=%d, fixed_edges=%d, GR_FREQ0=%d\n\n",
               iter, ms_init_hl, ms_bfs, ms_pr, kernel_cycles_this_iter,
               mode_str, MAX_PATH_LENGTH,
               ms_rm, h_active, h_active_high, h_fixed_edges, GR_FREQ0);


        iter++;
    } while (h_flag);

    flow_t max_flow = 0;
    cudaMemcpy(&max_flow, d_excess + sink, sizeof(flow_t), cudaMemcpyDeviceToHost);


    cudaEventDestroy(ev_k1_start);
    cudaEventDestroy(ev_k1_end);
    cudaEventDestroy(ev_k7_start);
    cudaEventDestroy(ev_k7_end);
    cudaEventDestroy(ev_init_hl_start);
    cudaEventDestroy(ev_init_hl_end);
    cudaEventDestroy(ev_bfs_start);
    cudaEventDestroy(ev_bfs_end);
    cudaEventDestroy(ev_pr_start);
    cudaEventDestroy(ev_pr_end);
    cudaEventDestroy(ev_rm_start);
    cudaEventDestroy(ev_rm_end);
    

    cudaDeviceSynchronize();
    auto q_end = chrono::high_resolution_clock::now();
    query_time_ms = chrono::duration_cast<chrono::milliseconds>(q_end - q_start).count();

    return max_flow;
}

// -------------------- 工具函数：创建目录 --------------------

void create_dir(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

// -------------------- 主函数 --------------------

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_dir> <output_dir>\n", argv[0]);
        return 1;
    }

    std::string input_dir  = argv[1];
    std::string output_dir = argv[2];

    create_dir(output_dir);

    cudaSetDevice(1); // Ensure correct device selection

    CSRGraph g_orig;
    long long build_time_ms = 0;
    vector<long long> sum_cap_out, sum_cap_in;
    vector<int> deg_out, deg_in;

    build_graph((input_dir + "/graph.txt").c_str(), g_orig, build_time_ms,
                sum_cap_out, sum_cap_in, deg_out, deg_in);

    int R = 128;  // region 数目（可调）
    

    vector<int> part;
    build_capacity_degree_partition_from_stats(
        g_orig.n, R,
        sum_cap_out, sum_cap_in,
        deg_out, deg_in,
        part
    );

    // 2) 根据 part[v] 做 region-ordered CSR 重排
    CSRGraph g;
    vector<int> new_id, old_id;
    vector<int> region_vertex_start, region_vertex_end;
    partition_and_reorder(g_orig, g, R, part,
                        new_id, old_id,
                        region_vertex_start, region_vertex_end);

    vector<int> region_id;
    build_region_id(g.n, region_vertex_start, region_vertex_end, region_id);


    // 3) 读取查询，并把 old_s/old_t 映射到 new 编号
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

    printf("Graph (reordered by Region): n=%d, m=%d, Regions=%d, Queries=%zu\n",
           g.n, g.m, R, qs.size());

    long long          total_query_time = 0;
    vector<flow_t>     flows;
    vector<long long>  times;

    for (size_t i = 0; i < qs.size(); ++i) {
        int s = qs[i].first;
        int t = qs[i].second;
        long long query_time = 0;

        int old_s = (s >= 0 && s < g.n) ? old_id[s] : -1;
        int old_t = (t >= 0 && t < g.n) ? old_id[t] : -1;

        fprintf(stderr, "Query %zu: s=%d, t=%d... ",
                i + 1, old_s + 1, old_t + 1);

        if (s < 0 || s >= g.n || t < 0 || t >= g.n || s == t) {
            flows.push_back(0);
            times.push_back(0);
            fprintf(stderr, "Invalid.\n");
        } else {
            flow_t flow = run_gpu(g, s, t, query_time,
                                  R,
                                  region_vertex_start,
                                  region_vertex_end,
                                  region_id);
            flows.push_back(flow);
            times.push_back(query_time);
            total_query_time += query_time;
            fprintf(stderr, "Flow=%lld, Time=%lld ms\n",
                    (long long)flow, query_time);
        }
    }

    // 输出结果
    std::string res_file = output_dir + "/res_mt.txt";
    FILE* out = fopen(res_file.c_str(), "w");
    if (!out) {
        fprintf(stderr, "Cannot open output file %s\n", res_file.c_str());
        return 1;
    }
    for (size_t qi = 0; qi < flows.size(); ++qi) {
        fprintf(out, "%lld %lld\n",
                (long long)flows[qi],
                (long long)times[qi]);
    }
    fclose(out);

    printf("Total time: %lld ms\n", total_query_time);

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
