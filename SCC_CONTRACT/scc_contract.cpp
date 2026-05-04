#include <bits/stdc++.h>
using namespace std;
using ll = long long;

struct Edge {
    int to;
    ll  cap;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_graph.txt> <output_graph_scc.txt>\n", argv[0]);
        return 1;
    }
    string in_file  = argv[1];
    string out_file = argv[2];

    // 1. Read the original graph. Only forward edges are stored here.
    FILE* f = fopen(in_file.c_str(), "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", in_file.c_str());
        return 1;
    }
    int n, m;
    if (fscanf(f, "%d %d", &n, &m) != 2) {
        fprintf(stderr, "Invalid header in %s\n", in_file.c_str());
        return 1;
    }
    vector<vector<Edge>> g(n), gr(n); // g: forward graph, gr: reverse graph
    int  u, v;
    long long c;
    for (int i = 0; i < m; ++i) {
        if (fscanf(f, "%d %d %lld", &u, &v, &c) != 3) break;
        --u; --v;
        if (u < 0 || u >= n || v < 0 || v >= n) continue;
        if (u == v) continue;
        if (c <= 0) continue;
        g[u].push_back({v, c});
        gr[v].push_back({u, c});  // The reverse graph only needs topology; capacity is unused.
    }
    fclose(f);
    fprintf(stderr, "[SCC] Graph loaded: n=%d\n", n);

    // 2. Kosaraju pass 1: iterative DFS to produce the finishing-order list.
    vector<char> used(n, 0);
    vector<int> order;
    order.reserve(n);

    for (int s = 0; s < n; ++s) {
        if (used[s]) continue;

        // Explicit stack that simulates recursion: (vertex, next_edge_index, state).
        // state = 0: children are still being scanned; state = 1: children are done, append to order.
        struct Frame { int v; int it; char state; };
        vector<Frame> st;

        st.push_back({s, 0, 0});
        used[s] = 1;

        while (!st.empty()) {
            Frame &fr = st.back();
            int vtx = fr.v;

            if (fr.state == 0) {
                // Scan outgoing neighbors.
                if (fr.it < (int)g[vtx].size()) {
                    int to = g[vtx][fr.it].to;
                    fr.it++;
                    if (!used[to]) {
                        used[to] = 1;
                        st.push_back({to, 0, 0});
                    }
                } else {
                    // All children have been processed. Switch state and pop on the next step.
                    fr.state = 1;
                }
            } else {
                // Equivalent to the recursive DFS exit time.
                order.push_back(vtx);
                st.pop_back();
            }
        }
    }

    // 3. Kosaraju pass 2: assign SCC IDs on the reverse graph in reverse finishing order.
    vector<int> comp_of_v(n, -1);
    int C = 0;
    for (int i = (int)order.size() - 1; i >= 0; --i) {
        int s0 = order[i];
        if (comp_of_v[s0] != -1) continue;

        // Traverse gr from s0 and mark all vertices in the same SCC.
        vector<int> st;
        st.push_back(s0);
        comp_of_v[s0] = C;
        while (!st.empty()) {
            int vtx = st.back(); st.pop_back();
            for (auto &e : gr[vtx]) {
                int to = e.to;
                if (comp_of_v[to] == -1) {
                    comp_of_v[to] = C;
                    st.push_back(to);
                }
            }
        }
        C++;
    }

    fprintf(stderr, "[SCC] found components: C=%d\n", C);

    // 4. Merge capacities between SCCs: (comp_u, comp_v) -> summed capacity.
    struct EdgeKey {
        int u, v;
        bool operator<(const EdgeKey& o) const {
            return (u < o.u) || (u == o.u && v < o.v);
        }
    };
    map<EdgeKey, ll> agg;

    for (int x = 0; x < n; ++x) {
        int cx = comp_of_v[x];
        for (auto &e : g[x]) {
            int y   = e.to;
            ll  cap = e.cap;
            int cy  = comp_of_v[y];
            if (cx == cy) continue; // Omit edges inside the same component.
            EdgeKey key{cx, cy};
            agg[key] += cap;        // Sum capacities of parallel component-level edges.
        }
    }

    int n_scc = C;
    int m_scc = (int)agg.size();
    fprintf(stderr, "[SCC] contracted graph: n'=%d, m'=%d\n", n_scc, m_scc);

    // 5. Write the contracted SCC graph.
    FILE* out = fopen(out_file.c_str(), "w");
    if (!out) {
        fprintf(stderr, "Cannot open %s for write\n", out_file.c_str());
        return 1;
    }
    fprintf(out, "%d %d\n", n_scc, m_scc);
    for (auto &kv : agg) {
        int cu = kv.first.u;
        int cv = kv.first.v;
        ll  cap = kv.second;
        fprintf(out, "%d %d %lld\n", cu + 1, cv + 1, cap);
    }
    fclose(out);

    // 6. Optional output: original vertex ID -> SCC component ID mapping.
    string map_file = out_file + ".map";
    FILE* mapf = fopen(map_file.c_str(), "w");
    if (mapf) {
        for (int i = 0; i < n; ++i) {
            fprintf(mapf, "%d %d\n", i + 1, comp_of_v[i] + 1);
        }
        fclose(mapf);
        fprintf(stderr, "[SCC] mapping written to %s\n", map_file.c_str());
    }

    fprintf(stderr, "[SCC] Done.\n");
    return 0;
}
