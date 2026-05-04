#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <orig_test.txt> <graph_scc.txt.map> <out_test_scc.txt>\n",
                argv[0]);
        return 1;
    }
    string test_file = argv[1];
    string map_file  = argv[2];
    string out_file  = argv[3];

    // 1) Read the map file: original vertex ID -> SCC component ID (1-based).
    FILE* mf = fopen(map_file.c_str(), "r");
    if (!mf) {
        fprintf(stderr, "Cannot open %s\n", map_file.c_str());
        return 1;
    }
    vector<int> comp_of_v; // 1-based indexing
    int orig_v, comp_id;
    int max_v = 0;
    vector<pair<int,int>> tmp;
    while (fscanf(mf, "%d %d", &orig_v, &comp_id) == 2) {
        tmp.push_back({orig_v, comp_id});
        if (orig_v > max_v) max_v = orig_v;
    }
    fclose(mf);
    comp_of_v.assign(max_v + 1, -1);
    for (auto &p : tmp) {
        comp_of_v[p.first] = p.second;  // Both IDs are 1-based.
    }

    // 2) Read the original test.txt and map each query endpoint to its SCC component ID.
    FILE* tf = fopen(test_file.c_str(), "r");
    if (!tf) {
        fprintf(stderr, "Cannot open %s\n", test_file.c_str());
        return 1;
    }
    FILE* of = fopen(out_file.c_str(), "w");
    if (!of) {
        fprintf(stderr, "Cannot open %s for write\n", out_file.c_str());
        return 1;
    }

    int u, v;
    while (fscanf(tf, "%d %d", &u, &v) == 2) {
        if (u <= 0 || u > max_v || v <= 0 || v > max_v) {
            // Out-of-range vertices are written as 0 0.
            fprintf(of, "0 0\n");
            continue;
        }
        int cu = comp_of_v[u]; // Component ID (1-based).
        int cv = comp_of_v[v];
        if (cu <= 0 || cv <= 0) {
            fprintf(of, "0 0\n");
        } else {
            // Output component IDs directly; they remain 1-based.
            fprintf(of, "%d %d\n", cu, cv);
        }
    }

    fclose(tf);
    fclose(of);
    fprintf(stderr, "[Convert] done, written to %s\n", out_file.c_str());
    return 0;
}
