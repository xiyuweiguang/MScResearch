#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr,
          "Usage: %s <test.txt> <graph_scc.txt.map> <test_inter.txt> <test_intra.txt>\n",
          argv[0]);
        return 1;
    }
    string test_file  = argv[1];
    string map_file   = argv[2];
    string out_inter  = argv[3];
    string out_intra  = argv[4];

    // 1) Read the map file: original vertex ID -> SCC component ID (1-based).
    FILE* mf = fopen(map_file.c_str(), "r");
    if (!mf) { fprintf(stderr, "Cannot open %s\n", map_file.c_str()); return 1; }

    vector<pair<int,int>> tmp;
    int orig_v, comp_id;
    int max_v = 0;
    while (fscanf(mf, "%d %d", &orig_v, &comp_id) == 2) {
        tmp.push_back({orig_v, comp_id});
        if (orig_v > max_v) max_v = orig_v;
    }
    fclose(mf);

    vector<int> comp_of_v(max_v + 1, -1);
    for (auto &p : tmp) comp_of_v[p.first] = p.second;

    // 2) Read test.txt and split queries according to whether comp(s) == comp(t).
    FILE* tf = fopen(test_file.c_str(), "r");
    if (!tf) { fprintf(stderr, "Cannot open %s\n", test_file.c_str()); return 1; }

    FILE* fi = fopen(out_inter.c_str(), "w");
    FILE* fa = fopen(out_intra.c_str(), "w"); // intra = same component
    if (!fi || !fa) {
        fprintf(stderr, "Cannot open output test files\n");
        return 1;
    }

    // Optional index file: records the original query ID, query class, and new line ID.
    FILE* idxf = fopen((out_inter + ".index").c_str(), "w");

    int u, v;
    int qid = 0;
    int inter_id = 0, intra_id = 0;
    while (fscanf(tf, "%d %d", &u, &v) == 2) {
        qid++;
        if (u <= 0 || u > max_v || v <= 0 || v > max_v) {
            // Out-of-range vertices are sent to the inter-component output.
            // The downstream solver can treat them as zero-flow or handle them separately.
            inter_id++;
            fprintf(fi, "%d %d\n", u, v);
            if (idxf) fprintf(idxf, "%d inter %d\n", qid, inter_id);
            continue;
        }
        int cu = comp_of_v[u];
        int cv = comp_of_v[v];
        if (cu <= 0 || cv <= 0) {
            inter_id++;
            fprintf(fi, "%d %d\n", u, v);
            if (idxf) fprintf(idxf, "%d inter %d\n", qid, inter_id);
        } else if (cu != cv) {
            // Cross-component query: it can be solved on the contracted SCC graph.
            inter_id++;
            fprintf(fi, "%d %d\n", cu, cv);  // Write component IDs directly (1-based).
            if (idxf) fprintf(idxf, "%d inter %d\n", qid, inter_id);
        } else {
            // Same-component query: solve it on the original graph and keep original IDs.
            intra_id++;
            fprintf(fa, "%d %d\n", u, v);
            if (idxf) fprintf(idxf, "%d intra %d\n", qid, intra_id);
        }
    }

    fclose(tf);
    fclose(fi);
    fclose(fa);
    if (idxf) fclose(idxf);

    fprintf(stderr, "[Split] total queries=%d, inter=%d, intra=%d\n",
            qid, inter_id, intra_id);
    return 0;
}
