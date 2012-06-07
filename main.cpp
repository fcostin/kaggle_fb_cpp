#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <vector>
#include <set>
#include <map>
#include <algorithm>
using namespace std;

struct Edge {
    int u, v;
    Edge(int u, int v) : u(u), v(v) {}
};


vector<Edge> load_edges(const char *csv_file_name) {
    FILE *f = fopen(csv_file_name, "r");
    size_t line_size = 512;
    char *line = (char*) malloc(sizeof(char) * line_size);
    bool reading = true;
    vector<Edge> edges;
    while (reading) {
        int n_read = getline(&line, &line_size, f);
        reading = n_read > 0;
        if (reading) {
            int u, v;
            int tokens_parsed = sscanf(line, "%d,%d\n", &u, &v);
            if (tokens_parsed == 2) {
                // n.b. convert to 0-based indexing
                edges.push_back(Edge(u - 1, v - 1));
            }
        }
    }
    free(line);
    fclose(f);
    return edges;
}


vector<int> load_nodes(const char *csv_file_name) {
    FILE *f = fopen(csv_file_name, "r");
    size_t line_size = 512;
    char *line = (char*) malloc(sizeof(char) * line_size);
    bool reading = true;
    vector<int> nodes;
    while (reading) {
        int n_read = getline(&line, &line_size, f);
        reading = n_read > 0;
        if (reading) {
            int u;
            int tokens_parsed = sscanf(line, "%d\n", &u);
            if (tokens_parsed == 1) {
                // n.b. convert to 0-based indexing
                nodes.push_back(u - 1);
            }
        }
    }
    free(line);
    fclose(f);
    return nodes;
}


int max_node(const vector<Edge> & edges) {
    int m = 0;
    vector<Edge>::const_iterator i;
    for (i = edges.begin(); i != edges.end(); ++i) {
        if (i->u > m) {
            m = i->u;
        }
        if (i->v > m) {
            m = i->v;
        }
    }
    return m;
}

// store directed graph in csr-like format:
//  value[begin[u]:end[u]] : neighbours of node u
struct Graph {
    vector<int> begin;
    vector<int> end;
    vector<int> value;
};

Graph make_graph(int size, const vector<Edge> & edges) {
    Graph graph;
    graph.begin.reserve(size);
    graph.end.reserve(size);
    graph.value.reserve(edges.size());
    
    if (edges.size() > 0) {
        int i = 0;
        graph.begin.push_back(i);
        int prev_u = edges[0].u;
        vector<Edge>::const_iterator edge;
        for (edge = edges.begin(); edge != edges.end(); ++edge) {
            graph.value[i] = edge->v;
            while (edge->u > prev_u) {
                graph.end.push_back(i);
                graph.begin.push_back(i);
                ++prev_u;
            }
            ++i;
        }
        graph.end.push_back(i);
    }
    return graph;
}


set<int> bfs(const Graph & graph, int source, int depth) {
    set<int> closed;
    set<int> curr;
    curr.insert(source);
    set<int> next;

    for (int d = 0; d <= depth; ++d) {
        set<int>::const_iterator u;
        set<int>::const_iterator curr_beg = curr.begin();
        set<int>::const_iterator curr_end = curr.end();
        for (u = curr_beg; u != curr_end; ++u) {
            closed.insert(*u);
            if (d < depth) {
                int v_beg = graph.begin[*u];
                int v_end = graph.end[*u];
                set<int>::const_iterator closed_end = closed.end();
                for (int v = v_beg; v != v_end; ++v) {
                    if (closed.find(graph.value[v]) != closed_end) {
                        continue;
                    }
                    next.insert(graph.value[v]);
                }
            }
        }
        curr.swap(next);
        next.clear();
    }
    return closed;
}

struct CompressedIndex {
    map<int, int> compress;
    map<int, int> inflate;
};

CompressedIndex compress_indices(const set<int> & indices) {
    CompressedIndex result;
    set<int>::const_iterator i;
    int j = 0;
    for(i = indices.begin(); i != indices.end(); ++i) {
        result.compress.insert(make_pair(*i, j));
        result.inflate.insert(make_pair(j, *i));
        ++j;
    }
    assert(result.compress.size() == result.inflate.size());
    return result;
}

vector<Edge> restrict_graph(const Graph & graph, const set<int> & subset,
        const CompressedIndex & index) {

    vector<Edge> result;
    int n = index.inflate.size();
    for (int i = 0; i < n; ++i) {
        int u = index.inflate.find(i)->second;
        if (subset.find(u) == subset.end()) {
            continue;
        }
        int v_beg = graph.begin[u];
        int v_end = graph.end[u];
        for (int v = v_beg; v != v_end; ++v) {
            if (subset.find(graph.value[v]) == subset.end()) {
                continue;
            }
            int j = index.compress.find(graph.value[v])->second;
            result.push_back(Edge(i, j));
        }
    }
    return result;
}

vector<double> make_initial_vector(CompressedIndex & index, int source_node) {
    vector<double> v;
    v.resize(index.compress.size(), 0.0);
    map<int, int>::const_iterator i;
    i = index.compress.find(source_node);
    assert(i != index.compress.end());
    v[i->second] = 1.0;
    return v;
}

vector<double> graph_matvec(const Graph & graph, const vector<double> & x) {
    int n = min(x.size(), graph.begin.size());
    vector<double> result;
    result.resize(x.size(), 0.0);
    for (int i = 0; i < n; ++i) {
        if (fabs(x[i]) < 1.0e-16) {
            continue;
        }
        int j_beg = graph.begin[i];
        int j_end = graph.end[i];
        if (j_beg == j_end) {
            continue;
        }
        // distribute mass uniformly from node i to all neighbouring nodes k
        double mass = x[i] / (double)(j_end - j_beg);
        for (int j = j_beg; j != j_end; ++j) {
            int k = graph.value[j];
            result[k] += mass;
        }
    }
    return result;
}

vector<double> axpby(double a, const vector<double> & x, double b, const vector<double> & y) {
    int n = x.size();
    assert(n == y.size());
    vector<double> c;
    c.resize(n);
    for (int i = 0; i < n; ++i) {
        c[i] = a * x[i] + b * y[i];
    }
    return c;
}

double norm_l1(vector<double> & x) {
    double result = 0.0;
    vector<double>::const_iterator i;
    for(i = x.begin(); i != x.end(); ++i) {
        result += fabs(*i);
    }
    return result;
}

double metric_l1(vector<double> & x, vector<double> & y) {
    double result = 0.0;
    assert(x.size() == y.size());
    int n = x.size();
    for (int i = 0; i < n; ++i) {
        result += fabs(x[i] - y[i]);
    }
    return result;
}

vector<double> random_walk_with_restart(const Graph & graph, const vector<double> & v,
        double c, double eps) {
    
    assert(v.size() >= graph.end.size());
    assert(v.size() >= graph.begin.size());
    assert(v.size() > 0);
    vector<double> u, au, u_next;

    // initialise u with a copy of v
    u.resize(v.size());
    for(int i = 0; i < v.size(); ++i) {
        u[i] = v[i];
    }

    int itercount = 0;
    double error;
    do {
        au = graph_matvec(graph, u);
        u_next = axpby((1.0-c), au, c, v);
        swap(u, u_next);
        error = metric_l1(u, u_next);
        ++itercount;
        // printf("\t\terror = %e\n", error);
    } while(error > eps);
    printf("\trandom walk with restart converged after %d iters\n", itercount);

    return u;
}


int main(int narg, char **argv) {
    const char * csv_file_name = "../train.csv";
    printf("loading edges from \"%s\"\n", csv_file_name);
    vector<Edge> edges = load_edges(csv_file_name);
    printf("\tok\n");

    int m = max_node(edges);
    printf("max node is %d\n", m);

    printf("building graph\n");
    Graph graph = make_graph(m + 1, edges);
    printf("built graph\n");

    const char * node_file_name = "../test.csv";
    printf("loading nodes from \"%s\"\n", node_file_name);
    vector<int> test_nodes = load_nodes(node_file_name);
    printf("\tok\n");

    printf("bfs:\n");
    vector<int>::const_iterator src;
    int depth = 3;
    int i = 0;
    int n = test_nodes.size();
    for (src = test_nodes.begin(); src != test_nodes.end(); ++src) {
        set<int> subset = bfs(graph, *src, depth);
        int spam_period = 1;
        if (i++ % spam_period == 0) {
            printf("bfs [%d/%d]", i, n);
        }

        // 1. restrict graph to subset
        //  -- 1.1. define indexing scheme to give nodes in subset small dense indices
        CompressedIndex ci = compress_indices(subset);
        //  -- 1.2. make vector of edges as function of graph, subset and compressed indices
        vector<Edge> sub_edges = restrict_graph(graph, subset, ci);
        printf("\tsize %d subset has %d edges\n", subset.size(), sub_edges.size());
        //  -- 1.3. make subgraph from transformed edges
        Graph subgraph = make_graph(ci.inflate.size(), sub_edges);

        // 2. perform random walk with reset on the restricted graph
        vector<double> p_0 = make_initial_vector(ci, *src);

        double restart_prob = 0.1;
        double eps = 1.0e-6;
        vector<double> p_stationary = random_walk_with_restart(subgraph, p_0, restart_prob, eps);

        // 3. find the top-k nodes that aren't the source or direct neighbours, for k = 10
        //  -- 3.1. need to invert indexing scheme

        // 4. spit output somewhere
    }
    printf("\tok\n");
    return 0;
}

