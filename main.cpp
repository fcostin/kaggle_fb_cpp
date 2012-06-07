#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <vector>
#include <set>
using namespace std;

// #define set unordered_set

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


int max_node(vector<Edge> & edges) {
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

struct Graph {
    vector<int> begin;
    vector<int> end;
    vector<int> value;
};

Graph make_graph_v2(int size, vector<Edge> & edges) {
    Graph graph;
    graph.begin.reserve(size);
    graph.end.reserve(size);
    graph.value.reserve(edges.size());
    
    int i = 0;
    graph.begin.push_back(i);
    int prev_u = -1;
    vector<Edge>::const_iterator edge;
    for (edge = edges.begin(); edge != edges.end(); ++edge) {
        graph.value[i] = edge->v;
        if (edge->u != prev_u) {
            graph.end.push_back(i);
            graph.begin.push_back(i);
        }
        prev_u = edge->u;
    }
    graph.end.push_back(i);
    return graph;
}


vector<vector<int> > make_graph(int size, vector<Edge> & edges) {
    vector<vector<int> > graph;
    graph.resize(size);

    vector<Edge>::const_iterator i;
    for (i = edges.begin(); i != edges.end(); ++i) {
        assert(i->u < size);
        assert(0 <= i->u);
        graph[i->u].push_back(i->v);
    }
    return graph;
}

set<int> bfs(vector<vector<int> > & graph, int source, int depth) {
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
                vector<int>::const_iterator v;
                vector<int>::const_iterator v_beg = graph[*u].begin();
                vector<int>::const_iterator v_end = graph[*u].end();
                set<int>::const_iterator closed_end = closed.end();
                for (v = v_beg; v != v_end; ++v) {
                    if (closed.find(*v) != closed_end) {
                        continue;
                    }
                    next.insert(*v);
                }
            }
        }
        curr.swap(next);
        next.clear();
    }
    return closed;
}


int main(int narg, char **argv) {
    const char * csv_file_name = "../train.csv";
    printf("loading edges from \"%s\"\n", csv_file_name);
    vector<Edge> edges = load_edges(csv_file_name);
    printf("\tok\n");

    int m = max_node(edges);
    printf("max node is %d\n", m);

    printf("building graph\n");
    vector<vector<int> > graph = make_graph(m + 1, edges);
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
        if (i++ % 100 == 0) {
            printf("bfs [%d/%d]\n", i, n);
        }
        set<int> subset = bfs(graph, *src, depth);
    }
    printf("\tok\n");
    return 0;
}
