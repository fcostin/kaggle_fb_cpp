#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <signal.h>


#include <vector>
#include <set>
#include <map>
#include <algorithm>
using namespace std;

// signal handling

bool abort_flag;

void main_sa_handler(int s) {
    fprintf(stderr, "\nCaught signal %d, aborting...\n", s);
    abort_flag = true;
}

void init_signal_handler() {
    // Set up signal handling before we try doing anything that will
    // take a long time. We want to intercept any SIGINT and then
    // set abort_flag to true.
    abort_flag = false;
    struct sigaction sig_interrupt_handler;
    sig_interrupt_handler.sa_handler = main_sa_handler;
    sigemptyset(&sig_interrupt_handler.sa_mask);
    sig_interrupt_handler.sa_flags = 0;
    sigaction(SIGINT, &sig_interrupt_handler, NULL);
}



struct Edge {
    int u, v;
    Edge(int u, int v) : u(u), v(v) {}
};

bool edge_compare(const Edge & a, const Edge & b) {
    if (a.u < b.u) {
        return true;
    } else if (a.u > b.u) {
        return false;
    } else {
        return (a.v < b.v);
    }
}


vector<Edge> load_edges(const char *csv_file_name) {
    FILE *f = fopen(csv_file_name, "r");
    size_t line_size = 512;
    char *line = (char*) malloc(sizeof(char) * line_size);
    bool reading = true;
    vector<Edge> edges;
    while (reading) {
        int n_read = (int)getline(&line, &line_size, f);
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
        int n_read = (int)getline(&line, &line_size, f);
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

CompressedIndex make_compressed_index(const set<int> & indices) {
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
    int n = (int)index.inflate.size();
    for (int i = 0; i < n; ++i) {
        int u = index.inflate.find(i)->second;
        if (subset.find(u) == subset.end()) {
            // XXX this code should never be executed, since
            // u in image of index.inflate <=> u in subset
            continue;
        }
        int v_beg = graph.begin[u];
        int v_end = graph.end[u];
        for (int v = v_beg; v != v_end; ++v) {
            // XXX this first test is redundant, since
            // w in subset <=> w in index.compress
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
    int n = min((int)x.size(), (int)graph.begin.size());
    vector<double> result;
    result.resize(x.size(), 0.0);
    for (int i = 0; i < n; ++i) {
        if (abs(x[i]) < 1.0e-16) {
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
    vector<double>::size_type n = x.size();
    assert(n == y.size());
    vector<double> c;
    c.resize(n);
    for (vector<double>::size_type i = 0; i < n; ++i) {
        c[i] = a * x[i] + b * y[i];
    }
    return c;
}

double norm_l1(vector<double> & x) {
    double result = 0.0;
    vector<double>::const_iterator i;
    for(i = x.begin(); i != x.end(); ++i) {
        result += abs(*i);
    }
    return result;
}

double metric_l1(vector<double> & x, vector<double> & y) {
    double result = 0.0;
    assert(x.size() == y.size());
    vector<double>::size_type n = x.size();
    for (vector<double>::size_type i = 0; i < n; ++i) {
        result += abs(x[i] - y[i]);
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
    for(int i = 0; (size_t)i < v.size(); ++i) {
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
    printf("\tcvgc took %d iters\n", itercount);

    return u;
}

bool argsort_compare(const pair<double, int> & a, const pair<double, int> & b) {
    return a.first < b.first;
}

vector<int> argsort(const vector<double> & a) {
    vector<pair<double, int> > ai;
    for (int i = 0; (size_t)i < a.size(); ++i) {
        ai.push_back(make_pair(a[i], i));
    }
    sort(ai.begin(), ai.end(), argsort_compare);
    vector<int> order;
    vector<pair<double, int> >::const_iterator i;
    for (i = ai.begin(); i != ai.end(); ++i) {
        order.push_back(i->second);
    }
    return order;
}

set<int> make_exclusion_set(const Graph & graph, int src) {
    set<int> result;
    result.insert(src);
    assert(0 <= src);
    // n.b. graph may not be full size if the nodes
    // with high indices have no outgoing edges.
    if ((size_t)src < graph.begin.size()) {
        int v_beg = graph.begin[src];
        int v_end = graph.end[src];
        for (int v = v_beg; v != v_end; ++v) {
            result.insert(graph.value[v]);
        }
    }
    return result;
}

vector<int> make_predictions(const set<int> & exclude, const vector<double> & p_stationary) {
    vector<int> order = argsort(p_stationary);
    vector<int>::const_reverse_iterator i;

    vector<int> predictions;
    // iterate through candidates in decreasing order of stationary probability
    for (i = order.rbegin(); i != order.rend(); ++i) {
        // ensure node i not in exclusion set
        if (exclude.find(*i) != exclude.end()) {
            continue;
        }
        predictions.push_back(*i);
        if (predictions.size() == 10) {
            break; // don't need more than 10 predictions
        }
    }
    return predictions;
}

vector<int> inflate_indices(const CompressedIndex & ci, const vector<int> & a) {
    vector<int> result;
    vector<int>::const_iterator i;
    for (i = a.begin(); i != a.end(); ++i) {
        map<int, int>::const_iterator j = ci.inflate.find(*i);
        assert(j != ci.inflate.end());
        result.push_back(j->second);
    }
    return result;
}

vector<int> compress_indices(const CompressedIndex & ci, const vector<int> & a) {
    vector<int> result;
    vector<int>::const_iterator i;
    for (i = a.begin(); i != a.end(); ++i) {
        map<int, int>::const_iterator j = ci.compress.find(*i);
        assert(j != ci.compress.end());
        result.push_back(j->second);
    }
    return result;
}

set<int> compress_indices_set(const CompressedIndex & ci, const set<int> & a) {
    set<int> result;
    set<int>::const_iterator i;
    for (i = a.begin(); i != a.end(); ++i) {
        map<int, int>::const_iterator j = ci.compress.find(*i);
        assert(j != ci.compress.end());
        result.insert(j->second);
    }
    return result;
}



vector<Edge> make_missing_reverse_edges(const vector<Edge> & edges, const Graph & graph) {
    vector<Edge> new_edges;
    
    vector<Edge>::const_iterator i;
    for (i = edges.begin(); i != edges.end(); ++i) {
        // handle special case where graph doesn't even have storage for
        // outgoing edges from node i->v
        if (graph.begin.size() <= (size_t)(i->v)) {
            new_edges.push_back(Edge(i->v, i->u));
            continue;
        }
        // linear search through outgoing edges from node i->v for
        // node i->u, if we don't find it, add an edge
        bool found = false;
        int j_beg = graph.begin[i->v];
        int j_end = graph.end[i->v];
        for (int j = j_beg; j != j_end; ++j) {
            if (graph.value[j] == i->u) {
                found = true;
                break;
            }
        }
        if (!found) {
            new_edges.push_back(Edge(i->v, i->u));
        }
    }

    return new_edges;
}

vector<Edge> extend_with_reverse_edges(const vector<Edge> & edges, const Graph & graph) {
    vector<Edge> result = make_missing_reverse_edges(edges, graph);
    // concat missing reverse edges and edges
    result.insert(result.end(), edges.begin(), edges.end());
    // sort vector of edges (treating each edge as pair and sorting
    // lexographically using standard integer ordering for node values)
    sort(result.begin(), result.end(), edge_compare);
    return result;
}


int main() {

    init_signal_handler();

    const char * csv_file_name = "../train.csv";
    printf("loading edges from \"%s\"\n", csv_file_name);
    vector<Edge> edges = load_edges(csv_file_name);
    printf("\tok\n");

    int m = max_node(edges);
    printf("max node is %d\n", m);

    printf("building graph\n");
    Graph graph = make_graph(m + 1, edges);
    printf("built graph\n");

    printf("building extended edges\n");
    vector<Edge> ext_edges = extend_with_reverse_edges(edges, graph);
    printf("ok\n");

    // n.b. adding edges does not change the nodes, so
    // the max node is still m.
    Graph ext_graph = make_graph(m + 1, ext_edges);

    const char * node_file_name = "../test.csv";
    printf("loading nodes from \"%s\"\n", node_file_name);
    vector<int> test_nodes = load_nodes(node_file_name);
    printf("\tok\n");

    vector<int>::const_iterator src;
    int depth = 3;
    int ticker = 0;
    int n = (int)test_nodes.size();

    FILE *f_out = fopen("predictions.csv", "w");
    fprintf(f_out, "source_node,destination_nodes\n");

    for (src = test_nodes.begin(); src != test_nodes.end(); ++src) {

        if (abort_flag) {
            fflush(f_out);
            break;
        }

        set<int> subset = bfs(ext_graph, *src, depth);
        printf("bfs [%d/%d]\n", ticker, n);

        // 1. restrict graph to subset
        //  -- 1.1. define indexing scheme to give nodes in subset small dense indices
        CompressedIndex ci = make_compressed_index(subset);
        //  -- 1.2. make vector of edges as function of graph, subset and compressed indices
        vector<Edge> sub_edges = restrict_graph(ext_graph, subset, ci);
        printf("\t%d nodes, %d edges\n", (int)subset.size(), (int)sub_edges.size());
        //  -- 1.3. make subgraph from transformed edges
        Graph subgraph = make_graph((int)ci.inflate.size(), sub_edges);

        // 2. perform random walk with reset on the restricted graph
        vector<double> p_0 = make_initial_vector(ci, *src);

        double restart_prob = 0.3;
        double eps = 1.0e-6;
        vector<double> p_stationary = random_walk_with_restart(subgraph, p_0, restart_prob, eps);

        // 3. find the top 10- nodes that aren't the source or direct neighbours

        // n.b. make exclusion set using graph, not ext_graph, so that we don't exclude
        // nodes that are only neighbours of src via artificial reverse edges. these
        // are the kind of nodes that we probably most likely want to include!
        set<int> base_unextended_exclude = make_exclusion_set(graph, *src);
        // n.b. translate excluded nodes to compressed indexing scheme
        set<int> exclude = compress_indices_set(ci, base_unextended_exclude);
        // make predictions
        vector<int> compressed_predictions = make_predictions(exclude, p_stationary);
        // then convert back from indexing scheme
        vector<int> predictions =  inflate_indices(ci, compressed_predictions);

        // 4. spit output somewhere. order of output matters.
        // write predictions to file in expected format
        // source_node,dest_nodes
        // where dest_nodes are separated by spaces
        fprintf(f_out, "%d,", (*src) + 1);
        for(int k = 0; (size_t)k < predictions.size(); ++k) {
            fprintf(f_out, "%d ", predictions[k] + 1);
        }
        fprintf(f_out, "\n");
        if (ticker++ % 1000 == 0) {
            fflush(f_out);
        }
    }
    fclose(f_out);
    printf("\tok\n");
    return 0;
}

