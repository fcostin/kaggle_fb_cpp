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
    double weight;
    Edge(int u, int v, double weight) : u(u), v(v), weight(weight) {}
};

bool edge_compare(const Edge & a, const Edge & b) {
    // lexical ordering on (u, v, weight)
    if (a.u < b.u) {
        return true;
    } else if (a.u > b.u) {
        return false;
    } else if (a.v < b.v) {
        return true;
    } else if (a.v > b.v) {
        return false;
    } else {
        return (a.weight < b.weight);
    }
}


vector<Edge> load_edges(const char *csv_file_name, double weight) {
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
                edges.push_back(Edge(u - 1, v - 1, weight));
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
    vector<double> weight;
};

Graph make_graph(int size, const vector<Edge> & edges) {
    Graph graph;
    graph.begin.reserve(size);
    graph.end.reserve(size);
    graph.value.reserve(edges.size());
    graph.weight.reserve(edges.size());
    
    if (edges.size() > 0) {
        int i = 0;
        graph.begin.push_back(i);
        int prev_u = edges[0].u;
        vector<Edge>::const_iterator edge;
        for (edge = edges.begin(); edge != edges.end(); ++edge) {
            graph.value[i] = edge->v;
            graph.weight[i] = edge->weight;
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
    vector<int> compress;
    vector<int> inflate;
};

CompressedIndex make_compressed_index(size_t size) {
    CompressedIndex result;
    // fill with -1s to mark missing items
    result.compress.resize(size + 1, -1);
    result.inflate.resize(size + 1, -1);
    return result;
}

void init_compressed_index(CompressedIndex & ci, const set<int> & indices) {
    set<int>::const_iterator i;
    int j = 0;
    for(i = indices.begin(); i != indices.end(); ++i) {
        ci.compress[*i] = j;
        ci.inflate[j] = *i;
        ++j;
    }
}

void clear_compressed_index(CompressedIndex & ci, const set<int> & indices) {
    set<int>::const_iterator i;
    int j = 0;
    for(i = indices.begin(); i != indices.end(); ++i) {
        ci.compress[*i] = -1;
        ci.inflate[j] = -1;
        ++j;
    }
}



vector<Edge> restrict_graph(int restricted_size,
        const Graph & graph,
        int error_node,
        const CompressedIndex & index) {

    assert(error_node >= 0);
    int error_node_k = index.compress[error_node];
    assert(error_node_k >= 0);

    vector<Edge> result;
    for (int i = 0; i < restricted_size; ++i) {
        int node_i = index.inflate[i];
        if (node_i == -1) {
            continue;
        }
        int j_beg = graph.begin[node_i];
        int j_end = graph.end[node_i];
        for (int j = j_beg; j != j_end; ++j) {
            int node_j = graph.value[j];
            double weight_j = graph.weight[j];
            int k = index.compress[node_j];
            // truncate edge if destination node not in subgraph
            // n.b. we may get many edges from a given node u
            // to the error node from this. this may or may not
            // lead to better accuracy! it certainly makes
            // the matvec operations a little more expensive.
            if (k == -1) {
                result.push_back(Edge(i, error_node_k, weight_j));
                continue;
            }
            result.push_back(Edge(i, k, weight_j));
        }
    }
    return result;
}

vector<double> make_initial_vector(const set<int> & subset, const CompressedIndex & index, int source_node) {
    vector<double> v;
    v.resize(subset.size(), 0.0);
    int i = index.compress[source_node];
    assert(i >= 0);
    v[i] = 1.0;
    return v;
}

void graph_matvec(const Graph & graph, const vector<double> & x,
        vector<double> & result) {
    int n = min((int)x.size(), (int)graph.begin.size());
    for (int i = 0; i < n; ++i) {
        if (x[i] < 1.0e-16) {
            continue;
        }
        // distribute mass uniformly from node i to all neighbouring nodes k
        int j_beg = graph.begin[i];
        int j_end = graph.end[i];
        double net_weight = 0.0;
        for (int j = j_beg; j < j_end; ++j) {
            net_weight += graph.weight[j];
        }
        double mass = x[i] / net_weight;
        for (int j = j_beg; j < j_end; ++j) {
            result[graph.value[j]] += mass * graph.weight[j];
        }
    }
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
    u.resize(v.size(), 0.0);
    for(int i = 0; (size_t)i < v.size(); ++i) {
        u[i] = v[i];
    }
    au.resize(v.size());

    int itercount = 0;
    double error;
    do {
        fill(au.begin(), au.end(), 0.0);
        graph_matvec(graph, u, au);
        u_next = axpby((1.0-c), au, c, v);
        swap(u, u_next);
        error = metric_l1(u, u_next);
        ++itercount;
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

set<int> make_exclusion_set(const Graph & graph, int src, int error_node) {
    set<int> result;
    result.insert(src);
    result.insert(error_node);
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
        int j = ci.inflate[*i];
        assert(j >= 0);
        result.push_back(j);
    }
    return result;
}

vector<int> compress_indices(const CompressedIndex & ci, const vector<int> & a) {
    vector<int> result;
    vector<int>::const_iterator i;
    for (i = a.begin(); i != a.end(); ++i) {
        int j = ci.compress[*i];
        assert(j >= 0);
        result.push_back(j);
    }
    return result;
}

set<int> compress_indices_set(const CompressedIndex & ci, const set<int> & a) {
    set<int> result;
    set<int>::const_iterator i;
    for (i = a.begin(); i != a.end(); ++i) {
        int j = ci.compress[*i];
        assert(j >= 0);
        result.insert(j);
    }
    return result;
}



vector<Edge> make_missing_reverse_edges(const vector<Edge> & edges, const Graph & graph,
        double weight) {
    vector<Edge> new_edges;
    
    vector<Edge>::const_iterator i;
    for (i = edges.begin(); i != edges.end(); ++i) {
        // handle special case where graph doesn't even have storage for
        // outgoing edges from node i->v
        if (graph.begin.size() <= (size_t)(i->v)) {
            new_edges.push_back(Edge(i->v, i->u, weight));
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
            new_edges.push_back(Edge(i->v, i->u, weight));
        }
    }

    return new_edges;
}

vector<Edge> extend_with_reverse_edges(const vector<Edge> & edges, const Graph & graph,
        double weight) {
    vector<Edge> result = make_missing_reverse_edges(edges, graph, weight);
    // concat missing reverse edges and edges
    result.insert(result.end(), edges.begin(), edges.end());
    // sort vector of edges (treating each edge as pair and sorting
    // lexographically using standard integer ordering for node values)
    sort(result.begin(), result.end(), edge_compare);
    return result;
}

void print_graph_info(const Graph & graph) {
    int n = (int)graph.begin.size();
    int sup = 0;
    for (int i = 0; i < n; ++i) {
        int j_beg = graph.begin[i];
        int j_end = graph.end[i];
        int n_neighbours = j_end - j_beg;
        sup = (n_neighbours > sup) ? n_neighbours : sup;
    }
    printf("max num neighbours : %d\n", sup);
}

struct Params {
    char * edges_file_name;
    char * nodes_file_name;
    char * predictions_file_name;
    int bfs_depth;
    double restart_prob;
    double eps;
    double forward_weight;
    double reverse_weight;
};

void print_usage_and_exit() {
    fprintf(stderr, (
        "usage: edges.csv nodes.csv predictions.csv depth restart_prob eps"
        "forward_weight reverse_weight\n"));
    exit(1);
}

Params parse_params(int narg, char **argv) {
    if (narg != 9) {
        print_usage_and_exit();
    }
    Params params;
    params.edges_file_name = argv[1];
    params.nodes_file_name = argv[2];
    params.predictions_file_name = argv[3];
    if (sscanf(argv[4], "%d", &(params.bfs_depth)) != 1) {
        print_usage_and_exit();
    }
    if (sscanf(argv[5], "%lf", &(params.restart_prob)) != 1) {
        print_usage_and_exit();
    }
    if (sscanf(argv[6], "%lf", &(params.eps)) != 1) {
        print_usage_and_exit();
    }
    if (sscanf(argv[7], "%lf", &(params.forward_weight)) != 1) {
        print_usage_and_exit();
    }
    if (sscanf(argv[8], "%lf", &(params.reverse_weight)) != 1) {
        print_usage_and_exit();
    }
    printf("\tedges_file_name := \"%s\"\n", params.edges_file_name);
    printf("\tnodes_file_name := \"%s\"\n", params.nodes_file_name);
    printf("\tpredictions_file_name := \"%s\"\n", params.predictions_file_name);
    printf("\tbfs_depth := %d\n", params.bfs_depth);
    printf("\trestart_prob := %e\n", params.restart_prob);
    printf("\teps := %e\n", params.eps);
    printf("\tforward_weight := %e\n", params.forward_weight);
    printf("\treverse_weight := %e\n", params.reverse_weight);
    return params;
}


int main(int narg, char **argv) {

    // parse command line arguments, exit with error message on failure
    Params params = parse_params(narg, argv);

    // initialise signal handler, useful for gracefully exiting
    // after hitting ctrl+C so we still get profiling information
    init_signal_handler();

    printf("loading edges from \"%s\"\n", params.edges_file_name);
    vector<Edge> edges = load_edges(params.edges_file_name, params.forward_weight);
    printf("\tok\n");

    int m = max_node(edges);
    printf("max node is %d\n", m);
    // artificially increase the number of nodes by 1 to make room for
    // a sink node to track trunctation error
    int error_node = m;
    m += 1;

    printf("building graph\n");
    Graph graph = make_graph(m + 1, edges);
    printf("built graph\n");

    printf("building extended edges\n");
    vector<Edge> ext_edges = extend_with_reverse_edges(edges, graph, params.reverse_weight);
    printf("ok\n");

    // n.b. adding edges does not change the nodes, so
    // the max node is still m.
    Graph ext_graph = make_graph(m + 1, ext_edges);

    print_graph_info(ext_graph);

    printf("loading nodes from \"%s\"\n", params.nodes_file_name);
    vector<int> test_nodes = load_nodes(params.nodes_file_name);
    printf("\tok\n");

    vector<int>::const_iterator src;
    int depth = params.bfs_depth;
    double restart_prob = params.restart_prob;
    double eps = params.eps;
    int ticker = 0;
    int n = (int)test_nodes.size();

    // allocate compressed index (implemented as pair of huge
    // lookup tables) to hold reindexing scheme
    CompressedIndex ci = make_compressed_index(m + 1);

    FILE *f_out = fopen(params.predictions_file_name, "w");
    fprintf(f_out, "source_node,destination_nodes\n");

    for (src = test_nodes.begin(); src != test_nodes.end(); ++src) {

        if (abort_flag) {
            fflush(f_out);
            break;
        }

        printf("bfs [%d/%d]\n", ticker, n);

        set<int> subset = bfs(ext_graph, *src, depth);
        // add error node to track truncation error
        subset.insert(error_node);

        // 1. restrict graph to subset
        //  -- 1.1. define indexing scheme to give nodes in subset small dense indices
        init_compressed_index(ci, subset);
        //  -- 1.2. make vector of edges as function of graph, subset and compressed indices
        vector<Edge> sub_edges = restrict_graph((int)subset.size(), ext_graph, error_node, ci);
        printf("\t%d nodes, %d edges\n", (int)subset.size(), (int)sub_edges.size());
        //  -- 1.3. make subgraph from transformed edges
        Graph subgraph = make_graph((int)subset.size(), sub_edges);

        // 2. perform random walk with reset on the restricted graph
        vector<double> p_0 = make_initial_vector(subset, ci, *src);

        vector<double> p_stationary = random_walk_with_restart(subgraph, p_0, restart_prob, eps);

        double mass_error = p_stationary[ci.compress[error_node]];
        printf("\tmass_error %3.1f %%\n", mass_error * 100.0);

        // 3. find the top 10- nodes that aren't the source or direct neighbours

        // n.b. make exclusion set using graph, not ext_graph, so that we don't exclude
        // nodes that are only neighbours of src via artificial reverse edges. these
        // are the kind of nodes that we probably most likely want to include!
        set<int> base_unextended_exclude = make_exclusion_set(graph, *src, error_node);
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

        // 5. clean up
        clear_compressed_index(ci, subset);
    }
    fclose(f_out);
    printf("\tok\n");
    return 0;
}

