up restart probability to 0.3 (perhaps makes more sense than 0.1 for truncated depth-3 domain)


some very helpful person on forums suggests adding in reverse edges
claims this gives at least 0.6something score (much better than me)

one way to adjust implementation to incorporate this:
    1. identify and add all missing reverse edges to yield a symmetrised graph
    2. use the symmetrised graph as normal for bfs searches and RWwR
    3. but the make_predictions function needs to be fixed to not
       exclude nodes that are only neighbours of the source by these
       artificial added edges
