# RAPTOR Activity Flowcharts

> Decision logic and branching for key operations

## 1. build_from_text() Flowchart

Main entry point for tree construction (tree_builder.py:260-295):

```
                          +------------------+
                          |     START        |
                          | build_from_text  |
                          |   (text: str)    |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | split_text()     |
                          |                  |
                          | text -> chunks[] |
                          | (max_tokens=100) |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | use_multi-       |
                          | threading?       |
                          +--------+---------+
                                   |
                      +------------+------------+
                      |                         |
                   TRUE                      FALSE
                      |                         |
                      v                         v
            +------------------+      +------------------+
            | multithreaded_   |      | Sequential loop  |
            | create_leaf_     |      |                  |
            | nodes(chunks)    |      | for idx, text in |
            |                  |      |   enumerate():   |
            | ThreadPoolExec.  |      |   create_node()  |
            +--------+---------+      +--------+---------+
                      |                         |
                      +------------+------------+
                                   |
                                   v
                          +------------------+
                          | leaf_nodes =     |
                          |   Dict[int,Node] |
                          |                  |
                          | layer_to_nodes   |
                          |   [0] = leaves   |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | all_nodes =      |
                          | deepcopy(leaf_   |
                          |   nodes)         |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | construct_tree() |
                          |                  |
                          | (See next        |
                          |  flowchart)      |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | Tree(            |
                          |   all_nodes,     |
                          |   root_nodes,    |
                          |   leaf_nodes,    |
                          |   num_layers,    |
                          |   layer_to_nodes)|
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          |    RETURN Tree   |
                          +------------------+
```

## 2. construct_tree() Loop Flowchart

Clustering-based tree construction (cluster_tree_builder.py:55-151):

```
                          +------------------+
                          |     START        |
                          | construct_tree   |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | next_node_index  |
                          |   = len(all_     |
                          |     tree_nodes)  |
                          +--------+---------+
                                   |
                                   v
            +-------->   +------------------+
            |            | FOR layer = 0    |
            |            | to num_layers-1  |
            |            +--------+---------+
            |                     |
            |                     v
            |            +------------------+
            |            | new_level_nodes  |
            |            |   = {}           |
            |            +--------+---------+
            |                     |
            |                     v
            |            +------------------+
            |            | node_list =      |
            |            | get_node_list(   |
            |            |   current_level) |
            |            +--------+---------+
            |                     |
            |                     v
            |            +------------------+
            |            | len(node_list)   |
            |            |   <= reduction_  |
            |            |   dimension + 1? |
            |            +--------+---------+
            |                     |
            |            YES      |      NO
            |              +------+------+
            |              |             |
            |              v             |
            |     +------------------+   |
            |     | num_layers =     |   |
            |     |   layer          |   |
            |     | LOG: "Stopping"  |   |
            |     +--------+---------+   |
            |              |             |
            |              v             |
            |     +------------------+   |
            |     |   BREAK LOOP     |   |
            |     +--------+---------+   |
            |              |             |
            |              |             v
            |              |    +------------------+
            |              |    | perform_         |
            |              |    | clustering()     |
            |              |    |                  |
            |              |    | (RAPTOR_Cluster) |
            |              |    +--------+---------+
            |              |             |
            |              |             v
            |              |    +------------------+
            |              |    | clusters =       |
            |              |    | [[Node,..],      |
            |              |    |  [Node,..],...]  |
            |              |    +--------+---------+
            |              |             |
            |              |             v
            |              |    +------------------+
            |              |    | use_multi-       |
            |              |    | threading?       |
            |              |    +--------+---------+
            |              |             |
            |              |    +--------+--------+
            |              |    |                 |
            |              | TRUE              FALSE
            |              |    |                 |
            |              |    v                 v
            |              | +---------+   +----------+
            |              | |ThreadPool|   |Sequential|
            |              | |Executor  |   |  loop    |
            |              | +---------+   +----------+
            |              |    |                 |
            |              |    +--------+--------+
            |              |             |
            |              |             v
            |              |  +--------------------+
            |              |  | FOR each cluster:  |
            |              |  +--------+-----------+
            |              |           |
            |              |           v
            |              |  +------------------+
            |              |  | node_texts =     |
            |              |  |   get_text(      |
            |              |  |     cluster)     |
            |              |  +--------+---------+
            |              |           |
            |              |           v
            |              |  +------------------+
            |              |  | summarized =     |
            |              |  |   self.summarize |
            |              |  |   (node_texts,   |
            |              |  |    max_tokens)   |
            |              |  +--------+---------+
            |              |           |
            |              |           v
            |              |  +------------------+
            |              |  | create_node(     |
            |              |  |   next_index,    |
            |              |  |   summarized,    |
            |              |  |   {children_idx})|
            |              |  +--------+---------+
            |              |           |
            |              |           v
            |              |  +------------------+
            |              |  | new_level_nodes  |
            |              |  |   [next_index]   |
            |              |  |   = new_node     |
            |              |  |                  |
            |              |  | next_index++     |
            |              |  +--------+---------+
            |              |           |
            |              |           v
            |              |  +------------------+
            |              |  | [End cluster     |
            |              |  |  loop]           |
            |              |  +--------+---------+
            |                          |
            |                          v
            |               +------------------+
            |               | layer_to_nodes   |
            |               |   [layer+1] =    |
            |               |   new_level_nodes|
            |               +--------+---------+
            |                          |
            |                          v
            |               +------------------+
            |               | current_level =  |
            |               |   new_level_nodes|
            |               +--------+---------+
            |                          |
            |                          v
            |               +------------------+
            |               | all_tree_nodes   |
            |               |   .update(       |
            |               |   new_level)     |
            |               +--------+---------+
            |                          |
            +<-------------------------+
            (next layer iteration)
                                       |
                                       v
                          +------------------+
                          | RETURN           |
                          | current_level_   |
                          |   nodes          |
                          | (root_nodes)     |
                          +------------------+
```

## 3. retrieve() Flowchart

Main retrieval entry point (tree_retriever.py:252-327):

```
                          +------------------+
                          |     START        |
                          |   retrieve()     |
                          |                  |
                          | (query, top_k,   |
                          |  max_tokens,     |
                          |  collapse_tree,  |
                          |  start_layer,    |
                          |  num_layers)     |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | Validate: query  |
                          |   is string?     |
                          +--------+---------+
                                   |
                           YES     |     NO
                             +-----+-----+
                             |           |
                             |           v
                             |  +------------------+
                             |  | RAISE ValueError |
                             |  | "query must be   |
                             |  |  a string"       |
                             |  +------------------+
                             |
                             v
                          +------------------+
                          | Validate:        |
                          | max_tokens >= 1? |
                          +--------+---------+
                                   |
                           YES     |     NO
                             +-----+-----+
                             |           |
                             |           v
                             |  +------------------+
                             |  | RAISE ValueError |
                             |  +------------------+
                             |
                             v
                          +------------------+
                          | Set defaults:    |
                          | start_layer =    |
                          |   self.start_    |
                          |   layer          |
                          | num_layers =     |
                          |   self.num_layers|
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | Validate:        |
                          | start_layer in   |
                          | [0, tree.layers]?|
                          +--------+---------+
                                   |
                           YES     |     NO
                             +-----+-----+
                             |           v
                             |  +------------------+
                             |  | RAISE ValueError |
                             |  +------------------+
                             v
                          +------------------+
                          | Validate:        |
                          | num_layers <=    |
                          |   start_layer+1? |
                          +--------+---------+
                                   |
                           YES     |     NO
                             +-----+-----+
                             |           v
                             |  +------------------+
                             |  | RAISE ValueError |
                             |  +------------------+
                             v
                          +------------------+
                          | collapse_tree    |
                          |   == True?       |
                          +--------+---------+
                                   |
               +-------------------+-------------------+
               |                                       |
            TRUE                                    FALSE
               |                                       |
               v                                       v
    +------------------+                    +------------------+
    | retrieve_info_   |                    | layer_nodes =    |
    | collapse_tree(   |                    |   tree.layer_to_ |
    |   query,         |                    |   nodes[start_   |
    |   top_k,         |                    |     layer]       |
    |   max_tokens)    |                    +--------+---------+
    +--------+---------+                             |
             |                                       v
             |                              +------------------+
             |                              | retrieve_        |
             |                              | information(     |
             |                              |   layer_nodes,   |
             |                              |   query,         |
             |                              |   num_layers)    |
             |                              +--------+---------+
             |                                       |
             +-------------------+-------------------+
                                 |
                                 v
                        +------------------+
                        | selected_nodes,  |
                        | context          |
                        +--------+---------+
                                 |
                                 v
                        +------------------+
                        | return_layer_    |
                        | information?     |
                        +--------+---------+
                                 |
               +----------------+----------------+
               |                                 |
            TRUE                              FALSE
               |                                 |
               v                                 v
    +------------------+              +------------------+
    | Build layer_info |              | RETURN context   |
    | for each node:   |              |                  |
    |   {node_index,   |              +------------------+
    |    layer_number} |
    +--------+---------+
             |
             v
    +------------------+
    | RETURN           |
    | (context,        |
    |  layer_info)     |
    +------------------+
```

## 4. Clustering Flowchart (perform_clustering)

Detailed clustering logic (cluster_utils.py:69-123):

```
                          +------------------+
                          |     START        |
                          | perform_         |
                          | clustering()     |
                          |                  |
                          | (embeddings,     |
                          |  dim, threshold) |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          |  GLOBAL REDUCE   |
                          |                  |
                          | n_neighbors =    |
                          |   sqrt(n-1)      |
                          | dim' = min(dim,  |
                          |         len-2)   |
                          | UMAP.fit_        |
                          |   transform()    |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | GLOBAL CLUSTER   |
                          |                  |
                          | GMM_cluster()    |
                          | optimal via BIC  |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | global_clusters, |
                          | n_global_clusters|
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | all_local_       |
                          | clusters = [[]   |
                          |   for each emb]  |
                          | total_clusters=0 |
                          +--------+---------+
                                   |
                                   v
         +-------->      +------------------+
         |               | FOR i in range   |
         |               | (n_global_       |
         |               |  clusters):      |
         |               +--------+---------+
         |                        |
         |                        v
         |               +------------------+
         |               | Get embeddings   |
         |               | in global        |
         |               | cluster i        |
         |               +--------+---------+
         |                        |
         |                        v
         |               +------------------+
         |               | len(cluster_     |
         |               |   embeddings)    |
         |               |   == 0?          |
         |               +--------+---------+
         |                        |
         |                YES     |     NO
         |                  +-----+-----+
         |                  |           |
         |                  v           |
         |         +----------+         |
         |         | CONTINUE |         |
         |         | (skip)   |         |
         |         +----------+         |
         |                              |
         |                              v
         |               +------------------+
         |               | len(cluster_     |
         |               |   embeddings)    |
         |               |   <= dim + 1?    |
         |               +--------+---------+
         |                        |
         |                YES     |     NO
         |                  +-----+-----+
         |                  |           |
         |                  v           v
         |      +----------------+ +------------------+
         |      | Single local   | | LOCAL REDUCE     |
         |      | cluster:       | |                  |
         |      | label = [0]    | | UMAP local       |
         |      | n_local = 1    | | dim reduction    |
         |      +-------+--------+ +--------+---------+
         |              |                   |
         |              |                   v
         |              |          +------------------+
         |              |          | LOCAL CLUSTER    |
         |              |          |                  |
         |              |          | GMM_cluster()    |
         |              |          | on local reduced |
         |              |          +--------+---------+
         |              |                   |
         |              +--------+----------+
         |                       |
         |                       v
         |               +------------------+
         |               | FOR j in range   |
         |               | (n_local_        |
         |               |  clusters):      |
         |               +--------+---------+
         |                        |
         |                        v
         |               +------------------+
         |               | Find indices in  |
         |               | original array   |
         |               | where embedding  |
         |               | matches local    |
         |               | cluster j        |
         |               +--------+---------+
         |                        |
         |                        v
         |               +------------------+
         |               | FOR each idx:    |
         |               |   all_local_     |
         |               |   clusters[idx]  |
         |               |   .append(j +    |
         |               |     total_       |
         |               |     clusters)    |
         |               +--------+---------+
         |                        |
         |                        v
         |               +------------------+
         |               | [End j loop]     |
         |               +--------+---------+
         |                        |
         |                        v
         |               +------------------+
         |               | total_clusters   |
         |               |   += n_local_    |
         |               |   clusters       |
         |               +--------+---------+
         |                        |
         +<-----------------------+
         (next i iteration)
                                  |
                                  v
                          +------------------+
                          | RETURN           |
                          | all_local_       |
                          | clusters         |
                          | (label arrays    |
                          |  for each emb)   |
                          +------------------+
```

## 5. RAPTOR_Clustering.perform_clustering() Flowchart

Node-level clustering with size validation (cluster_utils.py:132-185):

```
                          +------------------+
                          |     START        |
                          | RAPTOR_Clustering|
                          | .perform_        |
                          | clustering()     |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | Extract          |
                          | embeddings from  |
                          | nodes using      |
                          | embedding_model_ |
                          | name             |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | Call perform_    |
                          | clustering()     |
                          | (core function)  |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | clusters =       |
                          | [label_arrays]   |
                          | (one per node)   |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | node_clusters=[] |
                          +--------+---------+
                                   |
                                   v
         +-------->      +------------------+
         |               | FOR label in     |
         |               | unique(concat(   |
         |               |   clusters)):    |
         |               +--------+---------+
         |                        |
         |                        v
         |               +------------------+
         |               | indices = [i for |
         |               |   i, c in enum(  |
         |               |     clusters)    |
         |               |   if label in c] |
         |               +--------+---------+
         |                        |
         |                        v
         |               +------------------+
         |               | cluster_nodes =  |
         |               |   [nodes[i] for  |
         |               |    i in indices] |
         |               +--------+---------+
         |                        |
         |                        v
         |               +------------------+
         |               | len(cluster_     |
         |               |   nodes) == 1?   |
         |               +--------+---------+
         |                        |
         |                 YES    |    NO
         |                   +----+----+
         |                   |         |
         |                   v         |
         |          +------------+     |
         |          | BASE CASE  |     |
         |          | Append     |     |
         |          | single-node|     |
         |          | cluster    |     |
         |          | CONTINUE   |     |
         |          +------------+     |
         |                             |
         |                             v
         |               +------------------+
         |               | total_length =   |
         |               |   sum(len(tok(   |
         |               |     node.text))  |
         |               |   for node)      |
         |               +--------+---------+
         |                        |
         |                        v
         |               +------------------+
         |               | total_length >   |
         |               | max_length_in_   |
         |               |   cluster?       |
         |               | (default: 3500)  |
         |               +--------+---------+
         |                        |
         |                YES     |     NO
         |                  +-----+-----+
         |                  |           |
         |                  v           v
         |      +----------------+ +------------------+
         |      | RECLUSTER      | | ACCEPT           |
         |      |                | |                  |
         |      | LOG: "re-      | | node_clusters    |
         |      |  clustering"   | |   .append(       |
         |      |                | |   cluster_nodes) |
         |      | recursive call | +--------+---------+
         |      | with cluster_  |          |
         |      |   nodes only   |          |
         |      +-------+--------+          |
         |              |                   |
         |              v                   |
         |      +----------------+          |
         |      | Extend node_   |          |
         |      | clusters with  |          |
         |      | recursive      |          |
         |      | results        |          |
         |      +-------+--------+          |
         |              |                   |
         |              +--------+----------+
         |                       |
         +<----------------------+
         (next label iteration)
                                 |
                                 v
                        +------------------+
                        | RETURN           |
                        | node_clusters    |
                        | [[Node,...],     |
                        |  [Node,...],     |
                        |  ...]            |
                        +------------------+
```

## 6. get_optimal_clusters() - BIC Selection Flowchart

Selecting optimal cluster count via BIC (cluster_utils.py:46-57):

```
                          +------------------+
                          |     START        |
                          | get_optimal_     |
                          |   clusters()     |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | max_clusters =   |
                          |   min(50,        |
                          |   len(emb))      |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | n_clusters =     |
                          |   range(1, max)  |
                          | bics = []        |
                          +--------+---------+
                                   |
                                   v
         +-------->      +------------------+
         |               | FOR n in         |
         |               |   n_clusters:    |
         |               +--------+---------+
         |                        |
         |                        v
         |               +------------------+
         |               | gm = Gaussian    |
         |               |   Mixture(       |
         |               |   n_components=n)|
         |               | gm.fit(emb)      |
         |               +--------+---------+
         |                        |
         |                        v
         |               +------------------+
         |               | bics.append(     |
         |               |   gm.bic(emb))   |
         |               +--------+---------+
         |                        |
         +<-----------------------+
         (next n)
                                  |
                                  v
                          +------------------+
                          | optimal =        |
                          |   n_clusters[    |
                          |   argmin(bics)]  |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          | RETURN optimal   |
                          +------------------+
```

## Summary: Key Decision Points

| Flowchart | Decision Point | Condition | Impact |
|-----------|---------------|-----------|--------|
| `build_from_text` | Multithreading? | `use_multithreading` param | Parallel vs sequential leaf creation |
| `construct_tree` | Stop building? | `len(nodes) <= dim + 1` | Early termination |
| `construct_tree` | Process clusters | `use_multithreading` param | Parallel vs sequential summarization |
| `retrieve` | Mode selection | `collapse_tree` param | Flat vs hierarchical retrieval |
| `retrieve_info` | Selection mode | `selection_mode` config | top_k vs threshold filtering |
| `perform_clustering` | Local clustering | `len(emb) <= dim + 1` | Single cluster vs UMAP+GMM |
| `RAPTOR_Clustering` | Recluster? | `total_length > max_length` | Recursive subdivision |
| `get_optimal_clusters` | Best k | `argmin(BIC scores)` | GMM component count |
