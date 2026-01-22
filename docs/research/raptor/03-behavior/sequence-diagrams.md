# RAPTOR Sequence Diagrams

> Step-by-step interaction flows for key operations

## 1. add_documents() Flow - Full Indexing Pipeline

This sequence shows the complete flow when documents are added to RAPTOR for indexing.

```
+----------------+     +-------------------+     +-------------------+     +----------------+
|     User       |     | RetrievalAugm.    |     | ClusterTreeBuilder|     |  cluster_utils |
+----------------+     +-------------------+     +-------------------+     +----------------+
        |                      |                         |                        |
        | add_documents(docs)  |                         |                        |
        |--------------------->|                         |                        |
        |                      |                         |                        |
        |                      | [Check if tree exists]  |                        |
        |                      | (lines 211-217)         |                        |
        |                      |                         |                        |
        |                      | build_from_text(docs)   |                        |
        |                      |------------------------>|                        |
        |                      |                         |                        |
        |                      |                         | split_text()           |
        |                      |                         | (utils.py:22-100)      |
        |                      |                         |----------------------->|
        |                      |                         |<-----------------------|
        |                      |                         | chunks[]               |
        |                      |                         |                        |
        |                      |                         |                        |
        |                      |              +----------+----------+             |
        |                      |              | multithreaded_      |             |
        |                      |              | create_leaf_nodes() |             |
        |                      |              | (lines 238-258)     |             |
        |                      |              +----------+----------+             |
        |                      |                         |                        |
        |                      |                         |                        |
        |                      |              +----------+----------+             |
        |                      |              | For each chunk:     |             |
        |                      |              |   create_node()     |             |
        |                      |              |   (lines 158-179)   |             |
        |                      |              +----------+----------+             |
        |                      |                         |                        |
        |                      |                         | create_embedding()     |
        |                      |                         |----------------------->|
        |                      |                         |<-----------------------|
        |                      |                         | embedding vector       |
        |                      |                         |                        |
        |                      |                         | [Creates Node with     |
        |                      |                         |  text, index, emb]     |
        |                      |                         |                        |
        |                      |              +----------+----------+             |
        |                      |              | leaf_nodes created  |             |
        |                      |              | layer_to_nodes[0]   |             |
        |                      |              +----------+----------+             |
        |                      |                         |                        |
        |                      |                         |                        |
        |                      |                         | construct_tree()       |
        |                      |                         | (cluster_tree:55-151)  |
        |                      |                         |----------------------->|
        |                      |                         |                        |
        |                      |                         |    (See Tree Building  |
        |                      |                         |     Sequence Below)    |
        |                      |                         |                        |
        |                      |                         |<-----------------------|
        |                      |                         | root_nodes             |
        |                      |                         |                        |
        |                      |                         |                        |
        |                      |<------------------------|                        |
        |                      | Tree(all_nodes,         |                        |
        |                      |      root_nodes,        |                        |
        |                      |      leaf_nodes,        |                        |
        |                      |      num_layers,        |                        |
        |                      |      layer_to_nodes)    |                        |
        |                      |                         |                        |
        |                      | TreeRetriever(config,   |                        |
        |                      |               tree)     |                        |
        |                      | (line 220)              |                        |
        |                      |------------------------>+                        |
        |                      |                         |                        |
        |<---------------------|                         |                        |
        | [Indexing Complete]  |                         |                        |
        |                      |                         |                        |
```

## 2. construct_tree() - Tree Building Loop Sequence

Detailed sequence for the hierarchical tree construction (ClusterTreeBuilder.construct_tree):

```
+-------------------+     +----------------+     +------------------+     +----------------+
| ClusterTreeBuilder|     | RAPTOR_Cluster |     | Summariz. Model  |     | Embedding Model|
+-------------------+     +----------------+     +------------------+     +----------------+
        |                        |                       |                       |
        | [Layer 0: leaf_nodes   |                       |                       |
        |  already created]      |                       |                       |
        |                        |                       |                       |
   +----+----+                   |                       |                       |
   | FOR layer in               |                       |                       |
   | range(num_layers)          |                       |                       |
   +----+----+                   |                       |                       |
        |                        |                       |                       |
        | get_node_list()        |                       |                       |
        | (line 93)              |                       |                       |
        |                        |                       |                       |
        | [CHECK: len(nodes) <=  |                       |                       |
        |  reduction_dim + 1?]   |                       |                       |
        |                        |                       |                       |
        | [If YES: STOP, return] |                       |                       |
        | (lines 95-100)         |                       |                       |
        |                        |                       |                       |
        | perform_clustering()   |                       |                       |
        |----------------------->|                       |                       |
        |                        |                       |                       |
        |                        | get_embeddings()      |                       |
        |                        | (line 143)            |                       |
        |                        |                       |                       |
        |                        | global_cluster_emb()  |                       |
        |                        | UMAP reduce to dim    |                       |
        |                        | (lines 23-34)         |                       |
        |                        |                       |                       |
        |                        | GMM_cluster()         |                       |
        |                        | find optimal clusters |                       |
        |                        | (lines 60-66)         |                       |
        |                        |                       |                       |
        |                        | [For each global      |                       |
        |                        |  cluster:]            |                       |
        |                        |                       |                       |
        |                        | local_cluster_emb()   |                       |
        |                        | (lines 37-43)         |                       |
        |                        |                       |                       |
        |                        | GMM_cluster()         |                       |
        |                        | (lines 100-102)       |                       |
        |                        |                       |                       |
        |                        | [Validate cluster     |                       |
        |                        |  size <= max_length]  |                       |
        |                        | (lines 166-183)       |                       |
        |                        |                       |                       |
        |                        | [If too large:        |                       |
        |                        |  recursive recluster] |                       |
        |                        |                       |                       |
        |<-----------------------|                       |                       |
        | clusters[[Node,...],  |                       |                       |
        |          [Node,...]]   |                       |                       |
        |                        |                       |                       |
   +----+----+                   |                       |                       |
   | FOR each cluster           |                       |                       |
   | (lines 129-137)            |                       |                       |
   +----+----+                   |                       |                       |
        |                        |                       |                       |
        | get_text(cluster)      |                       |                       |
        | (line 69)              |                       |                       |
        |                        |                       |                       |
        | summarize(node_texts)  |                       |                       |
        |--------------------------------------->|       |                       |
        |                        |               |       |                       |
        |                        |               | LLM.summarize()               |
        |                        |               | (GPT-3.5/custom)              |
        |                        |               |       |                       |
        |<---------------------------------------|       |                       |
        | summarized_text        |               |       |                       |
        |                        |                       |                       |
        | create_node()          |                       |                       |
        | (lines 80-85)          |                       |                       |
        |---------------------------------------------->|                       |
        |                        |                       |                       |
        |                        |                       | create_embedding()    |
        |                        |                       | for summary text      |
        |                        |                       |                       |
        |<----------------------------------------------|                       |
        | new_parent_node        |                       |                       |
        | [with children_indices]|                       |                       |
        |                        |                       |                       |
   +----+----+                   |                       |                       |
   | new_level_nodes[           |                       |                       |
   |   next_index] = node       |                       |                       |
   | next_index++               |                       |                       |
   +----+----+                   |                       |                       |
        |                        |                       |                       |
        | [End cluster loop]     |                       |                       |
        |                        |                       |                       |
        | layer_to_nodes[layer+1]|                       |                       |
        |   = new_level_nodes    |                       |                       |
        | (line 139)             |                       |                       |
        |                        |                       |                       |
        | current_level_nodes =  |                       |                       |
        |   new_level_nodes      |                       |                       |
        | (line 140)             |                       |                       |
        |                        |                       |                       |
        | all_tree_nodes.update()|                       |                       |
        | (line 141)             |                       |                       |
        |                        |                       |                       |
        | [Next layer iteration] |                       |                       |
        |                        |                       |                       |
```

## 3. answer_question() Flow - Full Retrieval Pipeline

```
+----------------+     +-------------------+     +---------------+     +-------------+
|     User       |     | RetrievalAugm.    |     | TreeRetriever |     |   QAModel   |
+----------------+     +-------------------+     +---------------+     +-------------+
        |                      |                       |                     |
        | answer_question(     |                       |                     |
        |   question,          |                       |                     |
        |   top_k=10,          |                       |                     |
        |   collapse_tree=True)|                       |                     |
        |--------------------->|                       |                     |
        |                      |                       |                     |
        |                      | [Check retriever      |                     |
        |                      |  initialized]         |                     |
        |                      | (line 248-251)        |                     |
        |                      |                       |                     |
        |                      | retrieve(question,    |                     |
        |                      |          ...)         |                     |
        |                      | (lines 290-291)       |                     |
        |                      |--------------------->|                     |
        |                      |                       |                     |
        |                      |                       | [Validate params]   |
        |                      |                       | (lines 276-300)     |
        |                      |                       |                     |
        |                      |                       |                     |
        |                      |           +-----------+-----------+         |
        |                      |           | collapse_tree == True?|         |
        |                      |           +-----------+-----------+         |
        |                      |                       |                     |
        |                      |          YES          |          NO         |
        |                      |           |           |           |         |
        |                      |           v           |           v         |
        |                      |  +----------------+   |  +----------------+ |
        |                      |  | retrieve_info_ |   |  | retrieve_info()|  |
        |                      |  | collapse_tree()|   |  | (lines 197-250)|  |
        |                      |  | (lines 158-195)|   |  +----------------+ |
        |                      |  +----------------+   |           |         |
        |                      |           |           |           |         |
        |                      |           +-----------+-----------+         |
        |                      |                       |                     |
        |                      |                       | (See Retrieval      |
        |                      |                       |  Sequences Below)   |
        |                      |                       |                     |
        |                      |<---------------------|                     |
        |                      | context,              |                     |
        |                      | layer_information     |                     |
        |                      |                       |                     |
        |                      | answer_question(      |                     |
        |                      |   context, question)  |                     |
        |                      | (line 294)            |                     |
        |                      |------------------------------------------>|
        |                      |                       |                     |
        |                      |                       |     LLM.generate()  |
        |                      |                       |     (GPT-3.5/custom)|
        |                      |                       |                     |
        |                      |<------------------------------------------|
        |                      | answer                |                     |
        |                      |                       |                     |
        |<---------------------|                       |                     |
        | answer (or           |                       |                     |
        | answer + layer_info) |                       |                     |
        |                      |                       |                     |
```

## 4. retrieve_information_collapse_tree() - Collapsed Retrieval Sequence

```
+---------------+     +----------------+     +----------------+
| TreeRetriever |     | EmbeddingModel |     |     utils      |
+---------------+     +----------------+     +----------------+
        |                     |                      |
        | [Start: query str]  |                      |
        |                     |                      |
        | create_embedding(   |                      |
        |   query)            |                      |
        | (line 170)          |                      |
        |-------------------->|                      |
        |<--------------------|                      |
        | query_embedding     |                      |
        |                     |                      |
        | get_node_list(      |                      |
        |   tree.all_nodes)   |                      |
        | (line 174)          |                      |
        |------------------------------------>|      |
        |<------------------------------------|      |
        | node_list (ALL nodes               |      |
        |  from ALL layers)                  |      |
        |                     |                      |
        | get_embeddings(     |                      |
        |   node_list,        |                      |
        |   context_emb_model)|                      |
        | (line 176)          |                      |
        |------------------------------------>|      |
        |<------------------------------------|      |
        | embeddings[]        |                      |
        |                     |                      |
        | distances_from_     |                      |
        |   embeddings()      |                      |
        | (line 178)          |                      |
        |------------------------------------>|      |
        |<------------------------------------|      |
        | distances[]         |                      |
        | (cosine distances)  |                      |
        |                     |                      |
        | indices_of_nearest_ |                      |
        |   neighbors()       |                      |
        | (line 180)          |                      |
        |------------------------------------>|      |
        |<------------------------------------|      |
        | indices[] (sorted   |                      |
        |  by distance ASC)   |                      |
        |                     |                      |
   +----+----+                |                      |
   | FOR idx in indices[:top_k]                      |
   | (lines 183-192)         |                      |
   +----+----+                |                      |
        |                     |                      |
        | node = node_list[idx]                      |
        | node_tokens = tokenize(node.text)          |
        |                     |                      |
        | [IF total_tokens +  |                      |
        |  node_tokens >      |                      |
        |  max_tokens: BREAK] |                      |
        |                     |                      |
        | selected_nodes.     |                      |
        |   append(node)      |                      |
        | total_tokens +=     |                      |
        |   node_tokens       |                      |
        |                     |                      |
        | [End loop]          |                      |
        |                     |                      |
        | get_text(           |                      |
        |   selected_nodes)   |                      |
        | (line 194)          |                      |
        |------------------------------------>|      |
        |<------------------------------------|      |
        | context (concatenated text)        |      |
        |                     |                      |
        | RETURN (selected_nodes, context)          |
        |                     |                      |
```

## 5. retrieve_information() - Tree Traversal Retrieval Sequence

```
+---------------+     +----------------+     +----------------+
| TreeRetriever |     | EmbeddingModel |     |     utils      |
+---------------+     +----------------+     +----------------+
        |                     |                      |
        | [Input: current_nodes = layer_to_nodes[start_layer]]
        | [Input: query, num_layers]                 |
        |                     |                      |
        | create_embedding(query)                    |
        | (line 212)          |                      |
        |-------------------->|                      |
        |<--------------------|                      |
        | query_embedding     |                      |
        |                     |                      |
        | selected_nodes = [] |                      |
        | node_list = current_nodes                  |
        |                     |                      |
   +----+----+                |                      |
   | FOR layer in range(num_layers)                  |
   | (lines 218-248)         |                      |
   +----+----+                |                      |
        |                     |                      |
        | get_embeddings(     |                      |
        |   node_list, ...)   |                      |
        | (line 220)          |                      |
        |------------------------------------>|      |
        |<------------------------------------|      |
        | embeddings[]        |                      |
        |                     |                      |
        | distances_from_     |                      |
        |   embeddings()      |                      |
        | (line 222)          |                      |
        |------------------------------------>|      |
        |<------------------------------------|      |
        | distances[]         |                      |
        |                     |                      |
        | indices_of_nearest_ |                      |
        |   neighbors()       |                      |
        | (line 224)          |                      |
        |------------------------------------>|      |
        |<------------------------------------|      |
        | indices[]           |                      |
        |                     |                      |
   +----+----+                |                      |
   | selection_mode?         |                      |
   +----+----+                |                      |
        |                     |                      |
        | "threshold"         |      "top_k"         |
        |     |               |         |            |
        |     v               |         v            |
        | best_indices =      |  best_indices =      |
        | [i for i if         |  indices[:top_k]     |
        |  dist[i]>threshold] |  (line 232)          |
        | (lines 226-229)     |                      |
        |     |               |         |            |
        |     +-------+-------+         |            |
        |             |                              |
        | nodes_to_add = [node_list[idx]             |
        |                 for idx in best_indices]   |
        | (line 234)          |                      |
        |                     |                      |
        | selected_nodes.extend(nodes_to_add)        |
        | (line 236)          |                      |
        |                     |                      |
   +----+----+                |                      |
   | IF layer != num_layers-1 (not last layer)       |
   | (lines 238-247)         |                      |
   +----+----+                |                      |
        |                     |                      |
        | child_nodes = []    |                      |
        |                     |                      |
        | FOR idx in best_indices:                   |
        |   child_nodes.extend(                      |
        |     node_list[idx].children)               |
        | (lines 242-243)     |                      |
        |                     |                      |
        | child_nodes = unique(child_nodes)          |
        | (line 246)          |                      |
        |                     |                      |
        | node_list = [tree.all_nodes[i]             |
        |              for i in child_nodes]         |
        | (line 247)          |                      |
        |                     |                      |
        | [Next layer: drill down to children]       |
        |                     |                      |
   +----+----+                |                      |
   | End FOR layer           |                      |
   +----+----+                |                      |
        |                     |                      |
        | get_text(           |                      |
        |   selected_nodes)   |                      |
        | (line 249)          |                      |
        |------------------------------------>|      |
        |<------------------------------------|      |
        | context             |                      |
        |                     |                      |
        | RETURN (selected_nodes, context)          |
        |                     |                      |
```

## Key Decision Points Summary

| Method | Decision Point | Options | Impact |
|--------|---------------|---------|--------|
| `add_documents` | Tree exists? | Overwrite / Add to existing | Line 211-217 |
| `build_from_text` | Use multithreading? | Parallel / Sequential | Line 275-281 |
| `construct_tree` | Enough nodes for layer? | Continue / Stop building | Line 95-100 |
| `construct_tree` | Cluster too large? | Recluster / Accept | cluster_utils:172 |
| `retrieve` | collapse_tree? | Flat search / Tree traversal | Line 302-311 |
| `retrieve_information` | selection_mode | top_k / threshold | Lines 226-232 |
| `retrieve_information` | Last layer? | Stop / Get children | Line 238 |

## Object Lifecycle

```
1. RetrievalAugmentation.__init__()
   |
   +-- Creates ClusterTreeBuilder (with config)
   +-- Creates QAModel (GPT-3.5 default)
   +-- tree = None, retriever = None
   
2. add_documents(text)
   |
   +-- tree_builder.build_from_text() --> Tree object
   +-- Creates TreeRetriever(config, tree)
   
3. answer_question(question)
   |
   +-- retriever.retrieve() --> context string
   +-- qa_model.answer_question() --> answer string
```
