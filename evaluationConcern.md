Your concern is correct: none of the current final graph-generation metrics directly evaluates the decoded node or edge features.
Your concern is correct: none of the current final graph-generation metrics directly evaluates the decoded node or edge features.

`F1-PR` is not an adjacency-entry F1 score, but it is still effectively adjacency-only in the current pipeline. It is the harmonic mean of generative precision and recall measured in Random-GIN embedding space. However, those GIN embeddings are constructed entirely from graph topology or topology-derived features.

### Why F1-PR ignores decoded features

The final generator calls only the adjacency decoder:

- [`main.py:1966`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/main.py:1966) calls `model.decode(z)`.
- `model.decode` is specifically the adjacency decoder, while the node and edge heads are separate ([`model.py:94`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/model.py:94)).
- The feature heads are called only in the training `forward()` path ([`model.py:117`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/model.py:117)).
- Final evaluation saves `nx.to_numpy_array(G)`, so only adjacency matrices survive ([`main.py:2022`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/main.py:2022)). Reference graphs are also saved as adjacency matrices only.

For the local Random-GIN evaluation:

- NetworkX graphs are converted to DGL using topology only ([`stat_rnn.py:351`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/stat_rnn.py:351)).
- Because no node attribute is present, GIN substitutes node degree as its input feature ([`gin_evaluation.py:117`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/third_party/ggmeval/evaluation/gin_evaluation.py:117)).
- The evaluator uses `edge_feat_dim=0`, so no edge features are consumed ([`gin_evaluation.py:13`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/third_party/ggmeval/evaluation/gin_evaluation.py:13)).
- Precision, recall, and F1-PR are then computed from distances between these adjacency-derived GIN embeddings ([`gin_evaluation.py:344`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/third_party/ggmeval/evaluation/gin_evaluation.py:344)).

The third-party F1-PR is also adjacency-only. Its “structural features” are degree, clustering, and square-clustering computed from adjacency—not decoded attributes ([`evaluate_graph_realism_batch.py:214`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/scripts/evaluate_graph_realism_batch.py:214)).

### Current final evaluation criteria

| Criterion | Decoded node features | Decoded edge features | Actual input |
|---|---:|---:|---|
| Degree MMD | No | No | Adjacency |
| Clustering MMD | No | No | Adjacency |
| Orbit MMD | No | No | Adjacency |
| Spectral MMD | No | No | Adjacency |
| Diameter MMD | No | No | Adjacency |
| Triangle/sparsity/edge-count diagnostics | No | No | Adjacency |
| Local GIN MMD-RBF | No | No | Adjacency + degree derived from adjacency |
| Local precision/recall/F1-PR | No | No | Same GIN structural embeddings |
| Third-party GIN F1-PR/MMD-RBF/linear MMD | No | No | Adjacency-derived degree/clustering/square-clustering |
| Best-checkpoint score | No | No | Combination of the metrics above |

The current `table3_priority` checkpoint score combines MMD-RBF, F1-PR error, and the five structural MMDs ([`ranking_score.py:143`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/ranking_score.py:143)). Consequently, checkpoint selection is also blind to decoded node and edge quality.

A useful invariance statement is:

> Two models producing identical adjacency samples but completely different node and edge features will receive exactly the same current final metrics and the same validation checkpoint score.

### Criteria that directly use decoded features

These exist only inside training/reconstruction diagnostics, not final held-out generation evaluation:

| Criterion | Direct feature dependency | Limitation |
|---|---|---|
| `node_feat_loss` | Directly uses decoded node logits | Training reconstruction loss only |
| `edge_feat_loss` | Directly uses decoded edge logits | Training reconstruction loss; evaluated only on true existing edges |
| Soft relational motif loss | Directly uses decoded adjacency, node logits, and edge logits when its rules contain those predicates | Training objective, not final generated-sample metric |
| Hard motif loss / exact motif match | Uses discretized node/edge predictions for feature-containing motifs | Training-batch diagnostic; not saved as final test evaluation |
| Unit-relation motif loss | Adjacency only | Bare relation motifs do not test edge attributes |

The motif wrapper explicitly builds node assignments from `node_feat_logits` and edge assignments from `edge_feat_logits` ([`data.py:2496`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/data.py:2496)). The motif counter then has separate paths for node-feature predicates, adjacency predicates, and edge-feature predicates ([`motif_counter.py:424`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/motif_counting/motif_counter.py:424)).

Therefore, the only direct recognition of your method’s attributed-motif advantage currently happens during training. Any improvement in reported F1-PR or MMD is indirect—through changes to the shared latent representation or adjacency decoder.

### Recommended evaluation correction

The most important addition would be a feature-aware final evaluation pipeline:

1. During generation, decode and save `(A, X, E)` from the same sampled `z`, not just `A`.
2. Add a feature-aware Random-GIN F1-PR/MMD using decoded node attributes and edge attributes.
3. Report separate node-feature distribution and edge-feature distribution distances.
4. Most importantly for this method, compare distributions of attributed relational motif vectors on generated versus test graphs.
5. Include one or more of these feature-aware validation metrics in checkpoint selection.

The attributed-motif metric is especially important: marginal node/edge distributions alone can look correct while the relationship between topology and attributes is wrong. Your motif-count representation can evaluate that joint structure directly.
`F1-PR` is not an adjacency-entry F1 score, but it is still effectively adjacency-only in the current pipeline. It is the harmonic mean of generative precision and recall measured in Random-GIN embedding space. However, those GIN embeddings are constructed entirely from graph topology or topology-derived features.

### Why F1-PR ignores decoded features

The final generator calls only the adjacency decoder:

- [`main.py:1966`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/main.py:1966) calls `model.decode(z)`.
- `model.decode` is specifically the adjacency decoder, while the node and edge heads are separate ([`model.py:94`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/model.py:94)).
- The feature heads are called only in the training `forward()` path ([`model.py:117`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/model.py:117)).
- Final evaluation saves `nx.to_numpy_array(G)`, so only adjacency matrices survive ([`main.py:2022`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/main.py:2022)). Reference graphs are also saved as adjacency matrices only.

For the local Random-GIN evaluation:

- NetworkX graphs are converted to DGL using topology only ([`stat_rnn.py:351`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/stat_rnn.py:351)).
- Because no node attribute is present, GIN substitutes node degree as its input feature ([`gin_evaluation.py:117`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/third_party/ggmeval/evaluation/gin_evaluation.py:117)).
- The evaluator uses `edge_feat_dim=0`, so no edge features are consumed ([`gin_evaluation.py:13`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/third_party/ggmeval/evaluation/gin_evaluation.py:13)).
- Precision, recall, and F1-PR are then computed from distances between these adjacency-derived GIN embeddings ([`gin_evaluation.py:344`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/third_party/ggmeval/evaluation/gin_evaluation.py:344)).

The third-party F1-PR is also adjacency-only. Its “structural features” are degree, clustering, and square-clustering computed from adjacency—not decoded attributes ([`evaluate_graph_realism_batch.py:214`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/scripts/evaluate_graph_realism_batch.py:214)).

### Current final evaluation criteria

| Criterion | Decoded node features | Decoded edge features | Actual input |
|---|---:|---:|---|
| Degree MMD | No | No | Adjacency |
| Clustering MMD | No | No | Adjacency |
| Orbit MMD | No | No | Adjacency |
| Spectral MMD | No | No | Adjacency |
| Diameter MMD | No | No | Adjacency |
| Triangle/sparsity/edge-count diagnostics | No | No | Adjacency |
| Local GIN MMD-RBF | No | No | Adjacency + degree derived from adjacency |
| Local precision/recall/F1-PR | No | No | Same GIN structural embeddings |
| Third-party GIN F1-PR/MMD-RBF/linear MMD | No | No | Adjacency-derived degree/clustering/square-clustering |
| Best-checkpoint score | No | No | Combination of the metrics above |

The current `table3_priority` checkpoint score combines MMD-RBF, F1-PR error, and the five structural MMDs ([`ranking_score.py:143`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/ranking_score.py:143)). Consequently, checkpoint selection is also blind to decoded node and edge quality.

A useful invariance statement is:

> Two models producing identical adjacency samples but completely different node and edge features will receive exactly the same current final metrics and the same validation checkpoint score.

### Criteria that directly use decoded features

These exist only inside training/reconstruction diagnostics, not final held-out generation evaluation:

| Criterion | Direct feature dependency | Limitation |
|---|---|---|
| `node_feat_loss` | Directly uses decoded node logits | Training reconstruction loss only |
| `edge_feat_loss` | Directly uses decoded edge logits | Training reconstruction loss; evaluated only on true existing edges |
| Soft relational motif loss | Directly uses decoded adjacency, node logits, and edge logits when its rules contain those predicates | Training objective, not final generated-sample metric |
| Hard motif loss / exact motif match | Uses discretized node/edge predictions for feature-containing motifs | Training-batch diagnostic; not saved as final test evaluation |
| Unit-relation motif loss | Adjacency only | Bare relation motifs do not test edge attributes |

The motif wrapper explicitly builds node assignments from `node_feat_logits` and edge assignments from `edge_feat_logits` ([`data.py:2496`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/data.py:2496)). The motif counter then has separate paths for node-feature predicates, adjacency predicates, and edge-feature predicates ([`motif_counter.py:424`](/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/motif_counting/motif_counter.py:424)).

Therefore, the only direct recognition of your method’s attributed-motif advantage currently happens during training. Any improvement in reported F1-PR or MMD is indirect—through changes to the shared latent representation or adjacency decoder.

### Recommended evaluation correction

The most important addition would be a feature-aware final evaluation pipeline:

1. During generation, decode and save `(A, X, E)` from the same sampled `z`, not just `A`.
2. Add a feature-aware Random-GIN F1-PR/MMD using decoded node attributes and edge attributes.
3. Report separate node-feature distribution and edge-feature distribution distances.
4. Most importantly for this method, compare distributions of attributed relational motif vectors on generated versus test graphs.
5. Include one or more of these feature-aware validation metrics in checkpoint selection.

The attributed-motif metric is especially important: marginal node/edge distributions alone can look correct while the relationship between topology and attributes is wrong. Your motif-count representation can evaluate that joint structure directly.