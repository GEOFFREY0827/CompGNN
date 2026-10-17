# CompGNN

## Project layout

The source tree separates the reusable research core from the newest experiment
and older experiment implementations:

```text
src/compgnn/
├── core/
│   ├── encode.py                  # Dataset loading and embedding generation
│   ├── gnn_rl.py                  # Set GNN and allow/flag/block policy
│   └── intent_reconstruction.py   # Reconstruct the original task intent
├── experiments/
│   ├── latest/
│   │   ├── relation_retrieval.py  # Relation-conditioned node and graph retrieval
│   │   └── retrieval_rl.py        # REINFORCE training for that retriever
│   └── archive/
│       ├── edge_retrieval.py      # Earlier latent relation-slot retriever
│       ├── memory_retrieval.py    # Earlier noisy-memory retrieval experiment
│       └── label_graph.py         # Earlier weak edge-label generation
```

The `latest` directory is the current retrieval direction: explicit relation
conditioning followed by policy-gradient training. The archived modules remain
available for comparison and reproducibility, but are not part of the current
mainline experiment.

The command-line entry points are:

```bash
compgnn-encode
compgnn-label-graph
```

The latter intentionally points to the archived weak-label generator because
the current retrieval experiment consumes relation-labelled edges directly.
