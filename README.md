# ST-Graph and ConGNN Demo

This repository provides a reproducible demonstration of the social graph
construction and community detection workflow developed for the manuscript:

> **Exploring the Evolution of Public Cognition Toward Generative AI Through
> Sentiment-Topic Mining and Community Detection in Social Graphs**

**Journal:** *Information Processing & Management (IP&M)*

## Overview

The demo constructs a Semantic-Topological Social Graph (ST-Graph) from
synthetic Weibo-style posts and applies a GraphSAGE-based encoder trained with
an InfoNCE objective, referred to as ConGNN, to learn node representations for
unsupervised community detection.

The workflow includes:

1. Text representation using Chinese BERT `[CLS]` embeddings.
2. Log transformation and standardization of engagement and user-profile
   features.
3. Weighted fusion of textual, engagement, and user-profile similarities.
4. Temporally directed graph construction using adaptive K-nearest-neighbor
   filtering.
5. PageRank calculation and graph-feature enhancement.
6. GraphSAGE-based node representation learning with an InfoNCE objective.
7. K-means clustering with the number of communities selected from `K=2` to
   `K=8` according to the Silhouette coefficient.
8. Calculation of partition-stability and graph-structure metrics.

The repository focuses on the core ST-Graph and ConGNN workflow and does not
include the separate benchmark-dataset experiments reported in the manuscript.

## Repository Contents

```text
Graph_cluster_demo.py   End-to-end executable demonstration
README.md               Project description and usage instructions
```

## Demo Data

To protect the privacy of social media users and avoid redistributing platform
content, the script does not load the original Weibo dataset. Instead, it
contains an in-memory list of synthetic Weibo-style posts with the same input
schema used by the graph-construction pipeline:

| Field | Description |
|---|---|
| `text` | Post text |
| `timestamp` | Publication time |
| `likes` | Number of likes |
| `shares` | Number of reposts |
| `comments` | Number of comments |
| `user_followers` | Follower count of the publishing account |
| `user_activity` | Historical posting activity |
| `actor_type` | Type of publishing entity |

These records are provided only to demonstrate the execution and expected data
structure. They are not part of the empirical dataset analyzed in the paper.

## Requirements

Python 3.9 or later is recommended. Install the required packages with:

```bash
pip install numpy pandas scikit-learn networkx matplotlib torch torch-geometric transformers
```

PyTorch and PyTorch Geometric should be installed using versions compatible
with the local CPU or CUDA environment. See the official installation guides
for platform-specific commands.

## Running the Demo

Run the complete workflow with the public Chinese BERT model:

```bash
python Graph_cluster_demo.py
```

By default, the script attempts to load
`google-bert/bert-base-chinese` from Hugging Face. If the model cannot be
loaded, the script automatically uses a deterministic TF-IDF representation so
that the remaining workflow can run offline.

To run the lightweight offline version directly:

```bash
python Graph_cluster_demo.py --text-encoder tfidf
```

To require BERT and disable the TF-IDF fallback:

```bash
python Graph_cluster_demo.py --text-encoder bert --strict-bert
```

Optional arguments include:

```text
--bert-model MODEL_ID    Hugging Face model identifier
--epochs N               Number of contrastive-training epochs (default: 30)
--output-dir PATH        Output directory (default: demo_output)
```

## Outputs

The script creates the selected output directory and writes:

```text
community_assignments.csv   Synthetic posts, PageRank values, and community labels
graph_cluster_demo.png      ST-Graph, training loss, and Silhouette comparison
```

It also reports the following values in the terminal:

- Number of nodes and directed edges
- Silhouette coefficients for `K=2,...,8`
- Selected number of communities
- Mean ARI and NMI across repeated K-means initializations
- Modularity, graph density, and average clustering coefficient

These metrics describe the graph partition obtained for the synthetic demo
data and should not be interpreted as the empirical results reported in the
manuscript.

## Methodological Notes

- Nodes represent individual posts rather than users.
- Directed edges represent latent associations based on textual similarity,
  engagement similarity, user-profile similarity, and publication order.
- Edges are directed from later posts to earlier related posts to preserve the
  temporal ordering used in ST-Graph construction.
- The graph is a similarity-based analytical representation and does not treat
  each edge as an observed repost, comment, or follower relationship.
- The simulated data and compact full-batch training procedure are intended for
  code inspection and reproducibility testing. The empirical analysis uses the
  full study dataset and the parameter settings described in the manuscript.

## Reproducibility

The random seed is fixed at `42` for NumPy, PyTorch, K-means, and graph layout
operations. Exact numerical results may still vary slightly across operating
systems, library versions, and CPU/GPU implementations.

## Citation

Citation information will be added after publication. Until then, please cite
the manuscript by its title:

```text
Exploring the Evolution of Public Cognition Toward Generative AI Through
Sentiment-Topic Mining and Community Detection in Social Graphs.
Information Processing & Management.
```

## Data Availability

The public demo uses synthetic data. Access to the original Weibo data is
subject to platform terms, privacy considerations, and the data-availability
statement provided in the manuscript.

