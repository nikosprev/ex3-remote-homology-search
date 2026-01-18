 # ex3-remote-homology-search

Software project for algorithmic problems part 3 - Remote protein homology search


Part 1 - Embeddings with ESM-2


Conceptual pipeline (high level)

FASTA file (protein sequences)

↓

ESM-2 model

↓

Per-amino-acid representations (last layer)

↓

Mean pooling

↓

One vector per protein

↓

Save




Part 2 - BLAST

If not installed run:

sudo apt install ncbi-blast+


Then run build_blast.py once to create the blast database

BLAST provides:


A ranked list of similar proteins per query


Sequence identity percentages


A Top-N list used to evaluate ANN methods


Ground truth for Recall@N



# Adapting C++ code to Python


For Adapting the C++ code to python cleanly we decided to implement a wrapper for the C++ templated code.

We used ```Pybind11`` for this task .The implementation is located inside the Algos folder in src


# Tune the Parameters for the Methods on this Task


Before actively developing the ```protein_search.py``` we decided to tune the parameters for this task

We run a simple tuning script and evaluated each Algorithm to the QPS and Recall and avg distance against the groundtruth KNN


The results are this:


Here are the curated tables for each algorithm, selected to highlight the performance trade-offs (Recall vs. QPS) and how specific parameters influence the results.

### 1. LSH (Locality Sensitive Hashing)


> **Observation:** These trials demonstrate the classic LSH trade-off. Increasing the window size ($w$) or the number of hash tables ($L$) significantly boosts **Recall** but drastically reduces **QPS**. Lower $k$ generally yields higher recall but requires searching more candidates.


| Recall | QPS | Avg Neighbor Dist | $k$ (Hash Size) | $L$ (Num Tables) | $w$ (Window) |

| :--- | :--- | :--- | :--- | :--- | :--- |

| **1.0000** | 5.86 | 1.2568 | 2 | 5 | 4.0 |

| **1.0000** | 5.84 | 1.2568 | 2 | 8 | 4.0 |

| 0.9333 | 6.56 | 1.2578 | 4 | 5 | 4.0 |

| 0.9000 | 7.37 | 1.2579 | 6 | 8 | 4.0 |

| 0.8333 | 6.67 | 1.2584 | 2 | 8 | 2.0 |

| 0.7667 | 8.58 | 1.2596 | 2 | 5 | 2.0 |

| 0.5333 | 17.35 | 1.2645 | 2 | 5 | 1.0 |

| 0.4000 | 24.89 | 1.2677 | 4 | 5 | 2.0 |

| 0.2333 | **55.54** | 1.2730 | 2 | 5 | 0.5 |

| 0.2333 | **80.36** | 1.2764 | 6 | 5 | 2.0 |


---


### 2. Hypercube


> **Observation:** For Hypercube, achieving non-zero recall is challenging with low search limits. The "promising" trials here show that higher $M$ (candidates checked) and probes are necessary to get decent Recall, though this algorithm struggles with recall compared to LSH in this specific configuration.


| Recall | QPS | Avg Neighbor Dist | $M$ (Max Candidates) | $k$ (Dimensions) | Probes | $w$ (Window) |

| :--- | :--- | :--- | :--- | :--- | :--- | :--- |

| **0.4667** | 17.52 | 1.2651 | 10000 | 10 | 10 | 3.0 |

| 0.3667 | 15.35 | 1.2668 | 10000 | 9 | 10 | 3.0 |

| 0.3000 | 17.85 | 1.2687 | 10000 | 9 | 7 | 3.0 |

| 0.3000 | 29.22 | 1.2691 | 5000 | 10 | 10 | 3.0 |

| 0.3000 | 21.15 | 1.2698 | 10000 | 11 | 10 | 3.0 |

| 0.2667 | 33.67 | 1.2703 | 5000 | 10 | 7 | 3.0 |

| 0.2667 | 27.10 | 1.2703 | 10000 | 10 | 7 | 3.0 |

| 0.2333 | **39.51** | 1.2731 | 5000 | 10 | 5 | 3.0 |

| 0.2333 | 36.55 | 1.2731 | 10000 | 9 | 5 | 3.0 |

| 0.2000 | 39.44 | 1.2736 | 5000 | 9 | 2 | 3.0 |


---


### 3. IVF (Inverted File Index)


> **Observation:** This data illustrates the `nprobe` mechanic perfectly. As you increase the number of clusters probed (`nprobe`), the **Recall** increases significantly, but the search speed (**QPS**) drops proportionally.


| Recall | QPS | Avg Neighbor Dist | Num Clusters | `nprobe` |

| :--- | :--- | :--- | :--- | :--- |

| **0.7667** | 17.02 | 1.2584 | 1000 | 20 |

| 0.6333 | 25.28 | 1.2592 | 1000 | 10 |

| 0.4667 | **36.84** | 1.2628 | 1000 | 5 |


---


### 4. IVFPQ (IVF with Product Quantization)


> **Observation:** IVFPQ compresses vectors, which often allows for very high speed (QPS) but can sacrifice accuracy (Recall) if not tuned perfectly. In this dataset, recall is low (~3%), but the QPS is superior to other methods. This method is optimized for speed/memory, not precision on small samples.


| Recall | QPS | Avg Neighbor Dist | Num Clusters | $Ks$ (Sub-quantizers) | $M$ (Centroids) | `nprobe` |

| :--- | :--- | :--- | :--- | :--- | :--- | :--- |

| 0.0333 | 30.83 | 1.3185 | 1000 | 256 | 8 | 10 |

| 0.0333 | 118.89 | 0.9275 | 16 | 4 | 2 | - |

| 0.0333 | 77.02 | 0.9275 | 16 | 4 | 5 | - |

| 0.0167 | **139.55** | 0.9240 | 32 | 4 | 2 | - |

| 0.0167 | 72.68 | 0.9239 | 32 | 4 | 5 | - |




### 5. Neural Network Hyperparameter Search


> **Observation:** The choice of **Activation Function** is the single most critical factor here. **ReLU** consistently produces the top results (60%+ accuracy), while **Sigmoid** results in model collapse (accuracy ~6.56%, likely stuck at random chance).

>

> Additionally, larger **Batch Sizes** (256) and wider layers (Nodes > 300) tend to perform better than smaller, narrower configurations.


| Value (Accuracy) | Activation | Learning Rate | Batch Size | Nodes | Layers | Dropout |

| :--- | :--- | :--- | :--- | :--- | :--- | :--- |

| **63.13** | `relu` | 4.35e-3 | 256 | 512 | 2 | 0.408 |

| **62.25** | `relu` | 4.77e-3 | 256 | 448 | 2 | 0.459 |

| 62.11 | `relu` | 9.70e-3 | 256 | 416 | 2 | 0.328 |

| 61.88 | `relu` | 3.49e-3 | 256 | 480 | 2 | 0.433 |

| 59.81 | `relu` | 3.70e-4 | 32 | 352 | 2 | 0.254 |

| 56.36 | `tanh` | 8.18e-5 | 256 | 128 | 2 | 0.183 |

| 48.64 | `tanh` | 1.33e-4 | 32 | 128 | 1 | 0.433 |

| 32.84 | `tanh` | 1.57e-5 | 64 | 96 | 3 | 0.220 |

| 6.56 | `sigmoid` | 9.26e-3 | 256 | 512 | 3 | 0.301 |

| 6.56 | `sigmoid` | 7.15e-5 | 64 | 192 | 4 | 0.120 | 




# Search Protein



This Python script serves as the main driver for the experimental pipeline. It automates the benchmarking process by:

    Reading Ground Truth: It parses the pre-computed BLAST output to establish a baseline for accuracy.

    Executing Algorithms: It iterates through every query protein, running the selected Approximate Nearest Neighbor (ANN) method (e.g., LSH, Hypercube).

    Benchmarking: It compares the ANN results against the BLAST ground truth to calculate key performance metrics, including Recall@N and Average Search Time.

    Logging: All metrics and raw results are structured and written to an output log file for further analysis.

# Bio Analysis


This script acts as a biological validator for the computational results. Its primary goal is to investigate "hidden gems"—candidate proteins identified by the embedding model that BLAST failed to detect.

    Filtering: It isolates candidate pairs where the embedding distance is low but the BLAST identity is non-existent.

    Data Enrichment: For every unique pair, it queries the UniProt API to fetch rich biological metadata, including Pfam domains (structure), Gene Ontology (GO) terms (function), and Keywords.

    Comparative Analysis: It calculates the intersection of these terms between the query and the neighbor.

    Reporting: It generates a detailed human-readable report (detailed_bio_report.txt), automatically categorizing pairs as "Strong Homologs" (shared structure) or "Likely Homologs" (shared function), providing evidence for the embedding model's ability to find remote homology.