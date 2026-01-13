import argparse
import sys
import os
import time
import math
import random 
import numpy as np 
import pandas as pd 
import Algos.ann_algos
# -------------------------------
# Utility Loaders
# -------------------------------

def load_embeddings(npz_path):
    """
    Load protein embeddings from a .npz file as a dict {protein_id: vector}.
    """
    data = np.load(npz_path, allow_pickle=True)
    embeddings = {k: data[k] for k in data}
    return embeddings


def load_fasta(path):
    """
    Load query protein sequences from FASTA.
    Returns {id: sequence}
    """
    queries = {}
    current_id = None
    current_seq = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    queries[current_id] = "".join(current_seq)
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id:
            queries[current_id] = "".join(current_seq)

    return queries


# -------------------------------
# Search Method Stubs
# -------------------------------

def lsh_search(vectors, query_id):
    time.sleep(0.01)
    return random.sample(list(vectors.keys()), k=min(5, len(vectors)))


def hypercube_search(vectors, query_id):
    time.sleep(0.01)
    return random.sample(list(vectors.keys()), k=min(5, len(vectors)))


def neural_search(vectors, query_id):
    time.sleep(0.01)
    return random.sample(list(vectors.keys()), k=min(5, len(vectors)))
def ivf_search(vectors, query_id):
    time.sleep(0.01)
    return random.sample(list(vectors.keys()), k=min(5, len(vectors)))


# -------------------------------
# Method Dispatcher
# -------------------------------

def run_search_method(method, vectors, query_id):
    if method == "lsh":
        return lsh_search(vectors, query_id)
    elif method == "hypercube":
        return hypercube_search(vectors, query_id)
    elif method == "neural":
        return neural_search(vectors, query_id)
    elif method == "ivf":
        return ivf_search(vectors, query_id)
    elif method == "all":
        results = {}
        for m in ["lsh", "hypercube", "neural", "ivf"]:
            results[m] = run_search_method(m, vectors, query_id)
        return results
    else:
        raise ValueError(f"Unknown method: {method}")


# -------------------------------
# Main Function
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Protein Search Benchmark")
    parser.add_argument("-d", "--database", required=True, help="Protein vectors file (.dat)")
    parser.add_argument("-q", "--query", required=True, help="Query protein sequences (.fasta)")
    parser.add_argument("-o", "--output", required=True, help="Output results file")
    parser.add_argument(
        "-method", "--method",
        choices=["all", "lsh", "hypercube", "neural", "ivf"],
        default="all",
        help="Search method to use"
    )

    args = parser.parse_args()

    print(f"Loading protein vectors from {args.database}...")
    vectors = load_embeddings(args.database)
    print(f"Loaded {len(vectors)} protein vectors.")
    print(len(vectors))
    print(list(vectors.items())[0])
        
    print(f"Loading queries from {args.query}...")
    queries = load_fasta(args.query)
    print(f"Loaded {len(queries)} queries.")

    print(f"Running search using method: {args.method} ...")
    results = {}

    for qid in queries:
        res = run_search_method(args.method, vectors, qid)
        results[qid] = res

    print(f"Writing results to {args.output} ...")
    with open(args.output, "w") as f:
        for qid, res in results.items():
            if args.method == "all":
                f.write(f">{qid}\n")
                for m, hits in res.items():
                    f.write(f"{m}: {' '.join(hits)}\n")
            else:
                f.write(f">{qid} {' '.join(res)}\n")

    print("Done.")
    

if __name__ == "__main__":
    main()
