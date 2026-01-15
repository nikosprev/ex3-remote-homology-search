import argparse
import time
import random
import numpy as np
from Algos import ann_algos
import torch
from scipy.spatial.distance import cdist
from utils.load_model import load_model

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

# -------------------------------
# Search Wrappers
# -------------------------------

class LSHSearch:
    def __init__(self, vectors, hashTable_size=10, num_tables=2, HashFunction_size=1, w=5.0, seed=1):
        self.ids = list(vectors.keys())
        vec_dim = len(next(iter(vectors.values())))
        self.lsh = ann_algos.LSHFloat(
            hashTable_size=hashTable_size,
            num_tables=num_tables,
            HashFunction_size=HashFunction_size,
            w=w,
            vec_dim=vec_dim,
            seed=seed
        )
        for vec in vectors.values():
            self.lsh.insert_to_hashTables(vec)

    def query(self, query_vec, k=5, **kwargs):
        neighbors = self.lsh.returnANN(query_vec, k)
        return [(self.ids[n.idx], n.distance) for n in neighbors]


class HypercubeSearch:
    def __init__(self, vectors, k_proj=2, w=5.0, seed=42):
        self.ids = list(vectors.keys())
        vecs = list(vectors.values())
        vec_dim = len(vecs[0])
        self.cube = ann_algos.HyperCubeFloat(vecs, k_proj=k_proj, w=w, vec_dim=vec_dim, seed=seed)

    def query(self, query_vec, k=5, M=10, probe=2, **kwargs):
        neighbors = self.cube.returnANN(query_vec, M=M, k=k, probe=probe)
        return [(self.ids[n.idx], n.distance) for n in neighbors]


class IVFFlatSearch:
    def __init__(self, vectors, num_clusters=32, kmeans_iters=10, seed=42):
        self.ids = list(vectors.keys())
        vecs = list(vectors.values())
        vec_dim = len(vecs[0])
        self.ivf = ann_algos.IVFFlatFloat(num_clusters=num_clusters, vec_dim=vec_dim, kmeans_iters=kmeans_iters)
        self.ivf.add_vectors(vecs)
        self.ivf.train(seed=seed)

    def query(self, query_vec, k=5, nprobe=5, **kwargs):
        neighbors = self.ivf.query(query_vec, k=k, nprobe=nprobe)
        return [(self.ids[n.idx], n.distance) for n in neighbors]


class IVFPQSearch:
    def __init__(self, vectors, num_coarse_clusters=32, M=8, Ks=256, kmeans_iters=10, seed=42):
        self.ids = list(vectors.keys())
        vecs = list(vectors.values())
        vec_dim = len(vecs[0])
        self.pq = ann_algos.IVFPQFloat(
            num_coarse_clusters=num_coarse_clusters,
            vec_dim=vec_dim,
            M=M,
            Ks=Ks,
            kmeans_iters=kmeans_iters
        )
        self.pq.add_vectors(vecs)
        self.pq.train(seed=seed)

    def query(self, query_vec, k=5, nprobe=5, **kwargs):
        neighbors = self.pq.query(query_vec, k=k, nprobe=nprobe)
        return [(self.ids[n.idx], n.distance) for n in neighbors]


class NeuralSearch:
    def __init__(self, vectors, model_path, device="cpu"):
        self.ids = list(vectors.keys())
        self.vectors = np.vstack(list(vectors.values()))
        self.device = device
        self.model_path = model_path
        input_dim = self.vectors.shape[1]

        # Infer num_classes from checkpoint OR use max block id later
        # Safer: load once, build index, infer classes
        checkpoint = torch.load(model_path, map_location=device)
        if "architecture" in checkpoint:
            num_classes = checkpoint["architecture"]["output_size"]
        else:
            # fallback: must match training
            num_classes = 64  # <-- SAME m AS TRAINING

        self.model = load_model(
            model_path=self.model_path,
        device=device
        )

        self.inverted_index = self._build_inverted_index()

    def _build_inverted_index(self):
        """
        Uses the trained NN to assign each database vector to a block.
        """
        inverted = {}

        with torch.no_grad():
            for idx, vec in enumerate(self.vectors):
                v = torch.from_numpy(vec).float().unsqueeze(0).to(self.device)
                logits = self.model(v)
                block = torch.argmax(logits, dim=1).item()

                if block not in inverted:
                    inverted[block] = []
                inverted[block].append(idx)

        return inverted

    def _predict_blocks(self, query_vec, probes=3):
        q = torch.from_numpy(query_vec).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(q)
            probs = torch.softmax(logits, dim=1)

        return torch.topk(probs, probes).indices.cpu().numpy()[0]

    def query(self, query_vec, k=5, probes=3, **kwargs):
        blocks = self._predict_blocks(query_vec, probes)

        candidate_ids = []
        for b in blocks:
            candidate_ids.extend(self.inverted_index.get(int(b), []))

        if len(candidate_ids) == 0:
            return []

        candidate_ids = list(set(candidate_ids))
        candidate_vecs = self.vectors[candidate_ids]

        dists = cdist(
            query_vec.reshape(1, -1),
            candidate_vecs,
            metric="euclidean"
        )[0]

        topk = np.argsort(dists)[:k]

        return [
            (self.ids[candidate_ids[i]], float(dists[i]))
            for i in topk
        ]




# -------------------------------
# Dispatcher
# -------------------------------

def get_searcher(method, vectors, args):
    if method == "lsh":
        return LSHSearch(
            vectors,
            hashTable_size=args.lsh_table_size,
            num_tables=args.lsh_num_tables,
            HashFunction_size=args.lsh_hash_size,
            w=args.lsh_w,
            seed=args.seed
        )
    elif method == "hypercube":
        return HypercubeSearch(
            vectors,
            k_proj=args.hypercube_k_proj,
            w=args.hypercube_w,
            seed=args.seed
        )
    elif method == "ivf":
        return IVFFlatSearch(
            vectors,
            num_clusters=args.ivf_num_clusters,
            kmeans_iters=args.ivf_iters,
            seed=args.seed
        )
    elif method == "ivfpq":
        return IVFPQSearch(
            vectors,
            num_coarse_clusters=args.ivfpq_num_clusters,
            M=args.ivfpq_M,
            Ks=args.ivfpq_Ks,
            kmeans_iters=args.ivfpq_iters,
            seed=args.seed
        )
    elif method == "neural":
        return NeuralSearch(
            vectors,
            model_path=args.neural_model,
            device="cpu"
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# -------------------------------
# Main Function
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Protein Search Benchmark")
    parser.add_argument("-d", "--database", required=True)
    parser.add_argument("-q", "--query", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-method", "--method", choices=["all", "lsh", "hypercube", "ivf", "ivfpq", "neural"], default="all")
    parser.add_argument("--seed", type=int, default=42)

    # LSH parameters
    parser.add_argument("--lsh_table_size", type=int, default=10)
    parser.add_argument("--lsh_num_tables", type=int, default=2)
    parser.add_argument("--lsh_hash_size", type=int, default=1)
    parser.add_argument("--lsh_w", type=float, default=5.0)

    # Hypercube parameters
    parser.add_argument("--hypercube_k_proj", type=int, default=2)
    parser.add_argument("--hypercube_w", type=float, default=5.0)
    parser.add_argument("--hypercube_M", type=int, default=10)
    parser.add_argument("--hypercube_probe", type=int, default=2)

    # IVF parameters
    parser.add_argument("--ivf_num_clusters", type=int, default=32)
    parser.add_argument("--ivf_iters", type=int, default=10)
    parser.add_argument("--ivf_nprobe", type=int, default=5)

    # IVFPQ parameters
    parser.add_argument("--ivfpq_num_clusters", type=int, default=32)
    parser.add_argument("--ivfpq_M", type=int, default=8)
    parser.add_argument("--ivfpq_Ks", type=int, default=256)
    parser.add_argument("--ivfpq_iters", type=int, default=10)
    parser.add_argument("--ivfpq_nprobe", type=int, default=5)
    # Neural parameters
    parser.add_argument( "--neural_model",type=str,required=False,help="Path to trained neural NN model (.pth)")

    args = parser.parse_args()

    # -------------------------------
    # Load data
    # -------------------------------
    print(f"Loading database vectors from {args.database} ...")
    vectors = load_embeddings(args.database)
    print(f"Loaded {len(vectors)} vectors.")

    print(f"Loading queries from {args.query} ...")
    queries = load_embeddings(args.query)
    print(f"Loaded {len(queries)} queries.")

    # -------------------------------
    # Build all indices once
    # -------------------------------
    searchers = {}
    if args.method == "all":
        methods = ["lsh", "hypercube", "ivf", "ivfpq", "neural"]
    else:
        methods = [args.method]

    for m in methods:
        print(f"Building {m} index ...")
        searchers[m] = get_searcher(m, vectors, args)

    # -------------------------------
    # Run queries
    # -------------------------------
    results = {}
    for qid, qvec in queries.items():
        res = {}
        for m in methods:
            nprobe = getattr(args, f"{m}_nprobe", 5)
            hits = searchers[m].query(qvec, k=5, nprobe=nprobe, M=getattr(args, f"{m}_M", 10), probe=getattr(args, f"{m}_probe", 2))
            res[m] = hits
        results[qid] = res

    # -------------------------------
    # Write output
    # -------------------------------
    print(f"Writing results to {args.output} ...")
    with open(args.output, "w") as f:
        for qid, res_dict in results.items():
            f.write(f">{qid}\n")
            for m, hits in res_dict.items():
                formatted = " ".join(f"{pid}:{dist:.4f}" for pid, dist in hits)
                f.write(f"{m}: {formatted}\n")

    print("Done.")


if __name__ == "__main__":
    main()
