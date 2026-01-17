import argparse
import time
import random
import os
import numpy as np
from Algos import ann_algos
import torch
from scipy.spatial.distance import cdist
from utils.load_model import load_model

# -------------------------------
# Utility Loaders
# -------------------------------

def load_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if 'embeddings' in data and 'ids' in data:
        embeddings_array = data['embeddings']
        ids_array = data['ids']
        if embeddings_array.ndim == 1:
            return {str(ids_array): embeddings_array}
        else:
            embeddings = {}
            for i, protein_id in enumerate(ids_array):
                embeddings[str(protein_id)] = embeddings_array[i]
            return embeddings
    else:
        embeddings = {}
        for k in data:
            arr = data[k]
            if arr.ndim == 1:
                if 64 <= arr.size <= 4096:
                    embeddings[k] = arr
            elif arr.ndim == 2:
                for i, row in enumerate(arr):
                    embeddings[f"{k}_{i}"] = row
        return embeddings

# -------------------------------
# Search Wrappers
# -------------------------------

class LSHSearch:
    def __init__(self, vectors, hashTable_size=8192, num_tables=8, HashFunction_size=6, w=1.0, seed=1):
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
    def __init__(self, vectors, k_proj=12, w=1.5, seed=42):
        self.ids = list(vectors.keys())
        vecs = list(vectors.values())
        vec_dim = len(vecs[0])
        self.cube = ann_algos.HyperCubeFloat(vecs, k_proj=k_proj, w=w, vec_dim=vec_dim, seed=seed)

    def query(self, query_vec, k=5, M=5000, probe=2, **kwargs):
        neighbors = self.cube.returnANN(query_vec, M=M, k=k, probe=probe)
        return [(self.ids[n.idx], n.distance) for n in neighbors]


class IVFFlatSearch:
    def __init__(self, vectors, num_clusters=1000, kmeans_iters=10, seed=42):
        self.ids = list(vectors.keys())
        vecs = list(vectors.values())
        vec_dim = len(vecs[0])
        self.ivf = ann_algos.IVFFlatFloat(num_clusters=num_clusters, vec_dim=vec_dim, kmeans_iters=kmeans_iters)
        self.ivf.add_vectors(vecs)
        self.ivf.train(seed=seed)

    def query(self, query_vec, k=5, nprobe=50, **kwargs):
        neighbors = self.ivf.query(query_vec, k=k, nprobe=nprobe)
        return [(self.ids[n.idx], n.distance) for n in neighbors]


class IVFPQSearch:
    def __init__(self, vectors, num_coarse_clusters=1000, M=16, Ks=256, kmeans_iters=10, seed=42):
        self.ids = list(vectors.keys())
        # Convert to matrix for fast re-ranking
        self.vector_matrix = np.array(list(vectors.values())) 
        vec_dim = self.vector_matrix.shape[1]
        
        self.pq = ann_algos.IVFPQFloat(
            num_coarse_clusters=num_coarse_clusters,
            vec_dim=vec_dim,
            M=M,
            Ks=Ks,
            kmeans_iters=kmeans_iters
        )
        self.pq.add_vectors(self.vector_matrix)
        self.pq.train(seed=seed)

    def query(self, query_vec, k=5, nprobe=50, **kwargs):
        # 1. Get Approx Neighbors from C++ (Returns IDs and Approx Dist)
        neighbors = self.pq.query(query_vec, k=k, nprobe=nprobe)
        
        if not neighbors:
            return []

        # 2. Extract indices
        indices = [n.idx for n in neighbors]
        
        # 3. Vectorized True Distance Calculation (Fast Re-ranking)
        candidates = self.vector_matrix[indices]
        
        # Calculate Euclidean distance for candidates
        dists = np.linalg.norm(candidates - query_vec, axis=1)
        
        # 4. Zip with IDs and return
        results = []
        for i, idx in enumerate(indices):
            results.append((self.ids[idx], float(dists[i])))
            
        # Optional: Sort by true distance
        results.sort(key=lambda x: x[1])
        return results


class NeuralSearch:
    def __init__(self, vectors, model_path, device="cpu"):
        if model_path is None:
            raise ValueError("model_path cannot be None.")
        
        self.ids = list(vectors.keys())
        self.vectors = np.vstack(list(vectors.values()))
        self.device = device
        self.model_path = model_path
        self.model = load_model(model_path=self.model_path, device=device)
        self.inverted_index = self._build_inverted_index()

    def _build_inverted_index(self):
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
        dists = cdist(query_vec.reshape(1, -1), candidate_vecs, metric="euclidean")[0]
        topk = np.argsort(dists)[:k]
        return [(self.ids[candidate_ids[i]], float(dists[i])) for i in topk]

# -------------------------------
# Dispatcher
# -------------------------------

def get_searcher(method, vectors, args):
    if method == "lsh":
        return LSHSearch(vectors, hashTable_size=args.lsh_table_size, num_tables=args.lsh_num_tables, HashFunction_size=args.lsh_hash_size, w=args.lsh_w, seed=args.seed)
    elif method == "hypercube":
        return HypercubeSearch(vectors, k_proj=args.hypercube_k_proj, w=args.hypercube_w, seed=args.seed)
    elif method == "ivf":
        return IVFFlatSearch(vectors, num_clusters=args.ivf_num_clusters, kmeans_iters=args.ivf_iters, seed=args.seed)
    elif method == "ivfpq":
        return IVFPQSearch(vectors, num_coarse_clusters=args.ivfpq_num_clusters, M=args.ivfpq_M, Ks=args.ivfpq_Ks, kmeans_iters=args.ivfpq_iters, seed=args.seed)
    elif method == "neural":
        return NeuralSearch(vectors, model_path=args.neural_model, device="cpu")
    else:
        raise ValueError(f"Unknown method: {method}")


# -------------------------------
# Main Function
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--database", required=True)
    parser.add_argument("-q", "--query", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-method", "--method", choices=["all", "lsh", "hypercube", "ivf", "ivfpq", "neural"], default="all")
    parser.add_argument("--seed", type=int, default=42)

    # LSH parameters
    parser.add_argument("--lsh_table_size", type=int, default=8192)
    parser.add_argument("--lsh_num_tables", type=int, default=8)
    parser.add_argument("--lsh_hash_size", type=int, default=6)
    parser.add_argument("--lsh_w", type=float, default=1.0)

    # Hypercube parameters
    parser.add_argument("--hypercube_k_proj", type=int, default=12)
    parser.add_argument("--hypercube_w", type=float, default=1.5)
    parser.add_argument("--hypercube_M", type=int, default=5000)
    parser.add_argument("--hypercube_probe", type=int, default=2)

    # IVF parameters
    parser.add_argument("--ivf_num_clusters", type=int, default=1000)
    parser.add_argument("--ivf_iters", type=int, default=10)
    parser.add_argument("--ivf_nprobe", type=int, default=50)

    # IVFPQ parameters
    parser.add_argument("--ivfpq_num_clusters", type=int, default=1000)
    parser.add_argument("--ivfpq_M", type=int, default=16)
    parser.add_argument("--ivfpq_Ks", type=int, default=256)
    parser.add_argument("--ivfpq_iters", type=int, default=10)
    parser.add_argument("--ivfpq_nprobe", type=int, default=50)
    
    # Neural parameters
    parser.add_argument("--neural_model", type=str, required=False)

    args = parser.parse_args()

    # Load data
    print(f"Loading data...")
    vectors = load_embeddings(args.database)
    queries = load_embeddings(args.query)

    # Build indices
    searchers = {}
    if args.method == "all":
        methods = ["lsh", "hypercube", "ivf", "ivfpq", "neural"]
    else:
        methods = [args.method]

    print("Building indices...")
    for m in methods:
        searchers[m] = get_searcher(m, vectors, args)

    # -------------------------------
    # Run queries & Measure Time
    # -------------------------------
    print("Running queries...")
    results = {}
    
    start_time = time.time()
    
    for qid, qvec in queries.items():
        res = {}
        for m in methods:
            # Dispatch parameters dynamically
            nprobe = getattr(args, f"{m}_nprobe", 50)
            M_val = getattr(args, f"{m}_M", 5000)
            probe_val = getattr(args, f"{m}_probe", 2)
            
            hits = searchers[m].query(qvec, k=5, nprobe=nprobe, M=M_val, probe=probe_val)
            res[m] = hits
        results[qid] = res
        
    end_time = time.time()
    total_query_time = end_time - start_time
    
    num_queries = len(queries)
    qps = num_queries / total_query_time if total_query_time > 0 else 0.0
    print(f"Query phase finished in {total_query_time:.4f}s. QPS: {qps:.2f}")

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
        
        # WRITE QPS AT THE END
        f.write(f"QPS: {qps:.4f}\n")

    print("Done.")

if __name__ == "__main__":
    main()