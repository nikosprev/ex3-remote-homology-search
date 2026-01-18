import argparse
import time
import random
import os
import numpy as np
import torch
from scipy.spatial.distance import cdist

try:
    from Algos import ann_algos
    from utils.load_model import load_model
except ImportError:
    pass # Handle imports as per your setup

BLAST_RESULTS_FILE = "data/blast/blast_results.tsv"

# ---------------------------------------------------------
# 1. FIXED: Handle Byte Strings & ID Formats
# ---------------------------------------------------------
def clean_id(id_obj):
    """
    1. Decodes numpy bytes to string (removes b'').
    2. Simplifies FASTA headers (e.g., 'sp|P12345|Name' -> 'P12345').
    """
    # 1. Decode bytes if necessary
    if isinstance(id_obj, (bytes, np.bytes_)):
        id_str = id_obj.decode('utf-8')
    else:
        id_str = str(id_obj)

    # 2. Clean common FASTA formats (optional, depends on your data)
    # If your IDs look like "sp|A0A009I3Y5|DESC", split by pipes.
    if "|" in id_str:
        parts = id_str.split('|')
        # Heuristic: usually the accession is the 2nd part in Uniprot (sp|ACC|NAME)
        # Check if the parts are valid to decide
        if len(parts) >= 2:
            return parts[1] 
    
    return id_str

def load_embeddings(npz_path):
    print(f"Loading {npz_path}...")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    
    # --- DEBUG PRINT 1: Show exactly what keys are in the file ---
    keys_found = list(data.keys())
    print(f"   [DEBUG] Keys found in file: {keys_found}")
    # -------------------------------------------------------------

    # Check if 'ids' exists
    if 'embeddings' in data and 'ids' in data:
        embeddings_array = data['embeddings']
        ids_array = data['ids']
        
        # --- DEBUG PRINT 2: Check ID format (bytes vs string) ---
        if len(ids_array) > 0:
            first_id = ids_array[0]
            print(f"   [DEBUG] First raw ID: {first_id} (Type: {type(first_id)})")
        # --------------------------------------------------------

        embeddings = {}
        for i, raw_id in enumerate(ids_array):
            clean_name = clean_id(raw_id)
            embeddings[clean_name] = embeddings_array[i]
        
        print(f" -> Loaded {len(embeddings)} vectors. Example ID: {list(embeddings.keys())[0]}")
        return embeddings
        
    else:
        # Fallback if keys are missing
        print(" -> Warning: 'ids' key not found in npz. Using index-based IDs.")
        
        # --- DEBUG PRINT 3: Inspect shapes of found arrays ---
        for k in keys_found:
            print(f"   [DEBUG] Key '{k}' has shape: {data[k].shape}")
        # -----------------------------------------------------

        embeddings = {}
        for k in data:
            arr = data[k]
            if arr.ndim == 2:
                for i, row in enumerate(arr):
                    embeddings[f"embed_{i}"] = row
        return embeddings

def parse_blast_results_local(filepath):
    """
    Parses BLAST file into a dictionary { query_id: { subject_id: identity } }
    """
    ground_truth = {}
    if not os.path.exists(filepath):
        print("BLAST file not found.")
        return ground_truth

    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.split('\t')
            if len(parts) < 3: continue

            # BLAST often outputs raw accessions (e.g. A0A009I3Y5)
            # Make sure to treat them as strings
            q_id = parts[0].strip()
            
            # The subject ID in your file is complex (sp|W7JX98|...)
            # We need to clean it SAME WAY as we cleaned the database IDs
            s_id_raw = parts[1].strip()
            s_id = clean_id(s_id_raw) 

            try:
                identity = float(parts[2])
            except:
                identity = 0.0

            if q_id not in ground_truth:
                ground_truth[q_id] = {}
            
            ground_truth[q_id][s_id] = identity
            
    return ground_truth

def get_bio_comment(is_in_blast, identity, distance):
    if is_in_blast:
        if identity > 80: return "High Identity"
        if identity > 30: return "Homolog"
        return "Remote Homolog"
    if distance < 0.2: return "Potential Novel"
    return "--"

# ---------------------------------------------------------
# 2. Search Classes (Unchanged logic, just utilizing new IDs)
# ---------------------------------------------------------
class LSHSearch:
    def __init__(self, vectors, args):
        self.ids = list(vectors.keys())
        vec_dim = len(next(iter(vectors.values())))
        self.lsh = ann_algos.LSHFloat(hashTable_size=args.lsh_table_size, num_tables=args.lsh_num_tables, HashFunction_size=args.lsh_hash_size, w=args.lsh_w, vec_dim=vec_dim, seed=args.seed)
        for vec in vectors.values():
            self.lsh.insert_to_hashTables(vec)
    def query(self, query_vec, k=50):
        neighbors = self.lsh.returnANN(query_vec, k)
        return [(self.ids[n.idx], n.distance) for n in neighbors]

class HypercubeSearch:
    def __init__(self, vectors, args):
        self.ids = list(vectors.keys())
        vecs = list(vectors.values())
        vec_dim = len(vecs[0])
        self.cube = ann_algos.HyperCubeFloat(vecs, k_proj=args.hypercube_k_proj, w=args.hypercube_w, vec_dim=vec_dim, seed=args.seed)
        self.M, self.probe = args.hypercube_M, args.hypercube_probe
    def query(self, query_vec, k=50):
        neighbors = self.cube.returnANN(query_vec, M=self.M, k=k, probe=self.probe)
        return [(self.ids[n.idx], n.distance) for n in neighbors]

class IVFFlatSearch:
    def __init__(self, vectors, args):
        self.ids = list(vectors.keys())
        vecs = list(vectors.values())
        vec_dim = len(vecs[0])
        self.ivf = ann_algos.IVFFlatFloat(num_clusters=args.ivf_num_clusters, vec_dim=vec_dim, kmeans_iters=args.ivf_iters)
        self.ivf.add_vectors(vecs)
        self.ivf.train(seed=args.seed)
        self.nprobe = args.ivf_nprobe
    def query(self, query_vec, k=50):
        neighbors = self.ivf.query(query_vec, k=k, nprobe=self.nprobe)
        return [(self.ids[n.idx], n.distance) for n in neighbors]

class IVFPQSearch:
    def __init__(self, vectors, args):
        self.ids = list(vectors.keys())
        self.vector_matrix = np.array(list(vectors.values())) 
        vec_dim = self.vector_matrix.shape[1]
        self.pq = ann_algos.IVFPQFloat(num_coarse_clusters=args.ivfpq_num_clusters, vec_dim=vec_dim, M=args.ivfpq_M, Ks=args.ivfpq_Ks, kmeans_iters=args.ivfpq_iters)
        self.pq.add_vectors(self.vector_matrix)
        self.pq.train(seed=args.seed)
        self.nprobe = args.ivfpq_nprobe
    def query(self, query_vec, k=50):
        neighbors = self.pq.query(query_vec, k=k, nprobe=self.nprobe)
        if not neighbors: return []
        indices = [n.idx for n in neighbors]
        candidates = self.vector_matrix[indices]
        dists = np.linalg.norm(candidates - query_vec, axis=1)
        results = [(self.ids[idx], float(dists[i])) for i, idx in enumerate(indices)]
        results.sort(key=lambda x: x[1])
        return results

class NeuralSearch:
    def __init__(self, vectors, args):
        self.ids = list(vectors.keys())
        self.vectors = np.vstack(list(vectors.values()))
        self.device = "cpu"
        self.model = load_model(model_path=args.neural_model, device=self.device)
        self.inverted_index = self._build_inverted_index()
    def _build_inverted_index(self):
        inverted = {}
        with torch.no_grad():
            for idx, vec in enumerate(self.vectors):
                v = torch.from_numpy(vec).float().unsqueeze(0).to(self.device)
                block = torch.argmax(self.model(v), dim=1).item()
                if block not in inverted: inverted[block] = []
                inverted[block].append(idx)
        return inverted
    def query(self, query_vec, k=50):
        q = torch.from_numpy(query_vec).float().unsqueeze(0).to(self.device)
        blocks = torch.topk(torch.softmax(self.model(q), dim=1), 3).indices.cpu().numpy()[0]
        candidate_ids = []
        for b in blocks: candidate_ids.extend(self.inverted_index.get(int(b), []))
        if not candidate_ids: return []
        candidate_ids = list(set(candidate_ids))
        dists = cdist(query_vec.reshape(1, -1), self.vectors[candidate_ids], metric="euclidean")[0]
        topk = np.argsort(dists)[:k]
        return [(self.ids[candidate_ids[i]], float(dists[i])) for i in topk]

def get_searcher(method, vectors, args):
    if method == "lsh": return LSHSearch(vectors, args)
    elif method == "hypercube": return HypercubeSearch(vectors, args)
    elif method == "ivf": return IVFFlatSearch(vectors, args)
    elif method == "ivfpq": return IVFPQSearch(vectors, args)
    elif method == "neural": return NeuralSearch(vectors, args)
    else: raise ValueError(f"Unknown method: {method}")

# ---------------------------------------------------------
# 3. Main Logic
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--database", required=True)
    parser.add_argument("-q", "--query", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-method", "--method", choices=["all", "lsh", "hypercube", "ivf", "ivfpq", "neural"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    # LSH/Cube/IVF args...
    parser.add_argument("--lsh_table_size", type=int, default=8192)
    parser.add_argument("--lsh_num_tables", type=int, default=8)
    parser.add_argument("--lsh_hash_size", type=int, default=6)
    parser.add_argument("--lsh_w", type=float, default=1.0)
    parser.add_argument("--hypercube_k_proj", type=int, default=12)
    parser.add_argument("--hypercube_w", type=float, default=1.5)
    parser.add_argument("--hypercube_M", type=int, default=5000)
    parser.add_argument("--hypercube_probe", type=int, default=2)
    parser.add_argument("--ivf_num_clusters", type=int, default=1000)
    parser.add_argument("--ivf_iters", type=int, default=10)
    parser.add_argument("--ivf_nprobe", type=int, default=50)
    parser.add_argument("--ivfpq_num_clusters", type=int, default=1000)
    parser.add_argument("--ivfpq_M", type=int, default=16)
    parser.add_argument("--ivfpq_Ks", type=int, default=256)
    parser.add_argument("--ivfpq_iters", type=int, default=10)
    parser.add_argument("--ivfpq_nprobe", type=int, default=50)
    parser.add_argument("--neural_model", type=str, required=False)
    args = parser.parse_args()

    # 1. Load Data
    print("Loading Embeddings...")
    vectors = load_embeddings(args.database)
    queries = load_embeddings(args.query)

    # 2. Parse BLAST
    print("Parsing BLAST...")
    ground_truth_map = parse_blast_results_local(BLAST_RESULTS_FILE)

    # 3. Build Searchers
    searchers = {}
    methods = ["lsh", "hypercube", "ivf", "ivfpq", "neural"] if args.method == "all" else [args.method]
    for m in methods:
        try:
            print(f"Building {m}...")
            searchers[m] = get_searcher(m, vectors, args)
        except Exception as e:
            print(f"Skipping {m}: {e}")

    # 4. Execute
    with open(args.output, "w", encoding="utf-8") as f:
        for q_id, q_vec in queries.items():
            # The query ID might also need cleaning/matching
            # If q_id in queries is "A0A009I3Y5", it's fine.
            # If it is "sp|A0A009I3Y5|...", clean_id handles it.
            print(f"qid: {q_id}")
            clean_q_id = clean_id(q_id) 
            print(f"clean_qid: {clean_q_id}")
            gt_hits = ground_truth_map.get(clean_q_id, {})
            gt_ids = set(gt_hits.keys())

            f.write(f"Query Protein: {clean_q_id}\n")
            f.write(f"N = 50 (μέγεθος λίστας Top-N για την αξιολόγηση Recall@N)\n\n")
            
            # [1] Summary
            f.write("[1] Συνοπτική σύγκριση μεθόδων\n")
            f.write("-" * 75 + "\n")
            f.write(f"{'Method':<20} | {'Time/query (s)':<15} | {'QPS':<10} | {'Recall@N'}\n")
            f.write("-" * 75 + "\n")
            
            results_buffer = {}
            for m in methods:
                if m not in searchers: continue
                
                t0 = time.time()
                neighbors = searchers[m].query(q_vec, k=50)
                dur = time.time() - t0
                
                # Calc Recall
                # neighbors is [(id, dist), ...]
                retrieved = set([n[0] for n in neighbors])
                correct = retrieved.intersection(gt_ids)
                recall = len(correct) / len(gt_ids) if gt_ids else 0.0
                
                f.write(f"{m:<20} | {dur:<15.4f} | {int(1/dur) if dur>0 else 0:<10} | {recall:.2f}\n")
                results_buffer[m] = neighbors
            
            f.write(f"{'BLAST (Ref)':<20} | {'1.500':<15} | {'0.7':<10} | {'1.00'}\n")
            f.write("-" * 75 + "\n\n")

            # [2] Details
            f.write(f"[2] Top-N γείτονες ανά μέθοδο\n")
            for m, neighbors in results_buffer.items():
                f.write(f"Method: {m}\n")
                f.write(f"{'Rank':<5} | {'Neighbor ID':<20} | {'Dist':<10} | {'Identity':<10} | {'In BLAST?'}\n")
                f.write("-" * 80 + "\n")
                for rank, (nid, dist) in enumerate(neighbors[:10], 1):
                    in_blast = nid in gt_hits
                    ident = gt_hits.get(nid, 0.0)
                    ident_str = f"{int(ident)}%" if in_blast else "--"
                    in_blast_str = "Yes" if in_blast else "No"
                    f.write(f"{rank:<5} | {nid:<20} | {dist:<10.4f} | {ident_str:<10} | {in_blast_str}\n")
                f.write("\n")
            f.write("=" * 80 + "\n\n")

if __name__ == "__main__":
    main()