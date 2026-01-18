import argparse
import subprocess
import itertools
import time
import csv
import os
import statistics
import numpy as np
from scipy.spatial.distance import cdist
PARAM_GRIDS = {
    "lsh": {
        "lsh_hash_size": [2 , 4 , 6],
        "lsh_num_tables": [5 , 8],
        "lsh_w": [0.5 , 1.0,  2.0 , 4.0],
        "lsh_table_size": [16384] 
    },
    "hypercube": {
        "hypercube_k_proj": [9, 10  , 11 , 12 , 13 ,14 ],
        "hypercube_M": [100, 5000 , 10000],
        "hypercube_probe": [2 , 5 , 7 ,10],
        "hypercube_w": [1, 1.5 , 3.0]
    },
    "ivf": {
        "ivf_num_clusters": [1000],
        "ivf_nprobe": [5, 10, 20]
    },
    "ivfpq": {
        "ivfpq_num_clusters": [1000],
        "ivfpq_M": [8],
        "ivfpq_nprobe": [10],
        "ivfpq_Ks": [256]
    }
}


def load_embeddings(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    embeddings = {}
    if 'embeddings' in data and 'ids' in data:
        vecs = data['embeddings']
        ids = data['ids']
        if vecs.ndim == 1:
            return {str(ids): vecs}
        for i, pid in enumerate(ids):
            embeddings[str(pid)] = vecs[i]
    else:
        for k in data:
            arr = data[k]
            if arr.ndim == 1 and 64 <= arr.size <= 4096:
                embeddings[k] = arr
            elif arr.ndim == 2:
                for i, row in enumerate(arr):
                    embeddings[f"{k}_{i}"] = row
    return embeddings

def create_subset_file(input_path, output_path, ratio=0.5):
   
    print(f"Creating {ratio:.0%} subset of {input_path}...")
    
    data_dict = load_embeddings(input_path)
    
    all_ids = list(data_dict.keys())
    all_vecs = list(data_dict.values())
    
    limit = int(len(all_ids) * ratio)
    if limit == 0: 
        limit = len(all_ids) 
        
    subset_ids = np.array(all_ids[:limit])
    subset_vecs = np.array(all_vecs[:limit])
    
    np.savez(output_path, ids=subset_ids, embeddings=subset_vecs)
    print(f"Saved subset to {output_path} ({limit} items)")

def compute_ground_truth(db_path, query_path, k=5):
    print("Loading data for Ground Truth computation...")
    db_dict = load_embeddings(db_path)
    query_dict = load_embeddings(query_path)
    
    db_ids = list(db_dict.keys())
    db_vecs = np.array(list(db_dict.values()))
    
    q_ids = list(query_dict.keys())
    q_vecs = np.array(list(query_dict.values()))
    
    print(f"Computing exact neighbors for {len(q_vecs)} queries against {len(db_vecs)} items...")
    
    dists = cdist(q_vecs, db_vecs, metric='euclidean')
    
    ground_truth = {}
    for i, qid in enumerate(q_ids):
        nearest_indices = np.argsort(dists[i])[:k]
        true_neighbors = {db_ids[idx] for idx in nearest_indices}
        ground_truth[qid] = true_neighbors
        
    print("Ground Truth computed.\n")
    return ground_truth



def parse_results(output_path, method_name):
    distances = []
    predictions = {}
    qps = 0.0
    
    if not os.path.exists(output_path):
        return None, None, 0.0

    current_qid = None
    
    with open(output_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            if line.startswith("QPS:"):
                try:
                    qps = float(line.split(":")[1].strip())
                except ValueError:
                    qps = 0.0
                continue

            if line.startswith(">"):
                current_qid = line[1:]
                continue
                
            if line.startswith(f"{method_name}:") and current_qid:
                content = line.split(" ", 1)[1] if " " in line else ""
                if not content: 
                    predictions[current_qid] = set()
                    continue
                
                found_ids = set()
                items = content.split(" ")
                for item in items:
                    try:
                        pid, dist_str = item.rsplit(":", 1)
                        distances.append(float(dist_str))
                        found_ids.add(pid)
                    except ValueError:
                        continue
                predictions[current_qid] = found_ids

    return distances, predictions, qps

def calculate_metrics(distances, predictions, ground_truth):
    avg_dist = statistics.mean(distances) if distances else float('inf')
    
    recalls = []
    for qid, true_set in ground_truth.items():
        if qid not in predictions:
            recalls.append(0.0)
            continue
        found_set = predictions[qid]
        if not true_set:
            continue
        hits = len(found_set.intersection(true_set))
        recall = hits / len(true_set)
        recalls.append(recall)
        
    avg_recall = statistics.mean(recalls) if recalls else 0.0
    return avg_dist, avg_recall


def run_experiment(script_path, db_path, query_path, output_path, method, params, ground_truth):
    cmd = [
        "python", script_path,
        "-d", db_path,
        "-q", query_path,
        "-o", output_path,
        "--method", method
    ]

    for param_name, param_value in params.items():
        cmd.append(f"--{param_name}")
        cmd.append(str(param_value))

    print(f"Running {method} {params} ...", end=" ", flush=True)

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Failed.")
        print(result.stderr.strip())
        return None, None, None
    
    dists, preds, qps = parse_results(output_path, method)
    
    if preds is None:
         print("Error parsing output.")
         return None, None, None

    avg_dist, recall = calculate_metrics(dists, preds, ground_truth)
    
    print(f"Done (QPS: {qps:.1f} | Recall: {recall:.2%} | AvgDist: {avg_dist:.4f})")
    return qps, avg_dist, recall

def main():
    parser = argparse.ArgumentParser(description="Finetuning Runner")
    parser.add_argument("--script", default="./src/protein_search.py", help="Path to search script")
    parser.add_argument("-d", "--database", required=True)
    parser.add_argument("-q", "--query", required=True)
    parser.add_argument("--csv_out", default="finetuning_results_half.csv")
    parser.add_argument("--subset_ratio", type=float, default=0.5, help="Ratio of dataset to use")
    
    args = parser.parse_args()
    
    temp_output_file = "temp_tuning_results.txt"
    temp_db_file = "temp_subset_db.npz"
    temp_query_file = "temp_subset_query.npz"
    
    results_data = []

    if not os.path.exists(args.script):
        print(f"CRITICAL ERROR: '{args.script}' not found.")
        return

    try:
        create_subset_file(args.database, temp_db_file, ratio=args.subset_ratio)
        create_subset_file(args.query, temp_query_file, ratio=args.subset_ratio)

        ground_truth = compute_ground_truth(temp_db_file, temp_query_file, k=5)
    
        print(f"Starting finetuning on {args.subset_ratio*100}% of data...")

        for method, grid in PARAM_GRIDS.items():
            print(f"\n--- Tuning {method.upper()} ---")
            keys, values = zip(*grid.items())
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

            for params in combinations:
                # 3. Pass the SUBSET paths to the experiment runner
                qps, avg_dist, recall = run_experiment(
                    args.script, temp_db_file, temp_query_file, temp_output_file, 
                    method, params, ground_truth
                )
                
                if qps is not None:
                    row = {
                        "algorithm": method,
                        "recall": round(recall, 4),
                        "qps": round(qps, 2),
                        "avg_neighbor_distance": round(avg_dist, 4)
                    }
                    row.update(params)
                    results_data.append(row)

        if results_data:
            base_cols = ["algorithm", "recall", "qps", "avg_neighbor_distance"]
            all_keys = set().union(*(r.keys() for r in results_data))
            fieldnames = base_cols + sorted([k for k in all_keys if k not in base_cols])

            with open(args.csv_out, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in results_data:
                    writer.writerow(row)
            print(f"\nSuccess! Results written to {args.csv_out}")
        else:
            print("\nNo results collected due to errors.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nCleaning up temporary files...")
        for f in [temp_output_file, temp_db_file, temp_query_file]:
            if os.path.exists(f):
                os.remove(f)
        print("Cleanup complete.")

if __name__ == "__main__":
    main()