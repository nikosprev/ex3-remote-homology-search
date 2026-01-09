
import subprocess
from collections import defaultdict

def run_blast(
    query_fasta,
    db_path,
    output_tsv,
    max_targets=200,
    evalue=1e-5,
    threads=4
):
    cmd = [
        "blastp",
        "-query", query_fasta,
        "-db", db_path,
        "-outfmt", "6",
        "-max_target_seqs", str(max_targets),
        "-evalue", str(evalue),
        "-num_threads", str(threads),
        "-out", output_tsv
    ]
    subprocess.run(cmd, check=True)


def parse_blast_results(tsv_file):
 
    #Returns:
     #blast_results[query_id] = list of hits sorted by bit score
    # blast_identity[(query_id, subject_id)] = identity %
  
    blast_results = defaultdict(list)
    blast_identity = {}

    with open(tsv_file) as f:
        for line in f:
            cols = line.strip().split("\t")
            query_id = cols[0]
            subject_id = cols[1]
            identity = float(cols[2])
            bit_score = float(cols[11])

            blast_results[query_id].append({
                "subject_id": subject_id,
                "identity": identity,
                "bit_score": bit_score
            })

            blast_identity[(query_id, subject_id)] = identity

    # sort by bit score 
    for q in blast_results:
        blast_results[q].sort(
            key=lambda x: x["bit_score"],
            reverse=True
        )

    return blast_results, blast_identity


def get_blast_topN(blast_results, query_id, N=50):
   
    # Returns a set of subject_ids
 
    return set(
        hit["subject_id"]
        for hit in blast_results[query_id][:N]
    )

