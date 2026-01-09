
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blast_util import run_blast, parse_blast_results, get_blast_topN

# Step 1: Run BLAST
query_fasta ="data/targets.fasta"
db_path = "data/blast/swissprot_db"
output_tsv =  "data/blast/blast_results.tsv"


print("Running BLAST...")
run_blast(query_fasta, db_path, output_tsv)
print("BLAST finished!")

# Step 2: Parse BLAST results
blast_results, blast_identity = parse_blast_results(output_tsv)

# Step 3: Get top N hits for a query
query_id = next(iter(blast_results))   # Get the first query ID

top_hits = get_blast_topN(blast_results, query_id, N=10)

print(f"Top 10 hits for {query_id}:")
for hit in top_hits:
    identity = blast_identity[(query_id, hit)]
    print(f"  {hit} (Identity: {identity}%)")
