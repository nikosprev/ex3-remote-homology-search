import subprocess
import os

# Run once to set up BLAST database and perform BLASTP search

INPUT = "data/swissprot.fasta"
DB_PATH = "data/blast/swissprot_db"
OUTPUT_TSV = "data/blast/blast_results.tsv"
QUERY_FASTA = "data/targets.fasta"
MAX_TARGETS = 200
EVALUE = "1e-5"
THREADS = "4"

os.makedirs("data/blast", exist_ok=True)





# Build the command
command = [
    "makeblastdb",
    "-in", INPUT,
    "-dbtype", "prot",
    "-out", DB_PATH
]

# Run the command
try:
    subprocess.run(command, check=True)
    print(f"BLAST database created successfully at {DB_PATH}")
except subprocess.CalledProcessError as e:
    print(f"Error creating BLAST database: {e}")

cmd = [
    "blastp",
    "-query", QUERY_FASTA,
    "-db", DB_PATH,
    "-outfmt", "6",
    "-max_target_seqs", str(MAX_TARGETS),
    "-evalue", EVALUE,
    "-num_threads", THREADS,
    "-out", OUTPUT_TSV
]

print("Running BLASTP...")
print(" ".join(cmd))

subprocess.run(cmd, check=True)

print("BLASTP finished successfully.")
