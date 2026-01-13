#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
from Bio import SeqIO
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ---------------------------
# Configuration
# ---------------------------
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load model
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

# ---------------------------
# Functions
# ---------------------------

def load_fasta(fasta_path):
    """
    Load sequences from a FASTA file.
    Returns a list of tuples: (protein_id, sequence)
    """
    proteins = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        proteins.append((record.id, str(record.seq)))
    return proteins

def embed_sequence(sequence):
    """
    Embed a single protein sequence using ESM-2.
    Returns a 1D numpy array.
    """
    inputs = tokenizer(sequence, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Remove CLS and EOS tokens
    token_embeddings = outputs.last_hidden_state[0, 1:-1]

    # Mean pooling
    embedding = token_embeddings.mean(dim=0)
    return embedding.cpu().numpy()

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Protein sequence embedding using ESM-2")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Output .npz file for embeddings")
    args = parser.parse_args()

    input_fasta = Path(args.input)
    output_npz = Path(args.output)

    if not input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA file {input_fasta} not found!")

    # Load sequences
    print(f"Loading sequences from {input_fasta} ...")
    proteins = load_fasta(input_fasta)
    print(f"Found {len(proteins)} sequences")

    # Embed sequences
    embeddings = {}
    for protein_id, seq in tqdm(proteins, desc="Embedding proteins"):
        embeddings[protein_id] = embed_sequence(seq)

    # Save embeddings to .npz
    np.savez(output_npz, **embeddings)
    print(f"Saved {len(embeddings)} embeddings to {output_npz}")

# ---------------------------
if __name__ == "__main__":
    main()
