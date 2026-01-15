#!/usr/bin/env python3

import argparse
import numpy as np
from collections import defaultdict
from Bio import SeqIO
from itertools import product
from tqdm import tqdm

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

## Formats embeddings as a matrix of k-mer frequencies for nlsh  ##



def build_kmer_index(k):
    """Map each k-mer to a unique index."""
    kmers = [''.join(p) for p in product(AMINO_ACIDS, repeat=k)]
    return {kmer: i for i, kmer in enumerate(kmers)}


def embed_sequence(seq, k, kmer_index):
    """Compute k-mer frequency embedding."""
    vec = np.zeros(len(kmer_index), dtype=np.float32)
    total = 0

    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if all(c in AMINO_ACIDS for c in kmer):
            vec[kmer_index[kmer]] += 1
            total += 1

    if total > 0:
        vec /= total  # normalize counts

    return vec


def l2_normalize(x):
    norm = np.linalg.norm(x)
    if norm > 0:
        x /= norm
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, help="SwissProt FASTA file")
    parser.add_argument("--k", type=int, default=3, help="k-mer size")
    parser.add_argument("--out", default="swissprot_embeddings_matrix.npz")
    args = parser.parse_args()

    print("Reading FASTA...")
    records = list(SeqIO.parse(args.fasta, "fasta"))
    print(f"Loaded {len(records)} protein sequences")

    print("Building k-mer index...")
    kmer_index = build_kmer_index(args.k)

    embeddings = np.zeros((len(records), len(kmer_index)), dtype=np.float32)

    print("Computing embeddings...")
    for i, record in enumerate(tqdm(records)):
        seq = str(record.seq)
        vec = embed_sequence(seq, args.k, kmer_index)
        embeddings[i] = l2_normalize(vec)

    print("Saving embeddings...")
    np.savez_compressed(
        args.out,
        embeddings=embeddings
    )

    print(f"Done ✅ saved {args.out}")
    print("Shape:", embeddings.shape)


if __name__ == "__main__":
    main()
