import torch
import numpy as np
from pathlib import Path
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# Configuration

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


# Load FASTA

def load_fasta(fasta_path):
    proteins = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        proteins.append((record.id, str(record.seq)))
    return proteins

# Embed one sequence

def embed_sequence(sequence):
    inputs = tokenizer(sequence, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Remove CLS and EOS tokens
    token_embeddings = outputs.last_hidden_state[0, 1:-1]

    # Mean pooling
    embedding = token_embeddings.mean(dim=0)

    return embedding.cpu().numpy()

# Fixed paths for now
def main():
    proteins = load_fasta("data/swissprot.fasta")

    embeddings = {}
    for protein_id, sequence in tqdm(proteins, desc="Embedding proteins"):
        embeddings[protein_id] = embed_sequence(sequence)

    # Save embeddings
    np.savez("data/swissprot_embeddings.npz", **embeddings)

    print(f"Saved {len(embeddings)} embeddings to data/swissprot_embeddings.npz")

if __name__ == "__main__":
    import argparse

    
    main()
