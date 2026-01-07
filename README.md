# ex3-remote-homology-search
Software project for algorithmic problems part 3 - Remote protein homology search



Part one - Embeddings with ESM-2

    Conceptual pipeline (high level)
    FASTA file (protein sequences)
        ↓
    ESM-2 model
        ↓
    Per-amino-acid representations (last layer)
        ↓
    Mean pooling
        ↓
    One vector per protein
        ↓
       Save