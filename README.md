# ex3-remote-homology-search
Software project for algorithmic problems part 3 - Remote protein homology search



Part 1 - Embeddings with ESM-2

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



Part 2 - BLAST
        If not installed run:
            conda install -c bioconda blast

        Then run build_blast.py
        BLAST provides:

            A ranked list of similar proteins per query

            Sequence identity percentages

            A Top-N list used to evaluate ANN methods

            Ground truth for Recall@N




---> TO COMPILE THE PROJECT RUN : 
``` 
    python3 setup.py build_ext --inplace 
``` 

You will need to setup pybind11 and 
```
pip install pybind11 setuptools wheel
```

It worked with Python 3.12.3  
