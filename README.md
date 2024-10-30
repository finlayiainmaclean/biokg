# BioKG

1. Create local environment
```
micromamba install --name base --file environment.yaml && \
    micromamba clean --all --yes

RUN --mount=type=secret,id=pip,target=/root/.pip/pip.conf \
    micromamba run -n base pip install -r requirements.txt && \
    micromamba run -n base pip install -e biokg
```

2. Copy `Nodes.csv`, `Edges.csv`, `Embeddings.csv` and `Ground Truth.csv` to the `data` directory.

3. Run `jupyter execute notebooks/1_data.ipynb notebooks/2_degree_distribution.ipynb notebooks/3_model.ipynb` to run a simple pipeline. In short we:
- (`notebooks/1_data.ipynb`) Load the raw data and build a tripartite network of disease-molecule-genes, alongside adding NER embeddings from a recent BeRT model.
- (`notebooks/2_degree_distribution.ipynb`) Assess the effect of degree distribution (connectivity) and the prevalence of frequent fliers.
- (`notebooks/3_model.ipynb`) Evaluate two models (Node2Vec, GraphSage) alongside the provided embeddings on simple random splits. We then predict the full 100M disease-molecule matrix to identify highly predicted and enriched novel disease-pairs.
