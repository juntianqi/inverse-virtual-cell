# Inverse Virtual Cell

**From Cell-State Representations to Candidate Perturbations**

> **Status: Work in Progress / Research Prototype**  
> This repository contains an ongoing research prototype for inverse virtual cell modeling. The model design, experiments, evaluation protocol, and APIs are still under active development and may change substantially. This repository should not be interpreted as a completed or fully validated perturbation-design system.

## Overview

Most virtual-cell and perturbation-prediction models are formulated in the forward direction:

```text
Perturbation -> Cellular Response
```

This project explores the inverse problem:

```text
Desired / Target Cell State -> Candidate Perturbation
```

The long-term goal is to infer upstream genetic interventions that may move a cell toward a target state. The **current implementation is narrower than this full vision**: it learns perturbation-associated cell-state representations from existing Perturb-seq data and maps those representations into a GenePT gene-embedding space for candidate perturbation retrieval.

## Motivation

Forward perturbation models ask what cellular state will follow a known intervention. In many biological design settings, the complementary question is more directly actionable: given a cellular state of interest, which perturbation may be associated with that state?

Inverse modeling is challenging because observed single-cell variation contains both perturbation-related signals and background variation such as shared cellular programs. This prototype therefore separates the problem into two stages:

1. learn a perturbation-specific cell-state representation; and
2. map that representation into a gene representation space that supports perturbation retrieval.

## Current Framework

The current canonical pipeline is:

```text
Adamson Perturb-seq
        |
        v
Preprocessing and QC
        |
        v
contrastiveVI
        |
        v
Perturbation-specific salient representation
        |
        v
SalientToGenePredictor
        |
        v
GenePT embedding space
        |
        v
Cosine similarity ranking
        |
        v
Candidate perturbations
```

### 1. Perturb-seq preprocessing

The current prototype uses the public Adamson Perturb-seq dataset (GEO: **GSE90546**). The preprocessing script:

- reconstructs the Adamson single-cell count matrices and perturbation annotations;
- retains cells marked as `good coverage`;
- preserves raw counts for contrastiveVI;
- performs library-size normalization and log transformation for feature selection;
- selects highly variable genes while retaining perturbation target genes; and
- restricts perturbation targets to genes represented in the supplied GenePT embedding table.

### 2. Perturbation-specific representation learning

The representation stage uses **contrastiveVI** to distinguish variation shared between control and perturbed cells from variation enriched in perturbed cells.

The current default research configuration is:

```text
maximum cells per perturbation: 500
maximum control cells:          5000
salient latent dimension:       20
background latent dimension:    20
```

Groups smaller than the perturbation cap are retained as-is; this is cell-count **capping**, not strict class balancing.

The salient latent representation is used as the perturbation-associated cell-state representation for the inverse mapping stage.

### 3. Cell-state-to-gene mapping

A small multilayer perceptron, `SalientToGenePredictor`, maps each contrastiveVI salient vector into the **GenePT** embedding space.

The current projector uses:

```text
20-d salient state
      |
      v
256 hidden units
      |
      v
256 hidden units
      |
      v
GenePT embedding
```

The model is trained with cosine embedding loss.

### 4. Candidate perturbation retrieval

Perturbation targets are split at the **gene level**, rather than randomly splitting individual cells. For held-out perturbation genes, predicted embeddings are compared against candidate GenePT embeddings using cosine similarity.

The current evaluation reports:

- mean rank;
- median rank;
- Top-1 retrieval rate;
- Top-5 retrieval rate; and
- Top-10 retrieval rate.

### Important evaluation boundary

The held-out perturbation split currently applies to the **inverse mapping model**. The upstream contrastiveVI model is trained using the full set of control and perturbed cells before the perturbation-gene split is applied.

Therefore, the current evaluation should **not** be interpreted as strict end-to-end unseen-perturbation generalization. A fully isolated upstream representation-learning evaluation is part of the roadmap.

## Repository Structure

```text
inverse-virtual-cell/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- scripts/
|   |-- preprocess_adamson.py
|   |-- train_contrastivevi.py
|   `-- inverse_retrieval.py
|
|-- tests/
|   `-- test_smoke.py
|
`-- assets/
    `-- ...
```

### Core scripts

| Script | Purpose |
| --- | --- |
| `scripts/preprocess_adamson.py` | Prepare the Adamson Perturb-seq data and apply canonical QC / feature selection. |
| `scripts/train_contrastivevi.py` | Learn perturbation-associated salient representations with contrastiveVI. |
| `scripts/inverse_retrieval.py` | Map salient states to GenePT space and rank candidate perturbations. |
| `tests/test_smoke.py` | Dependency-light smoke tests for core Python logic using toy data. |

## Installation

The repository currently lists direct dependencies rather than a fully pinned environment because the refactored end-to-end pipeline has not yet been revalidated in a frozen software environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Data and Gene Embeddings

Raw datasets, processed AnnData files, model checkpoints, and large generated outputs are intentionally not included in the repository.

The current pipeline expects:

1. the Adamson Perturb-seq raw files from **GSE90546**; and
2. a GenePT embedding CSV corresponding to the perturbation genes used by the model.

GenePT embedding files are **not redistributed in this repository**. Please obtain the relevant embeddings from the upstream GenePT resources and review their current usage/redistribution terms before reuse.

Example local layout:

```text
data/
|-- raw/
|   `-- adamson/
|-- external/
|   `-- Embedding_small.csv
`-- processed/
```

The `data/` directory is ignored by Git.

## Running the Current Prototype

### 1. Preprocess Adamson Perturb-seq

```bash
python scripts/preprocess_adamson.py \
  --raw-dir data/raw/adamson \
  --gene-embeddings data/external/Embedding_small.csv \
  --output data/processed/adamson_processed.h5ad
```

### 2. Learn contrastiveVI salient representations

```bash
python scripts/train_contrastivevi.py \
  --input data/processed/adamson_processed.h5ad \
  --output data/processed/adamson_contrastivevi.h5ad
```

### 3. Run inverse perturbation retrieval

```bash
python scripts/inverse_retrieval.py \
  --input data/processed/adamson_contrastivevi.h5ad \
  --gene-embeddings data/external/Embedding_small.csv \
  --output-dir outputs/inverse_retrieval
```

By default, the inverse model is evaluated across 20 repeated perturbation-level splits.

## Smoke Test

A dependency-light smoke test is included to validate the core Python logic with toy data:

```bash
python tests/test_smoke.py
```

The smoke test covers:

- boolean QC parsing;
- perturbation/control cell-count capping;
- perturbation-level train/validation/test splitting;
- inverse MLP forward and training paths; and
- cosine-based candidate ranking.

## Current Status and Limitations

This repository is intentionally presented as a **research prototype**, not a production software package.

Currently implemented:

- Adamson Perturb-seq preprocessing;
- contrastiveVI-based salient representation learning;
- perturbation-level cell-count capping;
- mapping salient cell states to GenePT embeddings;
- perturbation-gene-level train/validation/test splitting;
- cosine-based candidate perturbation ranking; and
- toy-data smoke tests for the main Python logic.

Not yet established as completed functionality:

- arbitrary user-defined desired cell states;
- strict end-to-end unseen-perturbation generalization;
- validated cross-dataset generalization;
- combinatorial perturbation design;
- multimodal perturbation representations;
- experimentally validated inverse intervention recommendations;
- a stable public API; and
- fully pinned end-to-end reproducibility after the latest QC/refactoring changes.

The preprocessing pipeline now explicitly applies the Adamson `good coverage` filter. Because this differs from the historical notebook execution used during earlier exploration, previously saved performance values should not be treated as results from the current refactored pipeline until the full workflow is rerun.

## Roadmap

Planned research directions include:

- strict end-to-end held-out perturbation evaluation;
- perturbation retrieval on additional Perturb-seq datasets;
- stronger and multimodal gene representations;
- target-state-conditioned perturbation ranking;
- unseen perturbation generalization;
- extension from retrieval toward inverse intervention design; and
- evaluation of biologically meaningful target-state transitions.

## References

This prototype builds on the following public resources and methods:

- **contrastiveVI / scvi-tools** — salient representation learning for single-cell data: https://docs.scvi-tools.org/en/latest/tutorials/notebooks/scrna/contrastiveVI_tutorial.html
- **GenePT** — gene representations derived from biological text: https://github.com/yiqunchen/GenePT
- **Adamson Perturb-seq dataset (GSE90546)** — https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546

If you use these upstream methods or datasets, please cite their original publications.

## Citation and Contact

A standalone citation for **Inverse Virtual Cell** is not currently available; this project is ongoing research in progress.

For questions about this repository, please contact the repository owner or open a GitHub issue.
