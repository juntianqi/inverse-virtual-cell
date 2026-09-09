"""Dependency-light smoke tests for the Inverse Virtual Cell research prototype.

These tests deliberately avoid real Adamson data and contrastiveVI training. They verify
core Python logic, perturbation-level splitting, the inverse MLP forward/training path,
and candidate ranking with toy data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from preprocess_adamson import as_boolean
from train_contrastivevi import cap_cells_per_perturbation
from inverse_retrieval import (
    SalientToGenePredictor,
    evaluate_cosine_retrieval,
    split_by_perturbation_gene,
    summarize_ranks,
    train_predictor,
)


class ToyAnnData:
    """Minimal AnnData-like container needed by the dependency-light smoke tests."""

    def __init__(self, obs: pd.DataFrame, obsm: dict[str, np.ndarray]):
        self.obs = obs.reset_index(drop=True)
        self.obsm = {key: np.asarray(value) for key, value in obsm.items()}
        self.n_obs = len(self.obs)

    def __getitem__(self, index):
        if isinstance(index, pd.Series):
            index = index.to_numpy()
        index = np.asarray(index)
        positions = np.where(index)[0] if index.dtype == bool else index.astype(int)
        return ToyAnnData(
            self.obs.iloc[positions].copy(),
            {key: value[positions].copy() for key, value in self.obsm.items()},
        )

    def copy(self):
        return ToyAnnData(
            self.obs.copy(),
            {key: value.copy() for key, value in self.obsm.items()},
        )


def build_toy_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(10)]
    labels: list[str] = []
    latent: list[np.ndarray] = []
    embeddings: dict[str, np.ndarray] = {}

    for i, gene in enumerate(genes):
        embeddings[gene] = rng.normal(size=6).astype(np.float32)
        for _ in range(7):
            labels.append(gene)
            base = np.zeros(4, dtype=np.float32)
            base[i % 4] = 1.0
            latent.append(base + rng.normal(scale=0.05, size=4))

    for _ in range(11):
        labels.append("ctrl")
        latent.append(rng.normal(size=4))

    adata = ToyAnnData(
        pd.DataFrame({"perturbed_gene": labels}),
        {"X_salient": np.asarray(latent, dtype=np.float32)},
    )
    return adata, embeddings


def test_boolean_parser():
    parsed = as_boolean(pd.Series([True, False, "TRUE", "yes", "0", None]))
    assert parsed.tolist() == [True, False, True, True, False, False]


def test_cell_capping():
    adata, _ = build_toy_data()
    capped = cap_cells_per_perturbation(
        adata,
        max_perturbation_cells=5,
        max_control_cells=6,
        seed=1,
    )
    counts = capped.obs["perturbed_gene"].value_counts()
    assert counts["ctrl"] == 6
    assert all(counts[gene] == 5 for gene in counts.index if gene != "ctrl")


def test_inverse_mapping_and_retrieval():
    adata, embeddings = build_toy_data()
    subset, loaders, gene_split = split_by_perturbation_gene(
        adata,
        embeddings,
        salient_key="X_salient",
        test_fraction=0.2,
        val_fraction_of_remaining=0.2,
        batch_size=8,
        seed=1,
    )

    assert set(gene_split["train"]).isdisjoint(gene_split["val"])
    assert set(gene_split["train"]).isdisjoint(gene_split["test"])
    assert set(gene_split["val"]).isdisjoint(gene_split["test"])

    model = SalientToGenePredictor(4, 6, hidden_dims=(8, 8), dropout=0.1)
    assert model(torch.randn(4, 4)).shape == (4, 6)

    history = train_predictor(
        model,
        loaders["train"],
        loaders["val"],
        device=torch.device("cpu"),
        epochs=3,
        patience=2,
    )
    assert len(history) >= 1

    ranking = evaluate_cosine_retrieval(
        model,
        subset,
        gene_split,
        embeddings,
        salient_key="X_salient",
        device=torch.device("cpu"),
    )
    summary = summarize_ranks(ranking)
    assert len(ranking) == len(gene_split["test"])
    assert ranking["true_rank"].between(1, len(embeddings)).all()
    assert 0.0 <= summary["top10_rate"] <= 1.0


if __name__ == "__main__":
    test_boolean_parser()
    test_cell_capping()
    test_inverse_mapping_and_retrieval()
    print("Smoke tests passed.")
