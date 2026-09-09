"""Map contrastiveVI salient states to GenePT space for perturbation retrieval.

Important evaluation boundary:
Perturbation targets are held out from training the inverse mapping model. The upstream
contrastiveVI representation model is still trained on all perturbed cells and therefore
this is *not* a strict end-to-end unseen-perturbation evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SalientToGenePredictor(nn.Module):
    """MLP projector from salient cell-state latent space to GenePT embedding space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.best_model_state: dict | None = None
        self.early_stop = False

    def update(self, val_loss: float, model: nn.Module) -> None:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience


def split_by_perturbation_gene(
    adata: Any,
    gene_embeddings: dict[str, np.ndarray],
    salient_key: str,
    test_fraction: float = 0.2,
    val_fraction_of_remaining: float = 0.2,
    batch_size: int = 128,
    seed: int = 42,
):
    """Create train/validation/test loaders using perturbation-target-level splits."""
    if salient_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{salient_key!r}].")

    valid_gene_set = set(gene_embeddings)
    mask = (
        (adata.obs["perturbed_gene"] != "ctrl")
        & adata.obs["perturbed_gene"].isin(valid_gene_set)
    )
    subset = adata[mask].copy()
    unique_genes = np.unique(subset.obs["perturbed_gene"].astype(str).values)
    if len(unique_genes) < 5:
        raise ValueError(
            "At least 5 perturbation genes with GenePT embeddings are required "
            "for the default train/validation/test split."
        )

    train_val_genes, test_genes = train_test_split(
        unique_genes,
        test_size=test_fraction,
        random_state=seed,
    )
    train_genes, val_genes = train_test_split(
        train_val_genes,
        test_size=val_fraction_of_remaining,
        random_state=seed,
    )

    def make_loader(target_genes: np.ndarray, shuffle: bool) -> DataLoader:
        gene_mask = subset.obs["perturbed_gene"].isin(target_genes)
        chunk = subset[gene_mask]
        x = torch.tensor(chunk.obsm[salient_key], dtype=torch.float32)
        y = torch.tensor(
            np.stack([gene_embeddings[str(g)] for g in chunk.obs["perturbed_gene"]]),
            dtype=torch.float32,
        )
        drop_last = bool(shuffle and len(x) > 1 and len(x) % batch_size == 1)
        return DataLoader(
            TensorDataset(x, y),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    loaders = {
        "train": make_loader(train_genes, shuffle=True),
        "val": make_loader(val_genes, shuffle=False),
        "test": make_loader(test_genes, shuffle=False),
    }
    gene_split = {
        "train": [str(g) for g in train_genes],
        "val": [str(g) for g in val_genes],
        "test": [str(g) for g in test_genes],
    }
    return subset, loaders, gene_split


def train_predictor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 200,
    patience: int = 10,
) -> list[dict[str, float]]:
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CosineEmbeddingLoss()
    stopper = EarlyStopping(patience=patience)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            targets = torch.ones(batch_x.size(0), device=device)
            loss = criterion(predictions, batch_y, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_similarity = 0.0
        n_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                targets = torch.ones(batch_x.size(0), device=device)
                val_loss += criterion(predictions, batch_y, targets).item()
                val_similarity += F.cosine_similarity(predictions, batch_y).sum().item()
                n_val += batch_x.size(0)
        val_loss /= len(val_loader)
        val_similarity /= n_val
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_cosine": val_similarity,
            }
        )

        stopper.update(val_loss, model)
        if stopper.early_stop:
            break

    if stopper.best_model_state is None:
        raise RuntimeError("No valid model state was recorded during training.")
    model.load_state_dict(stopper.best_model_state)
    return history


def predict_mean_embedding(
    model: nn.Module,
    latent_vectors: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if len(latent_vectors) == 0:
        raise ValueError("Cannot predict an embedding from zero cells.")
    model.eval()
    x = torch.tensor(latent_vectors, dtype=torch.float32, device=device)
    with torch.no_grad():
        predictions = model(x).cpu().numpy()
    return predictions.mean(axis=0)


def evaluate_cosine_retrieval(
    model: nn.Module,
    adata: Any,
    gene_split: dict[str, list[str]],
    gene_embeddings: dict[str, np.ndarray],
    salient_key: str,
    device: torch.device,
) -> pd.DataFrame:
    """Rank candidate perturbation genes for each held-out test perturbation."""
    candidate_genes = gene_split["train"] + gene_split["val"] + gene_split["test"]
    candidate_matrix = np.stack([gene_embeddings[g] for g in candidate_genes])
    candidate_matrix = candidate_matrix / np.linalg.norm(candidate_matrix, axis=1, keepdims=True)

    rows: list[dict[str, float | int | str]] = []
    for query_gene in gene_split["test"]:
        query_mask = adata.obs["perturbed_gene"].astype(str).values == query_gene
        mean_prediction = predict_mean_embedding(
            model,
            adata.obsm[salient_key][query_mask],
            device,
        )
        mean_prediction = mean_prediction / np.linalg.norm(mean_prediction)
        similarities = candidate_matrix @ mean_prediction
        order = np.argsort(-similarities)
        true_index = candidate_genes.index(query_gene)
        rank = int(np.where(order == true_index)[0][0]) + 1
        rows.append(
            {
                "query_gene": query_gene,
                "true_rank": rank,
                "true_cosine": float(similarities[true_index]),
                "top1_gene": candidate_genes[int(order[0])],
                "top1_cosine": float(similarities[int(order[0])]),
            }
        )
    return pd.DataFrame(rows)


def summarize_ranks(ranking: pd.DataFrame) -> dict[str, float]:
    ranks = ranking["true_rank"].to_numpy()
    return {
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "top1_rate": float(np.mean(ranks <= 1)),
        "top5_rate": float(np.mean(ranks <= 5)),
        "top10_rate": float(np.mean(ranks <= 10)),
        "n_test_genes": int(len(ranks)),
    }


def load_genept_embeddings(path: Path) -> dict[str, np.ndarray]:
    table = pd.read_csv(path, index_col=0)
    table = table[table.index.astype(str) != "ctrl"]
    if table.empty:
        raise ValueError("GenePT embedding table contains no perturbation genes.")
    if table.isna().any().any():
        raise ValueError("GenePT embedding table contains missing values.")
    embeddings = {
        str(gene): table.loc[gene].to_numpy(dtype=np.float32)
        for gene in table.index
    }
    if any(np.linalg.norm(vector) == 0 for vector in embeddings.values()):
        raise ValueError("GenePT embedding table contains a zero-norm vector.")
    return embeddings


def run_repeated_splits(
    adata: Any,
    gene_embeddings: dict[str, np.ndarray],
    output_dir: Path,
    salient_key: str = "X_salient",
    n_splits: int = 20,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Run repeated perturbation-level splits and save transparent retrieval metrics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_summaries: list[dict[str, float | int]] = []
    for split_id in range(1, n_splits + 1):
        set_seed(split_id)
        subset, loaders, gene_split = split_by_perturbation_gene(
            adata,
            gene_embeddings,
            salient_key=salient_key,
            test_fraction=0.2,
            val_fraction_of_remaining=0.2,
            batch_size=batch_size,
            seed=split_id,
        )

        input_dim = subset.obsm[salient_key].shape[1]
        output_dim = len(next(iter(gene_embeddings.values())))
        model = SalientToGenePredictor(input_dim, output_dim).to(device)
        history = train_predictor(
            model,
            loaders["train"],
            loaders["val"],
            device=device,
        )
        ranking = evaluate_cosine_retrieval(
            model,
            subset,
            gene_split,
            gene_embeddings,
            salient_key,
            device,
        )
        summary = summarize_ranks(ranking)
        summary["split_id"] = split_id
        split_summaries.append(summary)

        ranking.to_csv(output_dir / f"split_{split_id:02d}_ranking.csv", index=False)
        pd.DataFrame(history).to_csv(
            output_dir / f"split_{split_id:02d}_training.csv",
            index=False,
        )
        with open(output_dir / f"split_{split_id:02d}_genes.json", "w") as handle:
            json.dump(gene_split, handle, indent=2)

    metrics = pd.DataFrame(split_summaries)
    metrics.to_csv(output_dir / "retrieval_metrics.csv", index=False)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/adamson_contrastivevi.h5ad"),
    )
    parser.add_argument(
        "--gene-embeddings",
        type=Path,
        required=True,
        help="GenePT embedding CSV (Embedding_small.csv in the original project).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inverse_retrieval"),
    )
    parser.add_argument("--salient-key", default="X_salient")
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import scanpy as sc

    adata = sc.read_h5ad(args.input)
    embeddings = load_genept_embeddings(args.gene_embeddings)
    metrics = run_repeated_splits(
        adata,
        embeddings,
        output_dir=args.output_dir,
        salient_key=args.salient_key,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
    )
    print(metrics)
    print("\nMean across perturbation-level splits:")
    print(metrics.drop(columns=["split_id"]).mean(numeric_only=True))
