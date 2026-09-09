"""Train the canonical contrastiveVI representation model.

Canonical research configuration selected from the original experiments:
- maximum 500 cells per non-control perturbation (smaller groups are retained),
- maximum 5,000 control cells,
- 20 salient and 20 background latent dimensions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def cap_cells_per_perturbation(
    adata: Any,
    obs_key: str = "perturbed_gene",
    control_label: str = "ctrl",
    max_perturbation_cells: int = 500,
    max_control_cells: int = 5000,
    seed: int = 42,
) -> Any:
    """Cap cell counts without dropping perturbations smaller than the cap.

    The original notebook called this operation "balancing". Because groups with fewer
    than ``max_perturbation_cells`` are retained as-is, "capping" is the more accurate
    description.
    """
    rng = np.random.default_rng(seed)
    selected: list[int] = []

    for label, indices in adata.obs.groupby(obs_key, observed=True).indices.items():
        indices = np.asarray(indices)
        cap = max_control_cells if label == control_label else max_perturbation_cells
        if len(indices) > cap:
            indices = rng.choice(indices, size=cap, replace=False)
        selected.extend(indices.tolist())

    capped = adata[np.asarray(selected, dtype=int)].copy()
    print(f"Cells before capping: {adata.n_obs}")
    print(f"Cells after capping:  {capped.n_obs}")
    print(capped.obs[obs_key].value_counts().describe())
    return capped


def train_contrastivevi(
    adata: Any,
    latent_dim: int = 20,
    salient_key: str = "X_salient",
    max_epochs: int = 500,
    seed: int = 42,
):
    """Train contrastiveVI using control cells as background and perturbed cells as target."""
    import scvi

    np.random.seed(seed)
    scvi.settings.seed = seed

    if "counts" not in adata.layers:
        raise KeyError("AnnData must contain raw counts in adata.layers['counts'].")

    scvi.external.ContrastiveVI.setup_anndata(adata, layer="counts")
    model = scvi.external.ContrastiveVI(
        adata,
        n_salient_latent=latent_dim,
        n_background_latent=latent_dim,
        use_observed_lib_size=True,
    )

    background_indices = np.where(adata.obs["perturbed_gene"] == "ctrl")[0]
    target_indices = np.where(adata.obs["perturbed_gene"] != "ctrl")[0]
    if len(background_indices) == 0 or len(target_indices) == 0:
        raise ValueError("Both control and perturbed cells are required for contrastiveVI.")

    print(f"Total cells:        {adata.n_obs}")
    print(f"Background (ctrl):  {len(background_indices)}")
    print(f"Perturbed targets:  {len(target_indices)}")

    model.train(
        background_indices=background_indices,
        target_indices=target_indices,
        early_stopping=True,
        max_epochs=max_epochs,
    )
    adata.obsm[salient_key] = model.get_latent_representation(
        adata,
        representation_kind="salient",
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/adamson_processed.h5ad"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/adamson_contrastivevi.h5ad"),
    )
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--max-perturbation-cells", type=int, default=500)
    parser.add_argument("--max-control-cells", type=int, default=5000)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import scanpy as sc

    adata = sc.read_h5ad(args.input)
    adata = cap_cells_per_perturbation(
        adata,
        max_perturbation_cells=args.max_perturbation_cells,
        max_control_cells=args.max_control_cells,
        seed=args.seed,
    )
    train_contrastivevi(
        adata,
        latent_dim=args.latent_dim,
        salient_key="X_salient",
        max_epochs=args.max_epochs,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output, compression="gzip")
    print(f"Saved: {args.output}")
