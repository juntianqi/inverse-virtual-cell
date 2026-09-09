"""Prepare the Adamson Perturb-seq dataset for the Inverse Virtual Cell prototype.

This script preserves the preprocessing logic from the original research notebook while
making paths explicit and filtering cells marked as ``good coverage`` before model
training. Only perturbation targets represented in the supplied GenePT embedding table
are retained.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONTROL_GUIDES = {"62(mod)", "63(mod)", "Gal4-4(mod)"}


def load_adamson_sample(
    raw_dir: Path,
    sample_id: str,
    prefix: str,
    min_gene_cells: int,
) -> Any:
    """Load one Adamson 10x sample and attach the published cell annotations."""
    import scanpy as sc
    matrix_path = raw_dir / f"{prefix}_matrix.mtx.gz"
    barcode_path = raw_dir / f"{prefix}_barcodes.tsv.gz"
    feature_path = raw_dir / f"{prefix}_features.tsv.gz"
    identity_path = raw_dir / f"{prefix}_cell_identities.csv.gz"

    adata = sc.read_mtx(matrix_path).T
    barcodes = pd.read_table(barcode_path, names=["CellID"])
    genes = pd.read_table(feature_path, names=["geneName"])
    adata.obs_names = barcodes["CellID"].astype(str).values
    adata.var_names = genes["geneName"].astype(str).values

    identities = pd.read_csv(identity_path, index_col="cell BC")
    shared_cells = adata.obs_names.intersection(identities.index)
    adata = adata[shared_cells].copy()
    adata.obs = identities.loc[adata.obs_names].copy()

    adata.obs_names = [f"{cell}-{sample_id}" for cell in adata.obs_names]
    adata.var_names_make_unique()

    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=min_gene_cells)
    return adata


def as_boolean(series: pd.Series) -> pd.Series:
    """Convert common boolean/string encodings to a boolean mask."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def preprocess_adamson(
    raw_dir: Path,
    gene_embedding_path: Path,
    output_path: Path,
    n_hvg: int = 5000,
) -> Any:
    """Run the canonical Adamson preprocessing pipeline and save an AnnData file."""
    import scanpy as sc
    sample_1 = load_adamson_sample(
        raw_dir=raw_dir,
        sample_id="GSM2406675",
        prefix="GSM2406675_10X001",
        min_gene_cells=3,
    )
    sample_2 = load_adamson_sample(
        raw_dir=raw_dir,
        sample_id="GSM2406681",
        prefix="GSM2406681_10X010",
        min_gene_cells=10,
    )

    common_genes = sample_1.var_names.intersection(sample_2.var_names)
    sample_1 = sample_1[:, common_genes].copy()
    sample_2 = sample_2[:, common_genes].copy()
    adata = sc.concat([sample_1, sample_2], join="inner")

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains(r"^HB[^(P)]", regex=True)
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        log1p=True,
    )

    if "guide identity" not in adata.obs:
        raise KeyError("Expected 'guide identity' in Adamson cell annotations.")

    adata.obs["perturbed_gene"] = [
        str(guide).split("_")[0] for guide in adata.obs["guide identity"]
    ]
    adata.obs["perturbed_gene"] = adata.obs["perturbed_gene"].replace(
        {guide: "ctrl" for guide in CONTROL_GUIDES}
    )

    gene_embeddings = pd.read_csv(gene_embedding_path, index_col=0)
    supported_targets = set(gene_embeddings.index.astype(str)) | {"ctrl"}
    adata = adata[adata.obs["perturbed_gene"].isin(supported_targets)].copy()

    if "good coverage" not in adata.obs:
        raise KeyError("Expected 'good coverage' in Adamson cell annotations.")
    coverage_mask = as_boolean(adata.obs["good coverage"])
    print(f"Cells before good-coverage filter: {adata.n_obs}")
    adata = adata[coverage_mask.values].copy()
    print(f"Cells after good-coverage filter:  {adata.n_obs}")

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=False)

    perturbation_targets = set(adata.obs["perturbed_gene"]) - {"ctrl"}
    keep_mask = adata.var["highly_variable"].to_numpy() | adata.var_names.isin(
        perturbation_targets
    )
    adata = adata[:, keep_mask].copy()
    adata.var["gene_name"] = adata.var_names

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    print(f"Saved: {output_path}")
    print(f"Final shape: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"Perturbation targets: {adata.obs['perturbed_gene'].nunique() - 1}")
    return adata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing the Adamson GSE90546 raw files.",
    )
    parser.add_argument(
        "--gene-embeddings",
        type=Path,
        required=True,
        help="GenePT embedding CSV (Embedding_small.csv in the original project).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/adamson_processed.h5ad"),
    )
    parser.add_argument("--n-hvg", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_adamson(
        raw_dir=args.raw_dir,
        gene_embedding_path=args.gene_embeddings,
        output_path=args.output,
        n_hvg=args.n_hvg,
    )
