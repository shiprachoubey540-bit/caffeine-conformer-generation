"""
Conformer Generation using Torsional Diffusion (Jing et al., NeurIPS 2022)
----------------------------------------------------------------------------
Generates conformers for a target molecule using the pretrained Torsional
Diffusion model, and saves the resulting 3D structures as a single SDF file.

Model repository: https://github.com/gcorso/torsional-diffusion
Paper: "Torsional Diffusion for Molecular Conformer Generation" (Jing et al., 2022)

WHAT THIS MODEL DOES
---------------------
Unlike classical methods (e.g. RDKit's ETKDG), which build a conformer directly
from empirical rules, Torsional Diffusion is a generative diffusion model that
operates ONLY over a molecule's rotatable-bond torsion angles (leaving bond
lengths and bond angles fixed at their standard chemical values). Starting from
random noise on those torsion angles, it iteratively denoises them, using a
GNN trained on the GEOM-Drugs dataset, until they converge to angles typical
of a low-energy, realistic conformer. Because it only searches the low-
dimensional torsional space rather than full 3D coordinate space, it needs far
fewer denoising steps than coordinate-based diffusion models (e.g. GeoDiff).

PREREQUISITES (must be done once, before running this script)
---------------------------------------------------------------
1. Clone the official repository and set up its conda environment:
       git clone https://github.com/gcorso/torsional-diffusion.git
       cd torsional-diffusion
       conda env create -f environment.yml
       conda activate torsional_diffusion
       pip install rdkit e3nn torch_geometric spyrmsd
       pip install torch_scatter torch_sparse torch_cluster \
           -f https://data.pyg.org/whl/torch-<TORCH_VERSION>+cu<CUDA_VERSION>.html

2. Download the pretrained GEOM-Drugs checkpoint (drugs_default) from the
   authors' Google Drive folder into ./workdir/ (see repo README for the link).

3. Run this script from INSIDE the cloned `torsional-diffusion` directory,
   since it depends on that repo's `generate_confs.py` and its `diffusion/`
   and `utils/` modules.

Usage:
    python generate_torsional_diffusion_conformers.py
"""

import os
import pickle
import subprocess
import sys

from rdkit import Chem


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"   # Caffeine
NUM_CONFORMERS = 10
MODEL_DIR = "workdir/workdir/drugs_default"   # path to pretrained checkpoint
INFERENCE_STEPS = 20
BATCH_SIZE = 128

TEST_CSV = "caffeine.csv"
RAW_PKL_OUTPUT = "caffeine_torsional_diffusion_confs.pkl"
SDF_OUTPUT = "caffeine_torsional_diffusion_conformers.sdf"


def build_input_csv(smiles: str, n_confs: int, csv_path: str) -> None:
    """Write the single-molecule input file expected by generate_confs.py.
    Format: smiles, n_confs, smiles (repeated column, as used in the
    official repo's own example test files)."""
    with open(csv_path, "w") as f:
        f.write("smiles,n_confs,smiles\n")
        f.write(f"{smiles},{n_confs},{smiles}\n")
    print(f"Wrote input file: {csv_path}")


def run_generation(csv_path: str, model_dir: str, n_confs: int,
                    inference_steps: int, batch_size: int, out_path: str) -> None:
    """Invoke the repository's own generate_confs.py with the pretrained model."""
    cmd = [
        sys.executable, "generate_confs.py",
        "--test_csv", csv_path,
        "--inference_steps", str(inference_steps),
        "--model_dir", model_dir,
        "--confs_per_mol", str(n_confs),
        "--out", out_path,
        "--tqdm", "--batch_size", str(batch_size), "--no_energy",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def save_conformers_to_sdf(pkl_path: str, smiles: str, sdf_path: str) -> None:
    """Load the generated conformers and write them to a single SDF file."""
    with open(pkl_path, "rb") as f:
        result = pickle.load(f)

    mols = result[smiles]
    print(f"Loaded {len(mols)} generated conformers")

    writer = Chem.SDWriter(sdf_path)
    for i, mol in enumerate(mols):
        mol.SetProp("_Name", f"caffeine_torsional_diffusion_conf_{i}")
        writer.write(mol)
    writer.close()
    print(f"Saved conformers to {sdf_path}")


def main():
    build_input_csv(SMILES, NUM_CONFORMERS, TEST_CSV)

    run_generation(
        csv_path=TEST_CSV,
        model_dir=MODEL_DIR,
        n_confs=NUM_CONFORMERS,
        inference_steps=INFERENCE_STEPS,
        batch_size=BATCH_SIZE,
        out_path=RAW_PKL_OUTPUT,
    )

    save_conformers_to_sdf(RAW_PKL_OUTPUT, SMILES, SDF_OUTPUT)


if __name__ == "__main__":
    main()
