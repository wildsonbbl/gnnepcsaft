"""Compute molecule properties using RDKit + PySCF DFT."""

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass
class DFTConfig:
    """Configuration for a single-point DFT dipole calculation."""

    charge: int = 0
    multiplicity: int = 1
    xc: str = "B3LYP"
    basis: str = "def2-SVP"


def smiles_to_3d(smiles: str, max_iters: int = 500):
    """Generate a 3D structure (with hydrogens) from SMILES using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES.")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()  # type: ignore
    params.useSmallRingTorsions = True
    params.useRandomCoords = True
    cid = AllChem.EmbedMolecule(mol, params)  # type: ignore
    if cid == -1:
        # Retry with random seed
        params.randomSeed = 0xF00D
        cid = AllChem.EmbedMolecule(mol, params)  # type: ignore
        if cid == -1:
            raise RuntimeError("Conformer embedding failed.")

    # MMFF optimize if possible, else UFF
    if AllChem.MMFFHasAllMoleculeParams(mol):  # type: ignore
        status = AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)  # type: ignore
    else:
        status = AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)  # type: ignore
    # status=0 means converged; we proceed regardless but warn
    if status != 0:
        sys.stderr.write("Warning: MMFF/UFF optimization did not fully converge.\n")

    conf = mol.GetConformer()
    symbols = []
    coords = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        symbols.append(atom.GetSymbol())
        coords.append([pos.x, pos.y, pos.z])

    # Formal charge from RDKit molecule
    total_charge = Chem.GetFormalCharge(mol)

    # Try to infer multiplicity (very rough): assume singlet unless radicals present
    # RDKit tracks total radical electrons
    radical_e = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    multiplicity = 1 if radical_e == 0 else (radical_e + 1)

    return symbols, np.array(coords), total_charge, multiplicity


def _build_pyscf_mol(symbols, coords, config: DFTConfig):
    """Internal helper to build a PySCF Mole object."""
    from pyscf import (  # pylint: disable=import-outside-toplevel,import-error # type: ignore
        gto,
    )

    atom_block = "\n".join(
        f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in zip(symbols, coords)
    )
    mol = gto.Mole()
    mol.atom = atom_block
    mol.charge = int(config.charge)
    mol.spin = int(max(0, config.multiplicity - 1))
    mol.basis = config.basis
    mol.unit = "Angstrom"
    mol.build()
    return mol


def run_dft_dipole(symbols, coords, config: Optional[DFTConfig] = None):
    """Run a single-point DFT calculation in PySCF and compute dipole (Debye)."""
    from pyscf import (  # pylint: disable=import-outside-toplevel,import-error # type: ignore
        dft,
    )

    if config is None:
        config = DFTConfig()

    mol = _build_pyscf_mol(symbols, coords, config)

    # Choose RKS/UKS depending on spin
    mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    mf.xc = config.xc
    mf.grids.level = 3
    mf.conv_tol = 1e-9
    mf.kernel()

    if not mf.converged:
        raise RuntimeError(
            "DFT did not converge. Try a different functional/basis or tighter preoptimization."
        )

    mu_vec = np.array(mf.dip_moment(unit="Debye"))
    mu_mag = float(np.linalg.norm(mu_vec))
    return mu_vec, mu_mag, mf


def get_dft_dipole_moment(
    smiles: str,
    config: Optional[DFTConfig] = None,
) -> float:
    """Compute the dipole moment (in Debye) of a molecule given its SMILES string.

    This function generates a 3D conformation of the molecule using RDKit,
    performs a single-point DFT calculation using PySCF, and computes the dipole moment.

    Args:
        smiles: SMILES string of the molecule.
        config: Optional DFT configuration. If None, defaults will be used.
    """
    # Generate 3D structure
    coords, symbols, total_charge, multiplicity = smiles_to_3d(smiles)
    if config is None:
        config = DFTConfig(charge=total_charge, multiplicity=multiplicity)
    else:
        # Override charge/multiplicity if provided in config
        config.charge = total_charge
        config.multiplicity = multiplicity

    # Run DFT and get dipole moment
    _, mu_mag, _ = run_dft_dipole(symbols, coords, config)
    return mu_mag
