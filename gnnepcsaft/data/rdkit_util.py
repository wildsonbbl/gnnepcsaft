"Module for helper functions with rdkit"

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# pylint: disable = no-name-in-module
from rdkit.Chem.Fragments import fr_COO2  # type: ignore
from rdkit.Chem.Fragments import fr_Imine, fr_isocyan, fr_isothiocyan  # type: ignore
from rdkit.Chem.rdMolDescriptors import (
    CalcExactMolWt,
    CalcNumHBA,
    CalcNumHBD,
    CalcNumRings,
    CalcNumRotatableBonds,
)

RDLogger.DisableLog("rdApp.*")  # type: ignore


# pylint: disable = invalid-name
def inchitosmiles(
    InChI: str, with_hydrogen: bool = False, kekulize: bool = False
) -> str:
    """Transform InChI to a SMILES.

    Args:
        InChI (str): InChI
        with_hydrogen (bool): Add hydrogens to the molecule
        kekulize (bool): Kekulize the molecule
    """
    mol = Chem.MolFromInchi(InChI)
    if mol is None:
        raise ValueError("InChI is not valid")

    # pylint: disable = no-member
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    smiles = Chem.MolToSmiles(mol)
    return smiles


def smilestoinchi(
    smiles: str, with_hydrogen: bool = False, kekulize: bool = False
) -> str:
    """Transform SMILES to InChI.

    Args:
        smiles (str): SMILES
        with_hydrogen (bool): Add hydrogens to the molecule
        kekulize (bool): Kekulize the molecule
    """
    # pylint: disable = no-member
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES is not valid")

    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    inchi = Chem.MolToInchi(mol)
    return inchi  # type: ignore


def assoc_number(inchi: str) -> Tuple[int, int]:
    """Calculates the number of H-bond acceptors/donors

    Args:
        inchi (str): InChI
    """

    exceptions = (
        "InChI=1S/H2O/h1H2",
        "InChI=1S/Cl2/c1-2",
        "InChI=1S/F2/c1-2",
    )  # From Esper et al. (2023) 10.1021/acs.iecr.3c02255
    if inchi in exceptions:
        return 1, 1
    mol = Chem.MolFromInchi(inchi, removeHs=False)
    if mol is None:
        raise ValueError("InChI is not valid")
    mol = Chem.AddHs(mol)
    na = CalcNumHBA(mol)
    nb = CalcNumHBD(mol)
    if na > 0:
        n_coo = fr_COO2(mol)
        n_Imine = fr_Imine(mol)
        n_isocyanates = fr_isocyan(mol)
        n_isothiocyanates = fr_isothiocyan(mol)
        n_priamide = len(
            mol.GetSubstructMatches(
                Chem.AddHs(
                    Chem.MolFromInchi("InChI=1S/CH3NO/c2-1-3/h1H,(H2,2,3)"),
                    onlyOnAtoms=[1, 2],
                )
            )
        )
        n_sulfuro = len(
            mol.GetSubstructMatches(
                Chem.AddHs(Chem.MolFromSmiles("S(=O)(=O)O"), onlyOnAtoms=[3])
            )
        )

        na -= (
            n_coo + n_priamide + n_Imine - n_isocyanates - n_isothiocyanates + n_sulfuro
        )

    return na, nb


def mw(inchi: str) -> float:
    """Calcultes molecular weight.

    Args:
        inchi (str): InChI
    """
    try:
        mol = Chem.MolFromInchi(inchi, removeHs=False, sanitize=False)
        mol_weight = CalcExactMolWt(mol)
    except (TypeError, ValueError):
        mol_weight = 0

    return mol_weight


def ECFP(smiles: str, radius: int = 3, nBits: int = 3072) -> np.ndarray:
    """Calculates ECFP fingerprints.

    Args:
        smiles (str): SMILES
        radius (int): Radius
        nBits (int): Number of bits
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)  # type: ignore
        fp = np.array([fp], dtype=np.int8)
    except (TypeError, ValueError):
        fp = np.zeros((1, nBits), dtype=np.int8)
    return fp


def ring_count(smiles: str) -> int:
    """Calculates the number of rings.

    Args:
        smiles (str): SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        _ring_count = CalcNumRings(mol)
    except (TypeError, ValueError):
        _ring_count = 0
    return _ring_count


def rbond_count(smiles: str) -> int:
    """Calculates the number of rotatable bonds.

    Args:
        smiles (str): SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        _rbond_count = CalcNumRotatableBonds(mol)
    except (TypeError, ValueError):
        _rbond_count = 0
    return _rbond_count


def atom_count(smiles: str) -> List[int]:
    """Count the number of each atom in a molecule.

    Args:
        smiles (str): SMILES
    """
    atom_count_list = [0] * 119
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        for atom in mol.GetAtoms():
            atom_count_list[atom.GetAtomicNum()] += 1
    except (TypeError, ValueError):
        pass
    return atom_count_list


def get_dipole_moment(smiles: str) -> float:
    """
    Calculates the dipole moment of a molecule from its SMILES string.

    Args:
        smiles: The SMILES string of the molecule.

    Returns:
        The magnitude of the dipole moment in Debye.
    """
    # Create a molecule object from SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    # Add hydrogens, as they are crucial for dipole moment calculation
    mol = Chem.AddHs(mol)

    # Generate a 3D conformation using the ETKDG algorithm
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == -1:  # type: ignore
        raise ValueError("Could not generate 3D conformation for the molecule.")

    # Optimize the geometry using the MMFF94 force field
    AllChem.MMFFOptimizeMolecule(mol)  # type: ignore

    # Calculate Gasteiger partial charges for each atom
    AllChem.ComputeGasteigerCharges(mol)  # type: ignore

    # Get the first (and only) conformer
    conformer = mol.GetConformer()

    # Initialize the dipole moment vector (μ)
    dipole_vector = np.zeros(3, dtype=np.float64)  # A 3D vector [x, y, z]

    # The dipole moment is the sum of (partial charge * position vector) for each atom
    # μ = Σ(q_i * r_i)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        # Get the partial charge (q_i)
        charge = float(atom.GetProp("_GasteigerCharge"))
        # Get the 3D coordinates (r_i)
        position = np.array(conformer.GetAtomPosition(idx))
        # Add the contribution of this atom to the total dipole vector
        dipole_vector += charge * position

    # The result is in units of (electron charge * Angstrom)
    # To convert to Debye, we use the conversion factor: 1 e·Å = 4.803 Debye
    conversion_factor = 4.803

    # Calculate the magnitude of the vector and apply the conversion
    dipole_magnitude_debye = (
        np.linalg.norm(dipole_vector).astype(np.float64) * conversion_factor
    )
    return dipole_magnitude_debye


@dataclass
class ConformerDipoleConfig:
    """Configuration for conformer dipole moment calculations."""

    smiles: str
    n_confs: int = 50
    prune_rms_thresh: float = 0.5
    seed: int = 0
    temperature: float = 298.15
    use_boltzmann: bool = True
    charge_model: str = "gasteiger"  # ou "mmff"


def _generate_conformers(config: ConformerDipoleConfig):
    mol = Chem.MolFromSmiles(config.smiles)
    if mol is None:
        raise ValueError("SMILES inválido.")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()  # type: ignore
    params.randomSeed = config.seed
    params.pruneRmsThresh = config.prune_rms_thresh
    conf_ids = AllChem.EmbedMultipleConfs(  # type: ignore
        mol,
        numConfs=config.n_confs,
        params=params,
    )
    if not conf_ids:
        raise ValueError("Falha ao gerar confôrmeros.")
    return mol, conf_ids


def _optimize_and_get_energies(mol) -> np.ndarray:
    opt_results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)  # type: ignore
    return np.array([e for (_cid, e) in opt_results], dtype=np.float64)


def _get_charge_function(mol, charge_model: str):
    if charge_model.lower() == "gasteiger":
        AllChem.ComputeGasteigerCharges(mol)  # type: ignore
        return lambda idx: float(mol.GetAtomWithIdx(idx).GetProp("_GasteigerCharge"))
    if charge_model.lower() == "mmff":
        mp = AllChem.MMFFGetMoleculeProperties(mol)  # type: ignore
        return mp.GetMMFFPartialCharge  # type: ignore
    raise ValueError("charge_model deve ser 'gasteiger' ou 'mmff'.")


def _compute_dipole_vectors(mol, conf_ids, get_charge):
    n_atoms = mol.GetNumAtoms()
    dipole_vectors = np.zeros((len(conf_ids), 3), dtype=np.float64)
    conv = 4.803
    for i, cid in enumerate(conf_ids):
        conf = mol.GetConformer(int(cid))
        vec = np.zeros(3, dtype=np.float64)
        for a in range(n_atoms):
            q = get_charge(a)
            pos = conf.GetAtomPosition(a)
            vec += q * np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        dipole_vectors[i] = vec * conv
    return dipole_vectors


def _boltzmann_weights(energies: np.ndarray, temperature: float) -> np.ndarray:
    R = 0.001987204258  # kcal/mol/K
    deltaE = energies - energies.min()
    weights = np.exp(-deltaE / (R * temperature))
    weights /= weights.sum()
    return weights


def get_conformer_dipole_distribution(
    smiles: str,
    config: Optional[ConformerDipoleConfig] = None,
) -> Dict[str, Any]:
    """
    Compute dipole moments for a conformer ensemble and related statistics.

    Args:
      smiles (str): SMILES string of the molecule.
      config (ConformerDipoleConfig, optional): Calculation parameters. If None,
        a default configuration is created using the provided SMILES.

    Returns:
      Output (Dict[str, Any]): Dictionary with:
        - dipole_vectors: Dipole vectors (Debye) for each conformer (shape: n_confs x 3).
        - dipole_magnitudes: Dipole magnitudes (Debye) for each conformer.
        - energies_kcal: MMFF energies (kcal/mol) for each conformer.
        - weights: Boltzmann weights (if use_boltzmann=True), uniform otherwise.
        - mean_vector: Arithmetic mean dipole vector (Debye).
        - mean_magnitude: Arithmetic mean dipole magnitude (Debye).
        - boltzmann_vector: Boltzmann-weighted mean dipole vector (Debye) or NaNs if disabled.
        - boltzmann_mean_magnitude: Boltzmann-weighted mean dipole
          magnitude (Debye) or NaN if disabled.
        - charge_model: Charge model used ("gasteiger" or "mmff").
    """
    if config is None:
        config = ConformerDipoleConfig(smiles=smiles)
    mol, conf_ids = _generate_conformers(config)
    energies = _optimize_and_get_energies(mol)
    get_charge = _get_charge_function(mol, config.charge_model)
    dipole_vectors = _compute_dipole_vectors(mol, conf_ids, get_charge)
    dipole_magnitudes = np.linalg.norm(dipole_vectors, axis=1)
    mean_vector = dipole_vectors.mean(axis=0)
    mean_magnitude = float(dipole_magnitudes.mean())

    if config.use_boltzmann:
        weights = _boltzmann_weights(energies, config.temperature)
        boltzmann_vector = np.tensordot(weights, dipole_vectors, axes=1)
        boltzmann_mean_magnitude = float(np.linalg.norm(boltzmann_vector))
    else:
        weights = np.full(len(dipole_vectors), 1.0 / len(dipole_vectors))
        boltzmann_vector = np.full(3, float("nan"))
        boltzmann_mean_magnitude = float("nan")

    return {
        "dipole_vectors": dipole_vectors,
        "dipole_magnitudes": dipole_magnitudes,
        "energies_kcal": energies,
        "weights": weights,
        "mean_vector": mean_vector,
        "mean_magnitude": mean_magnitude,
        "boltzmann_vector": boltzmann_vector,
        "boltzmann_mean_magnitude": boltzmann_mean_magnitude,
        "charge_model": config.charge_model,
    }
