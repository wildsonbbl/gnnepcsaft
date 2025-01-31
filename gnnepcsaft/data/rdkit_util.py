"Module for helper functions with rdkit"
from rdkit import Chem

# pylint: disable = no-name-in-module
from rdkit.Chem.Fragments import fr_COO2, fr_Imine, fr_isocyan, fr_isothiocyan
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD


# pylint: disable = invalid-name
def inchitosmiles(InChI, with_hydrogen, kekulize):
    "Transform InChI to a SMILES."
    mol = Chem.MolFromInchi(InChI)

    # pylint: disable = no-member
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    smiles = Chem.MolToSmiles(mol)
    return smiles


def smilestoinchi(smiles, with_hydrogen=False, kekulize=False):
    "Transform SMILES to InChI."
    # pylint: disable = no-member
    mol = Chem.MolFromSmiles(smiles)

    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    inchi = Chem.MolToInchi(mol)
    return inchi


def assoc_number(inchi: str):
    "Calculates the number of H-bond acceptors/donors"

    exceptions = (
        "InChI=1S/H2O/h1H2",
        "InChI=1S/Cl2/c1-2",
        "InChI=1S/F2/c1-2",
    )  # From Esper et al. (2023) 10.1021/acs.iecr.3c02255
    if inchi in exceptions:
        return 1, 1
    mol = Chem.MolFromInchi(inchi, removeHs=False)
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
