"""Group-contribution utilities for FEOS PC-SAFT."""

from pathlib import Path
from typing import Dict, List

from feos import EquationOfState  # pyright: ignore[reportAttributeAccessIssue]
from feos import GcParameters  # pyright: ignore[reportAttributeAccessIssue]
from feos import SegmentRecord  # pyright: ignore[reportAttributeAccessIssue]
from feos import SmartsRecord  # pyright: ignore[reportAttributeAccessIssue]


def parameters_gc_pcsaft(smiles: str) -> List[float]:
    """
    Calculates PC-SAFT parameters with Group Contribution method.

    Args:
        smiles (str): SMILES of the compound

    Returns:
        out (List[float]): Estimated PC-SAFT parameters in the order
            [m, sigma, epsilon_k, kappa_ab, epsilon_k_ab, mu, na, nb].
    """

    parameters = GcParameters.from_smiles(
        [smiles],
        SmartsRecord.from_json(
            str(
                Path(__file__).resolve().parent.parent.parent
                / "data/gc_pcsaft/sauer2014_smarts.json"
            )
        ),
        SegmentRecord.from_json(
            str(
                Path(__file__).resolve().parent.parent.parent
                / "data/gc_pcsaft/rehner2023_hetero.json"
            )
        ),
    )
    eos = EquationOfState.pcsaft(parameters)
    gc_parameters = eos.parameters
    assert isinstance(gc_parameters, Dict)

    m = gc_parameters.get("m")
    sigma = gc_parameters.get("sigma")
    e = gc_parameters.get("epsilon_k")
    mu = gc_parameters.get("mu")
    kab = gc_parameters.get("kappa_ab")
    eab = gc_parameters.get("epsilon_k_ab")
    na = gc_parameters.get("na")
    nb = gc_parameters.get("nb")

    return [
        m.item() if m else 0.0,
        sigma.item() if sigma else 0.0,
        e.item() if e else 0.0,
        kab.item() if kab else 0.0,
        eab.item() if eab else 0.0,
        mu.item() if mu else 0.0,
        na.item() if na else 0.0,
        nb.item() if nb else 0.0,
    ]
