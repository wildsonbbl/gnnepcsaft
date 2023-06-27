from ml_pc_saft import epcsaft_pure_VP, epcsaft_pure_den
from graphdataset import ThermoMLDataset, ThermoML_padded
from jaxopt import LevenbergMarquardt
import jax.numpy as jnp
import jax
import torch
from jax import dlpack as jdlpack
from torch.utils import dlpack as tdlpack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor_to_array(tensor: torch.Tensor) -> jnp.ndarray:
        dlpack = tdlpack.to_dlpack(tensor)
        array = jdlpack.from_dlpack(dlpack)
        return array


path = osp.join("data", "thermoml")
train_dataset = ThermoMLDataset(path, subset="train")
test_dataset = ThermoMLDataset(path, subset="test")

train_dataset = ThermoML_padded(train_dataset, 64)
test_dataset = ThermoML_padded(test_dataset, 16)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

pcsaft_den = jax.vmap(epcsaft_pure_den, (None, 0))
pcsaft_vp = jax.vmap(epcsaft_pure_VP, (None, 0))

solver = LevenbergMarquardt(OF, 200)

for graphs in train_loader:
    graphs = graphs.to(device)
    if graphs.vp:
        vp = tensor_to_array(graphs.vp)
    else:
        vp = None
    rho = tensor_to_array(graphs.rho)
    params, state = solver.run(x0, rho, vp)

def OF(params, rho, vp):
    
    pred_rho = pcsaft_den(params, rho)
    msle = jnp.square(
            jnp.log(jnp.abs(rho[:,-1]) + 1) - jnp.log(jnp.abs(pred_rho) + 1)
        )
    if vp:
        pred_vp = pcsaft_vp(params, vp)
        msle_vp = jnp.square(
            jnp.log(jnp.abs(vp[:,-1]) + 1) - jnp.log(jnp.abs(pred_vp) + 1)
        )
        msle = jnp.concatenate([msle, msle_vp])

    return jnp.mean(msle)
    