# For more information, please refer to https://aka.ms/vscode-docker-python
FROM pytorch/pytorch:latest

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


# Install requirements
RUN conda install conda=23.5.0 -y
RUN conda install mamba -n base -c conda-forge -S -y
RUN pip install --upgrade pip
RUN mamba install pyg -c pyg -y
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
#RUN mamba install cudnn=8.4* cudatoolkit -c conda-forge -c nvidia -y
RUN CONDA_OVERRIDE_CUDA="11.8" mamba install jaxlib=*=*cuda11* jax cuda-nvcc cudnn=8.4* -c conda-forge -c nvidia -y
RUN mamba install -c conda-forge rdkit torchmetrics wandb ml-collections polars absl-py notebook -y

