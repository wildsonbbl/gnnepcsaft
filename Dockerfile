# For more information, please refer to https://aka.ms/vscode-docker-python
FROM paperspace/gradient-base:pt112-tf29-jax0317-py39-20230125

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


# Install requirements
RUN pip install --upgrade pip
RUN pip install rdkit torchmetrics ml-collections polars clu jraph jaxopt
RUN pip install -U "ray[tune]"
RUN pip install torch_geometric