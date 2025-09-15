# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-bookworm

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


# Install requirements
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# COPY requirements.txt .
# COPY requirements-torch.txt .
# RUN uv venv
# RUN uv pip install -r requirements-torch.txt
# RUN uv pip install -r requirements.txt