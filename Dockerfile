# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-bookworm

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


# Install requirements
RUN python -m pip install --upgrade pip
COPY requirements.txt .
# RUN python -m pip install -r requirements.txt
# RUN sh install_pcsaft.sh