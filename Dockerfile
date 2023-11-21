# For more information, please refer to https://aka.ms/vscode-docker-python
FROM gcr.io/kaggle-gpu-images/python:latest

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


# Install requirements
RUN python -m pip install --upgrade pip
RUN curl -O -L https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip
RUN curl -O -L https://github.com/zmeri/PC-SAFT/archive/refs/tags/v1.4.1.zip
RUN unzip eigen-master.zip
RUN unzip v1.4.1.zip
RUN cp -r eigen-master/. PC-SAFT-1.4.1/external/eigen
RUN pip install PC-SAFT-1.4.1
RUN rm -rf PC-SAFT* eigen-master* v1.4.1.zip