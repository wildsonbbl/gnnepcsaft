# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9.18-bookworm

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


# Install requirements
RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
# RUN curl -O -L https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip
# RUN curl -O -L https://github.com/zmeri/PC-SAFT/archive/refs/tags/v1.4.1.zip
# RUN unzip -q eigen-master.zip
# RUN unzip -q v1.4.1.zip
# RUN cp -rf eigen-master/. PC-SAFT-1.4.1/externals/eigen
# RUN pip install ./PC-SAFT-1.4.1
# RUN rm -rf PC-SAFT* eigen-master* v1.4.1.zip