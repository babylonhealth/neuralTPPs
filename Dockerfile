FROM babylon.jfrog.io/classic-dev-docker-virtual/babylonhealth/envoy-preflight:latest as builder
FROM babylon.jfrog.io/classic-dev-docker-virtual/babylonhealth/ml-kubeflow:neural-tpps-base
COPY --from=builder /envoy-preflight /envoy-preflight

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    texlive-latex-extra \
    texlive-fonts-recommended dvipng \
    && rm -rf /var/lib/apt/lists/*

# Install Requirements
RUN pip uninstall -y enum34
COPY requirements.txt /neural-tpps/requirements.txt
RUN pip install -r /neural-tpps/requirements.txt
COPY requirements-git.txt /neural-tpps/requirements-git.txt
WORKDIR /torchsearchsorted
RUN pip install -r /neural-tpps/requirements-git.txt
WORKDIR /root

# Install Code
COPY . /neural-tpps

# Run
WORKDIR /neural-tpps
RUN pip install -e .

RUN chmod -R +x runs/*
RUN useradd -ms /bin/bash air

ENTRYPOINT [ "/envoy-preflight"]
