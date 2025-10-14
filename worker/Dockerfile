FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

COPY worker/scripts/base_setup.sh /scripts/base_setup.sh
RUN /scripts/base_setup.sh

COPY backend/requirements.txt /backend/
COPY worker/scripts/conda_env_setup.sh /scripts/conda_env_setup.sh
RUN /scripts/conda_env_setup.sh


ENV PATH="/opt/conda/bin:$PATH"

# Note that we are using an old version of google cloud CLI...
RUN conda install -y python=3.9

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
# Install curl for this script, and aria2 and rsync for AF DB downloading.
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-401.0.0-linux-x86_64.tar.gz && \
  tar -xf google-cloud-cli-401.0.0-linux-x86_64.tar.gz && \
  ./google-cloud-sdk/install.sh -q && \
  . /google-cloud-sdk/path.bash.inc

RUN mkdir -p /backend && mkdir -p /worker/docking


# Install autodockVINA requirements.
WORKDIR /worker/docking
COPY worker/docking/requirements.txt ./

# Combine conda operations into a single RUN command to reduce layers
# Use mamba instead of conda for faster resolution
# Pin specific versions to avoid resolution issues

# 7/31/25: BROKEN BY ANACONDA https://app.semanticdiff.com/gh/scikit-learn/scikit-learn/pull/31771/overview
RUN conda install -n base -c conda-forge mamba --override-channels && \
    mamba create -n dock python=3.7 && \
    mamba install -n dock -c conda-forge -c defaults \
        openbabel=3.1.1 \
        numpy=1.21 \
        tqdm \
        seaborn \
        vina \
        biopython>=1.79 && \
    mamba clean -afy && \
    conda clean -afy
RUN wget https://ccsb.scripps.edu/adfr/download/1038/ADFRsuite_x86_64Linux_1.0.tar.gz
RUN tar zxvf ADFRsuite_x86_64Linux_1.0.tar.gz && \
  cd ADFRsuite_x86_64Linux_1.0 && \
  echo Y | ./install.sh -d /adfrsuite -c 0


# Install annotation requirements.
WORKDIR /annotations
# 7/31/25: BROKEN BY ANACONDA https://app.semanticdiff.com/gh/scikit-learn/scikit-learn/pull/31771/overview
# RUN /opt/conda/bin/conda config --add channels defaults && \
#   /opt/conda/bin/conda config --add channels bioconda && \
#   /opt/conda/bin/conda config --add channels conda-forge
# RUN /opt/conda/bin/conda create -y -n annotations python=3.9 conda java-jdk
RUN /opt/conda/bin/conda create -y -n annotations -c bioconda -c conda-forge --override-channels python=3.9 conda java-jdk
# TODO: maybe switch to muscle version 3.8.31, which we have elsewhere,
# but did not install properly here.
RUN /opt/conda/envs/annotations/bin/conda install -y -c bioconda -c conda-forge --override-channels \
  hmmer2 hmmer diamond fasttree prodigal blast muscle=3.8.1551 glimmerhmm && \
  /opt/conda/envs/annotations/bin/conda clean -afy


# Install DiffDock requirements.
WORKDIR /worker/diffdock
RUN git clone https://github.com/gcorso/DiffDock.git && \
  cd DiffDock && \
  git checkout c1abe25 && \
  cd -
# 7/31/25: BROKEN BY ANACONDA https://app.semanticdiff.com/gh/scikit-learn/scikit-learn/pull/31771/overview
RUN /opt/conda/bin/conda create -y -n diffdock -c bioconda -c conda-forge -c pytorch --override-channels python=3.8 conda pip && \
  /opt/conda/envs/diffdock/bin/conda install -y -c bioconda -c conda-forge -c pytorch --override-channels pytorch==1.13.0 cpuonly && \
  /opt/conda/envs/diffdock/bin/conda clean -afy

# Install scipy, then install torch from a specific wheel, not from pip (no-index), then
# install the rest of the dependencies.
RUN /opt/conda/envs/diffdock/bin/python -m pip install --no-cache-dir scipy
RUN /opt/conda/envs/diffdock/bin/python -m pip install --no-cache-dir --no-index \
  torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
RUN /opt/conda/envs/diffdock/bin/python -m pip install --no-cache-dir \
  PyYAML scipy "networkx[default]" biopython rdkit-pypi e3nn spyrmsd pandas biopandas \
  "fair-esm" \
  torch_geometric

# TODO: move this into setup.sh.
RUN /opt/conda/envs/worker/bin/pip --no-cache-dir install openpyxl

# Copy in code.
COPY backend/ /backend/
COPY worker/*.sh /worker/
COPY worker/*.py /worker/
COPY worker/docking/* /worker/docking/

WORKDIR /backend

# Make sure to use the exec form of ENTRYPOINT, rather than the shell
# form, so that SIGTERM gets propagated to rq.
# https://medium.com/@tasdikrahman/handling-signals-for-applications-running-in-kubernetes-dc6537f9b542
# https://docs.docker.com/engine/reference/builder/#entrypoint
ENTRYPOINT ["/opt/conda/envs/worker/bin/flask"]
