FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

WORKDIR /

# Set the environment variable for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Preconfigure the time zone to avoid interactive prompts
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    echo "Etc/UTC" > /etc/timezone

# Install basic utilities and dependencies.
RUN apt-get -y update && \
  apt-get -y install tree vim curl wget bzip2 git aria2 rsync && \
  rm -rf /var/lib/apt/lists/*

# Install Miniconda.
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init bash

# Add Conda to PATH.
ENV PATH="/opt/conda/bin:$PATH"

# Install backend requirements.
RUN conda create -y -n worker python=3.12
RUN conda install -y -n worker cudatoolkit=11.8 pytorch-cuda=12.1 pytorch torchvision torchaudio -c pytorch -c nvidia -c conda-forge \ 
    && conda clean -a -y
    
COPY backend/requirements.txt /backend/
RUN /opt/conda/envs/worker/bin/pip install --no-cache-dir -r /backend/requirements.txt

# Install autodockVINA requirements.
WORKDIR /worker/docking
COPY worker/docking/requirements.txt ./
RUN conda create -y -n dock python=3.12 conda
RUN /opt/conda/envs/dock/bin/conda install -y -c conda-forge openbabel numpy tqdm seaborn vina biopython && \
  /opt/conda/envs/dock/bin/conda clean -afy
# RUN /opt/conda/envs/dock/bin/pip install --no-input --no-cache-dir -r requirements.txt
RUN wget https://ccsb.scripps.edu/adfr/download/1038/ADFRsuite_x86_64Linux_1.0.tar.gz
RUN tar zxvf ADFRsuite_x86_64Linux_1.0.tar.gz && \
  cd ADFRsuite_x86_64Linux_1.0 && \
  echo Y | ./install.sh -d /adfrsuite -c 0


# Install annotation requirements.
WORKDIR /annotations
RUN /opt/conda/bin/conda config --add channels defaults && \
  /opt/conda/bin/conda config --add channels bioconda && \
  /opt/conda/bin/conda config --add channels conda-forge
RUN /opt/conda/bin/conda create -y -n annotations python=3.9 conda java-jdk
# TODO: maybe switch to muscle version 3.8.31, which we have elsewhere,
# but did not install properly here.
RUN /opt/conda/envs/annotations/bin/conda install -y \
  hmmer2 hmmer diamond fasttree prodigal blast muscle=3.8.1551 glimmerhmm && \
  /opt/conda/envs/annotations/bin/conda clean -afy


# Install DiffDock requirements.
WORKDIR /worker/diffdock
RUN git clone https://github.com/gcorso/DiffDock.git && \
  cd DiffDock && \
  git checkout c1abe25 && \
  cd -
RUN /opt/conda/bin/conda create -y -n diffdock python=3.8 conda pip
RUN /opt/conda/envs/diffdock/bin/conda install -y pytorch==1.13.0 cpuonly -c pytorch && \
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


# Copy in code.
COPY backend/ /backend/
COPY worker/*.sh /worker/
COPY worker/*.py /worker/
COPY worker/docking/* /worker/docking/

WORKDIR /backend/src

# Make sure to use the exec form of ENTRYPOINT, rather than the shell
# form, so that SIGTERM gets propagated to rq.
# https://medium.com/@tasdikrahman/handling-signals-for-applications-running-in-kubernetes-dc6537f9b542
# https://docs.docker.com/engine/reference/builder/#entrypoint
ENTRYPOINT ["/opt/conda/envs/worker/bin/flask"]