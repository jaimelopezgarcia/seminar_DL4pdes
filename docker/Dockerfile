ARG BASE_IMAGE="quay.io/fenicsproject/stable:current"
FROM $BASE_IMAGE

USER root

ARG FILES_PARENT_DIR_ARG="/opt"
ENV FILES_PARENT_DIR="$FILES_PARENT_DIR_ARG"
ENV FILES_DIR="$FILES_PARENT_DIR/files"

COPY files $FILES_DIR

WORKDIR /home/project




ENV FILES_DIR_PYTORCH="$FILES_PARENT_DIR/pytorch"

COPY files_pytorch $FILES_DIR_PYTORCH




RUN apt-get update && apt-get install --no-install-recommends -y \
    python3.8 \
    python3-pip \
    vim \
    git \
    locate \
    curl &&\
    $FILES_DIR/vim_config.sh $FILES_DIR &&\
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 &&\
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 &&\
    $FILES_DIR/jupyterlab_config.sh &&\
    pip install --upgrade pip &&\
    pip install -r $FILES_DIR/requirements.txt &&\
    pip install -r $FILES_DIR_PYTORCH/requirements.txt &&\

    rm -rf /var/lib/apt/lists/*
