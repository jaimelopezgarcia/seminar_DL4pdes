#!/bin/bash

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash


export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm

nvm install v12.18.2

pip install jupyterlab==2.2.8

pip install --upgrade jupyterlab-git

pip install jupyterlab "ipywidgets>=7.5"

jupyter labextension install --no-build @jupyterlab/toc

jupyter labextension install --no-build @jupyter-widgets/jupyterlab-manager

jupyter labextension install --no-build jupyterlab-plotly

jupyter labextension install --no-build @jupyterlab/git


jupyter lab build --dev-build=False --minimize=False
