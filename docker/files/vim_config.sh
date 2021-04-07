#!/bin/bash

curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

FILES_DIR="$1"

cat $FILES_DIR/.vimrc>>~/.vimrc

vim -c :PlugInstall +qall