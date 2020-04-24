#!/bin/bash
#
# Setup CPU-only TensorFlow (see README.md)
#
[[ ! -e ~/Envs/tensorflow_cpu ]] && { echo "No tensorflow_cpu -- exiting"; exit 1; }
module load python3/3.7.4
export VIRTUALENVWRAPPER_PYTHON="$(which python3)"
export WORKON_HOME=~/Envs
source ~/.local/bin/virtualenvwrapper.sh
workon tensorflow_cpu
