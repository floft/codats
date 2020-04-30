# Multi-Source Time Series Domain Adaptation

Steps:

- Convert to .tfrecord files for TensorFlow (*./generate_tfrecords.sh*)
- Train models (*main.py* or *kamiak_train.srun*)
- Evaluate models (*main_eval.py* or *kamiak_eval.srun*)
- Analyze results (*analysis.py*)

## Installation

This requires the following packages (*module load* for Kamiak):

    module load cuda/10.1.105 cudnn/7.6.4.38_cuda10.1 python3/3.7.4
    pip install --user --upgrade pip
    export PATH="$HOME/.local/bin:$PATH"
    pip3 install --user --upgrade pip
    pip3 install --user --upgrade numpy cython
    pip3 install --user --upgrade tensorflow-gpu pillow lxml jupyter matplotlib pandas scikit-learn scipy tensorboard rarfile tqdm pyyaml grpcio absl-py

    # If using --moving_average (typically tensorflow-addons, but that errors at the moment with TF 2.2)
    pip3 install --user git+https://github.com/tensorflow/addons.git@r0.9

For the CPU-only jobs like *kamiak_train_simple.srun*:

    module load python3/3.7.4
    export PATH="$HOME/.local/bin:$PATH"
    pip3 install --user --upgrade virtualenvwrapper
    export VIRTUALENVWRAPPER_PYTHON="$(which python3)"
    mkdir -p ~/Envs
    export WORKON_HOME=~/Envs
    source ~/.local/bin/virtualenvwrapper.sh
    mkvirtualenv -p python3 tensorflow_cpu

    which pip # check it's ~/Envs/tensorflow_cpu/bin/pip
    which python3 # check it's ~/Envs/tensorflow_cpu/bin/python3

    # Note: we don't use --user for virtual environments
    pip install --upgrade numpy cython
    # Note: here it's "tensorflow" not "tensorflow-gpu" -- the rest is the same.
    pip install --upgrade tensorflow pillow lxml jupyter matplotlib pandas scikit-learn scipy tensorboard rarfile tqdm pyyaml grpcio absl-py

    # If using --moving_average
    pip install git+https://github.com/tensorflow/addons.git@r0.9

## Training

    sbatch -J train kamiak_train.srun adapt

## Evaluating

    sbatch -J eval kamiak_eval.srun adapt

Then look at the resulting *results/results_\*.txt* file or analyze with *analysis.py*.
