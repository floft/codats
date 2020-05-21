# Multi-Source Deep Domain Adaptation with Weak Supervision for Time-Series Sensor Data

Steps:
- Download data and convert to .tfrecord files for TensorFlow
  (*./generate_tfrecords.sh*)
- Train models (*main.py* or *kamiak_train_\*.srun*)
- Evaluate models (*main_eval.py* or *kamiak_eval_\*.srun*)
- Analyze results (*analysis.py*)

## Installation

We require the following packages (*module load* for Kamiak). Adjust for your
computer setup.

    module load cuda/10.1.105 cudnn/7.6.4.38_cuda10.1 python3/3.7.4
    pip install --user --upgrade pip
    export PATH="$HOME/.local/bin:$PATH"
    pip3 install --user --upgrade pip
    pip3 install --user --upgrade numpy cython
    pip3 install --user --upgrade tensorflow-gpu pillow lxml jupyter matplotlib pandas scikit-learn scipy tensorboard rarfile tqdm pyyaml grpcio absl-py

    # If using --moving_average (typically tensorflow-addons, but that errors at the moment with TF 2.2)
    pip3 install --user git+https://github.com/tensorflow/addons.git@r0.9

Or, to use only the CPU, set up as follows and modify the train scripts to
source *kamiak_tensorflow_cpu.sh*.

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

## Usage

### Example

Train a CoDATS-WS model (i.e. using DA-WS) on person 14 of the UCI HAR dataset
and adapt to person 19.

    python3 main.py \
        --logdir=example-logs --modeldir=example-models \
        --method=daws --dataset=ucihar --sources=14 \
        --target=19 --uid=0 --debugnum=0 --gpumem=0

Then evaluate that model on the holdout test data, outputting the results to a
YAML file.

    mkdir -p results
    python3 main_eval.py \
        --logdir=example-logs --modeldir=example-models \
        --jobs=1 --gpus=1 --gpumem=0 \
        --match="ucihar-0-daws-[0-9]*" --selection="best_target" \
        --output_file=results/results_example_best_target-ucihar-0-daws.yaml

Note: there are a number of other options (e.g. --ensemble=5,
--moving_average), models (e.g. --model=inceptiontime), methods (e.g.
--method=dann_pad), datasets (e.g. --dataset=wisdm_at), etc. implemented that
you can experiment with beyond what was included in the paper.

### All Experiments

To run all the experiments in the paper, see the Slurm scripts for training
(*kamiak_train_\*.srun*) and evaluation (*kamiak_eval_\*.srun*). Tweak
for your cluster. Then, you can run the experiments, e.g. for the single-source
experiments (excluding the upper bound, which has its own scripts):

    sbatch -J train kamiak_train_ssda.srun adapt
    sbatch -J eval kamiak_eval_ssda.srun adapt

### Analysis

Then look at the resulting *results/results_\*.yaml* files or analyze with
*analysis.py*.
