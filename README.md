# Multi-Source Deep Domain Adaptation with Weak Supervision for Time-Series Sensor Data

Domain adaptation (DA) offers a valuable means to reuse data and models for new
problem domains. However, robust techniques have not yet been considered for
time series data with varying amounts of data availability. In our paper, we
make three main contributions to fill this gap. First, we propose a novel
*Convolutional deep Domain Adaptation model for Time Series data (CoDATS)* that
significantly improves accuracy and training time over state-of-the-art DA
strategies on real-world sensor data benchmarks. By utilizing data from multiple
source domains, we increase the usefulness of CoDATS to further improve
accuracy over prior single-source methods, particularly on complex time series
datasets that have high variability between domains. Second, we propose a novel
*Domain Adaptation with Weak Supervision (DA-WS)* method by utilizing weak
supervision in the form of target-domain label distributions, which may be
easier to collect than additional data labels. Third, we perform comprehensive
experiments on diverse real-world datasets to evaluate the effectiveness of our
domain adaptation and weak supervision methods. Results show that CoDATS for
single-source DA significantly improves over the state-of-the-art methods, and
we achieve additional improvements in accuracy using data from multiple source
domains and weakly supervised signals.

Preprint: https://arxiv.org/abs/2005.10996

Overview:

- Download data and convert to .tfrecord files for TensorFlow
  (*./generate_tfrecords.sh*)
- Train models (*main.py* or *kamiak_train_\*.srun*)
- Evaluate models (*main_eval.py* or *kamiak_eval_\*.srun*)
- Analyze results (*analysis.py*)

## Installation

We require the following packages (*module load* used on Kamiak, WSU's cluster).
Adjust for your computer setup.

    module load cuda/10.1.105 cudnn/7.6.4.38_cuda10.1 python3/3.7.4
    pip install --user --upgrade pip
    export PATH="$HOME/.local/bin:$PATH"
    pip3 install --user --upgrade pip
    pip3 install --user --upgrade numpy cython
    pip3 install --user --upgrade tensorflow-gpu pillow lxml jupyter matplotlib pandas scikit-learn scipy tensorboard rarfile tqdm pyyaml grpcio absl-py

    # If using --moving_average or F1 score metrics (typically tensorflow-addons, but that errors at the moment with TF 2.2)
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

    # If using --moving_average or F1 score metrics
    pip install git+https://github.com/tensorflow/addons.git@r0.9

## Usage

### Example

Train a CoDATS-WS model (i.e. using DA-WS) on person 14 of the UCI HAR dataset
and adapt to person 19.

    python3 main.py \
        --logdir=example-logs --modeldir=example-models \
        --method=daws --dataset=ucihar --sources=14 \
        --target=19 --uid=0 --debugnum=0 --gpumem=0

Note: if passing multiple sources (comma separated, e.g. ``--sources=1,2,3``)
with DA-WS, you probably want to pass ``--batch_division=sources`` to make sure
you have enough target samples in the batch to estimate the predicted label
distribution (see the Appendix in the paper).

Then evaluate that model on the holdout test data, outputting the results to a
YAML file.

    mkdir -p results
    python3 main_eval.py \
        --logdir=example-logs --modeldir=example-models \
        --jobs=1 --gpus=1 --gpumem=0 \
        --match="ucihar-0-daws-[0-9]*" --selection="best_target" \
        --output_file=results/results_example_best_target-ucihar-0-daws.yaml

Note: there are a number of other options (e.g. ``--ensemble=5``,
``--moving_average``), models (e.g. ``--model=inceptiontime``), methods (e.g.
``--method=dann_pad``), datasets (e.g. ``--dataset=wisdm_at``), etc. implemented that
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

In addition, there are scripts for visualizing the datasets
(*datasets/view_datasets.py*), viewing dataset statistics
(*dataset_statistics.py*), and displaying or plotting the class balance of the
data (*class_balance.py*, *class_balance_plot.py*).

## Navigating the Code

Here is an outline of the key elements of the code.

### Models

In the paper we propose using a particular feature extractor, task classifier,
and domain classifier for CoDATS. However, to support variations, the model is
split into a variety of classes using inheritance.

- *models.py:DannModel* -- trivial class which is used in *methods.py* for
  ``--method=dann``, just inherits from *DannModelBase* and *CnnModelBase* (which
  then calls *FcnModelMaker* to create the model since ``--model=fcn`` is the
  default option)
- *models.py:DannModelBase* -- insert the gradient reversal layer
  (*FlipGradient*) between the feature extractor and the domain classifier
- *models.py:FcnModelMaker* -- create the CoDATS feature extractor
- *models.py:CodatsModelMakerBase* -- create the CoDATS task and domain
  classifiers

The two most notable baseline models VRADA and R-DANN are also implemented here.

- *models.py:VradaModel* and *models.py:RDannModel* -- trivial classes used in
  *methods.py* for ``--method=vrada`` and ``--method=rdann`` respectively, just inherit
  from *DannModelBase* and *RnnModelBase*
- *models.py:RnnModelBase* -- create the VRADA or R-DANN feature extractor, task
  classifier, and domain classifier
- *models.py:VradaFeatureExtractor* -- either use a VRNN or LSTM (depending on
  if this is VRADA or R-DANN) followed by some dense layers
- *vrnn.py:VRNN* -- implementation of the variational RNN used in VRADA

### Adaptation Methods

In the paper we propose using a multi-source version of domain adversarial
training for CoDATS and also a weak supervision method (DA-WS) for CoDATS-WS.
These can be selected with ``--method=dann`` or ``--method=daws``.

- *methods.py:MethodDann* - create a *DannModel*, handle using labeled data from
  multiple source domains and unlabeled data from the target domain
- *methods.py:MethodDaws* - inherits from *MethodDann*, but simulates knowing
  the target-domain label proportions by estimating on the source training data,
  also implements the DA-WS loss

Then, similarly we have the baselines ``--method=vrada`` and ``--method=rdann``.

- *methods.py:MethodRDann* - same as *MethodDann* but uses R-DANN model instead
  of the CoDATS model
- *methods.py:MethodVrada* - inherits from *MethodDann*, but implements the
  VRNN KL divergence and negative log likelihood losses


## Citation

If you use this code in your research, please consider citing our paper:

    @inproceedings{wilson2020codats,
        title={Multi-Source Deep Domain Adaptation with Weak Supervision for Time-Series Sensor Data},
        author={Wilson, Garrett and Doppa, Janardhan Rao and Cook, Diane J.},
        booktitle={KDD},
        year={2020}
    }
