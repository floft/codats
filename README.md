# Time-Series Adaptation

Method: adaptation of time-series data.

Steps:

- Generate synthetic datasets (*datasets/generate_trivial_datasets.py*)
- Convert to .tfrecord files for TensorFlow (*datasets/generate_tfrecords.py*)
- Optionally view the datasets (*datasets/view_datasets.py*)
- Train models (*main.py* or *kamiak_train_\*.srun*)
- Evaluate models (*main_eval.py* or *kamiak_eval_\*.srun*)
- Analyze results (*analysis.py*)

## Installation

This requires the following packages:

    pip install --user numpy cython
    pip install --user tf-nightly-gpu-2.0-preview pillow lxml jupyter matplotlib pandas sklearn scipy tb-nightly rarfile POT dtw

## Training

    sbatch -J train_synthetic kamiak_train_synthetic.srun synthetic
    sbatch -J train_real kamiak_train_real.srun real

## Evaluating

    sbatch -J eval_synthetic kamiak_eval_synthetic.srun synthetic
    sbatch -J eval_real kamiak_eval_real.srun real

Then look at the resulting *results_\*.txt* file or analyze with *analysis.py*.
