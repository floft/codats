# CoDATS: A Convolutional Deep Domain Adaptation Method for Time Series Sensor Data

The code for our paper, performing domain adaptation on time series sensor
datasets. Note: below we have both synthetic and real datasets, but in the paper
we only include the results on the real datasets.

Steps:

- Generate synthetic datasets (*datasets/generate_trivial_datasets.py*)
- Convert to .tfrecord files for TensorFlow (*datasets/generate_tfrecords.py*)
- Optionally view the datasets, look at class balance, etc. (*datasets/{view_datasets,class_balance}.py*)
- Train models (*main.py* or *kamiak_train_\*.srun*)
- Evaluate models (*main_eval.py* or *kamiak_eval_\*.srun*)
- Analyze results (*analysis.py*)

## Installation

This requires the following packages:

    pip install --user numpy cython
    pip install --user tf-nightly-gpu-2.0-preview pillow lxml jupyter matplotlib pandas sklearn scipy tb-nightly rarfile POT dtw

## Training

    sbatch -J train_real kamiak_train_real.srun real
    sbatch -J train_synthetic kamiak_train_synthetic.srun synthetic

## Evaluating

    sbatch -J eval_real kamiak_eval_real.srun real
    sbatch -J eval_synthetic kamiak_eval_synthetic.srun synthetic

Then look at the resulting *results_\*.txt* file or analyze with *analysis.py*.

## Supplementary materials experiments
Train:

    for i in {10..100..10}; do
        sbatch -J seq$i kamiak_seqlen_real.srun seqlen$i --trim_time_steps=$i
    done
    for i in {1..6}; do
        sbatch -J subset$i kamiak_seqlen_real.srun subset$i --feature_subset=$i
    done

Test:

    for i in {10..100..10}; do
        sbatch -J eval_seq$i kamiak_evalseqlen_real.srun seqlen$i
    done
    for i in {1..6}; do
        sbatch -J eval_subset$i kamiak_evalseqlen_real.srun subset$i --feature_subset=$i
    done

Generate plots:

    ./analysis.py
