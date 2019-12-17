# Multi-Source Time Series Domain Adaptation

(add details later)

Steps:

- Convert to .tfrecord files for TensorFlow (*datasets/generate_tfrecords.py*)
- Optionally view the datasets, look at class balance, etc. (*datasets/{view_datasets,class_balance}.py*)
- Hyperparameter tune (*kamiak_{train,eval}_tune.srun*)
- Train models (*main.py* or *kamiak_train_real.srun*)
- Evaluate models (*main_eval.py* or *kamiak_eval_real.srun*)
- Analyze results (*analysis.py*)

## Installation

This requires the following packages (*module load* for Kamiak):

    # Once they install cudnn/7.6.4.38_cuda10.1.... change to that
    #module load cuda/10.1.105 cudnn/7.5.1.10_cuda10.1 python3/3.7.4
    module load cuda/10.0.130 cudnn/7.6.4.38_cuda10.0 python3/3.7.4
    pip3 install --user --upgrade pip
    export PATH="$HOME/.local/bin:$PATH"
    pip3 install --user --upgrade numpy cython
    #pip3 install --user --upgrade tensorflow-gpu==2.1.0rc1 pillow lxml jupyter matplotlib pandas sklearn scipy tensorboard rarfile tqdm pyyaml POT dtw
    pip3 install --user tensorflow-gpu==2.0.0

## Hyperparameter tuning
If you want to re-run hyperparameter tuning, you can train, eval, then pick
the best hyperparameters and update the

    sbatch -J tune kamiak_train_tune.srun tune
    sbatch -J tune_eval kamiak_eval_tune.srun tune
    # Update hyperparameter_{source,target} in pick_multi_source.py
    ./hyperparameters.py --selection=best_source
    ./hyperparameters.py --selection=best_target
    datasets/pick_multi_source.py  # then, update kamiak_{train,eval}_real.srun

## Training

    sbatch -J train_real kamiak_train_real.srun real

## Evaluating

    sbatch -J eval_real kamiak_eval_real.srun real

Then look at the resulting *results/results_\*.txt* file or analyze with *analysis.py*.
