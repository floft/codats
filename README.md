# Time-Series Adaptation

Method: adaptation of time-series data.

Steps:

- Download and preprocess datasets (*datasets/generate_tfrecords.py*)
- Optionally view the datasets (*datasets/view_datasets.py*)
- Train models (*main.py* or *kamiak_train.srun*)
- Evaluate models (*main_eval.py* or *kamiak_eval.srun*)

## Training
For example, to train on USPS to MNIST with no adaptation:

    ./kamiak_queue.sh test1 --model=vada_small --source=usps --target=mnist --method=none

Note: these examples assume you're using SLURM. If not, you can modify *kamiak_queue.sh* to not queue with sbatch but run with bash.

## Evaluating
For example, to evaluate the above "test1" trained models:

    sbatch kamiak_eval.srun test1 --eval_batch=2048 --jobs=1
