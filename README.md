# Time-Series Adaptation

Method: adaptation of time-series data.

Steps:

- Download and preprocess datasets (*datasets/generate_tfrecords.py*)
- Optionally view the datasets (*datasets/view_datasets.py*)
- Train models (*main.py* or *kamiak_train.srun*)
- Evaluate models (*main_eval.py* or *kamiak_eval.srun*)

## Training
For example, to train on a synthetic frequency shift dataset with no adaptation:

    ./kamiak_queue.sh test1 --model=flat --method=none --source=freqshift_low --target=freqshift_high --debugnum=1

Note: these examples assume you're using SLURM. If not, you can modify *kamiak_queue.sh* to not queue with sbatch but run with bash.

## Evaluating
For example, to evaluate the above "test1" trained models:

    sbatch kamiak_eval.srun test1 scale-none --match="*scale*-*-flat-[0-9]*" --eval_batch=2048 --jobs=1

Then look at the resulting *results_test1_scale-none.txt* file.
