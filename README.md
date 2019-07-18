# Time-Series Adaptation

Method: adaptation of time-series data.

Steps:

- Generate synthetic datasets (*datasets/generate_trivial_datasets.py*)
- Convert to .tfrecord files for TensorFlow (*datasets/generate_tfrecords.py*)
- Optionally view the datasets (*datasets/view_datasets.py*)
- Train models (*main.py* or *kamiak_train.srun*)
- Evaluate models (*main_eval.py* or *kamiak_eval.srun*)
- Analyze results (*analysis.py*)

## Training
For example, to train on a synthetic frequency shift dataset with CyCADA
adaptation:

    ./kamiak_queue.sh test1 --model=fcn --method=cycada --source=freqshift_a --target=freqshift_b5 --debugnum=1

Note: these examples assume you're using SLURM. If not, you can modify *kamiak_queue.sh* to not queue with sbatch but run with bash.

Training 5 runs of each method and upper/lower bounds on synthetic datasets:

    # Lower bound and multiple adaptation methods
    for method in none dann cyclegan cyclegan_dann cycada; do
    for i in 1 2 3 4 5; do
    ./kamiak_queue_array.sh runwalk freqshift_phase --model=fcn --method=$method --debugnum=$i
    ./kamiak_queue_array.sh runwalk freqscale_phase --model=fcn --method=$method --debugnum=$i
    ./kamiak_queue_array.sh runwalk jumpmean_phase --model=fcn --method=$method --debugnum=$i
    done
    done

    # Upper bound
    for shift in 0 1 2 3 4 5; do
    ./kamiak_queue.sh runwalk --model=fcn --method=none --source=freqshift_phase_b$shift --best_source --notrain_on_source_valid --debugnum=$i
    ./kamiak_queue.sh runwalk --model=fcn --method=none --source=freqscale_phase_b$shift --best_source --notrain_on_source_valid --debugnum=$i
    ./kamiak_queue.sh runwalk --model=fcn --method=none --source=jumpmean_phase_b$shift --best_source --notrain_on_source_valid --debugnum=$i
    done

## Evaluating
For example, to evaluate the above "test1" trained models (pass *--best_source*
for evaluating when not using target domain):

    sbatch kamiak_eval.srun test1 freqshift-cycada --match="freqshift_a-freqshift_b5-fcn-cycada-[0-9]*" --eval_batch=2048 --jobs=1

Then look at the resulting *results_test1_freqshift-cycada.txt* file.

Evaluating all the tests on the synthetic datasets (remove *--last* to evaluate
the best model instead of the last model):

    network=fcn
    for dataset in freqshift_phase freqscale_phase jumpmean_phase; do
        for method in none dann cyclegan cyclegan_dann cycada; do
            sbatch kamiak_eval_array.srun runwalk01 last-${dataset}-${method} ${dataset} ${network} ${method} --last
        done
        sbatch kamiak_eval_sourceonly.srun runwalk01 last-${dataset}-upper ${dataset} ${network} none --last
    done
