#!/bin/bash
#
# Start up all the kamiak_train_array.srun jobs
#
if [[ -z $1 || -z $2 ]]; then
    echo "Usage: ./kamiak_queue_array.sh suffix dataset <other arguments>"
    echo "Note: there must exist datasets dataset_{a,b0,b1,b2,b3,b4,b5}"
    echo "Note: outputs to kamiak-{models,logs}-suffix"
    exit 1
fi

. kamiak_config.sh

# First argument is the suffix, then the rest are arguments for the training
suffix="$1"
modelFolder="$modelFolder-$suffix"
logFolder="$logFolder-$suffix"

dataset="$2"
shift
shift

echo "Queueing array $target"
sbatch -J "${suffix}_${dataset}" kamiak_train_array.srun "$dataset" \
    --logdir="$logFolder" --modeldir="$modelFolder" "$@"
