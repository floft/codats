#!/bin/bash
#
# Start up all the kamiak_train.srun jobs for all datasets in the cross validation
#
if [[ -z $1 ]]; then
    echo "Usage:"
    echo "  ./kamiak_queue_all.sh suffix <other arguments>"
    echo "  Note: outputs to kamiak-{models,logs}-suffix"
    exit 1
fi

. kamiak_config.sh

# First argument is the suffix, then the rest are arguments for the training
suffix="$1"
modelFolder="$modelFolder-$suffix"
logFolder="$logFolder-$suffix"
shift

echo "Queueing $target"
sbatch -J "$suffix" kamiak_train.srun \
    --logdir="$logFolder" --modeldir="$modelFolder" "$@"
