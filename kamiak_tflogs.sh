#!/bin/bash
#
# Download the TF logs every once in a while to keep TensorBoard updated
# Then run: tensorboard  --logdir logs/
#
. kamiak_config.sh

# Note both have trailing slashes
from="$remotessh:$remotedir"
to="$localdir"

# TensorFlow logs
while true; do
    # * Exclude the large model files
    # * --inplace so we don't get "file created after file even though it's
    #   lexicographically earlier" in TensorBoard, which basically makes it
    #   never update without restarting TensorBoard

    rsync -Pahuv \
        --inplace \
        --include="$logFolder*/" --include="$logFolder*/*" --include="$logFolder*/*/*" \
        --include="$modelFolder*/" --include="$modelFolder*/*" --include="$modelFolder*/*/*" \
        --exclude="*" --exclude="*/" "$from" "$to"
    sleep 30
done
