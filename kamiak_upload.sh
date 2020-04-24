#!/bin/bash
#
# Upload files to high performance cluster
#
. kamiak_config.sh

# Note both have trailing slashes
from="$localdir"
to="$remotessh:$remotedir"

# Make SLURM log folder
ssh "$remotessh" "mkdir -p \"$remotedir/slurm_logs\""

# Copy only select files
rsync -Pahuv --include="./" --include="*.py" --include="*.sh" --include="*.srun" \
    --include="datasets/" --include="datasets/*" --include="datasets/tfrecords/*" \
    --include="*.tfrecord" --include="*.tar.gz" --include="*.zip" \
    --exclude="*" "$from" "$to"
