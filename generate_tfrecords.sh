#!/bin/bash
#
# Generate tfrecord files for the datasets
#
/usr/bin/time python -m datasets.main --jobs=1 --debug \
   |& tee datasets/output.txt
