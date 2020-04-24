#!/bin/bash
#
# Setup GPU-based TensorFlow (see README.md)
#
# We install tensorflow-gpu as --user (see README.md), so we don't need to open
# a virtual environment here
#
module load cuda/10.1.105 cudnn/7.6.4.38_cuda10.1 python3/3.7.4
