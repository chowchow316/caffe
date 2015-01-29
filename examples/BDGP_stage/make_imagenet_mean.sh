#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

./build/tools/compute_image_mean examples/BDGP_stage/BDGP_train_lmdb \
  data/BDGP_stage/BDGP_mean.binaryproto

echo "Done."
