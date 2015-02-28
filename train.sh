#!/bin/bash
CAFFE_PATH="../caffe/build/tools/caffe"

if [ -z "$1"  ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]
then
  echo "Usage: train.sh GPU_ID SOLVER_PROTOTXT NETWORK_PROTOTXT OUTPUT_FILE"
else
  $CAFFE_PATH train -gpu $1 -model $3 -solver $2 &>> $4
fi
