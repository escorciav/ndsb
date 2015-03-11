#!/bin/bash
CAFFE_PATH="../caffe/build/tools/caffe"

if [ -z "$1"  ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]
then
  echo -n "Usage: train.sh GPU_ID SOLVER_PROTOTXT NETWORK_PROTOTXT "
  echo "OUTPUT_FILE SNAPSHOT(optional for resume training)"
else
  if [ -z "$5" ]
  then
    $CAFFE_PATH train -gpu $1 -model $3 -solver $2 &>> $4
  else
    $CAFFE_PATH train -gpu $1 -model $3 -solver $2 -snapshot $5 &>> $4
  fi
fi
