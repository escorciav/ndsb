#!/bin/bash
CAFFE_PATH="../caffe/build/tools/caffe"

if [ -z "$1"  ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]
then
  echo "Usage: resume.sh GPU_ID SOLVER_PROTOTXT SNAPSHOT_MODEL_PROTOBIN LOGFILE"
else
  $CAFFE_PATH train -gpu $1 -solver $2 -snapshot $3 &>> $4
fi
