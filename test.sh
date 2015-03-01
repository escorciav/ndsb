#!/bin/bash
CAFFE_PATH="../caffe/build/tools/extract_features"

if [ -z "$7" ]
then
  echo "Assuming 10 crops per images and batch size 200"
  TEST_ITER=6520
else
  TEST_ITER=$7
fi

if [ -z "$1"  ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$6" ]
then
  echo "Usage: test.sh GPU_ID SOLVER_PROTOTXT MODEL_PROTOBIN BLOB OUTPUT_FILE OUTPUT_TYPE TEST_ITER(default=6520)"
else
  $CAFFE_PATH $3 $2 $4 $5 $TEST_ITER $6 GPU $1

  echo "Assuming a database of features with 1304000 entries"
  python src/s.save_csv.py new_entry.csv -i $5
fi
