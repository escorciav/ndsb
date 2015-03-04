#!/bin/bash
CAFFE_PATH="../caffe/build/tools/extract_features"

if [ -z "$8" ]
then
  echo "Assuming 10 crops per images and batch size 200"
  TEST_ITER=6520
else
  TEST_ITER=$8
fi

if [ -z "$7" ]
then
  CSV_FILE=new_entry.csv
else
  CSV_FILE=$7
fi

if [ -z "$1"  ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$6" ]
then
  echo -n "Usage: test.sh GPU_ID NETWORK_PROTOTXT MODEL_PROTOBIN BLOB "
  echo -n "OUTPUT_FILE OUTPUT_TYPE CSV_FILE(default=new_entry.csv) "
  echo "TEST_ITER(default=6520)"
else
  $CAFFE_PATH $3 $2 $4 $5 $TEST_ITER $6 GPU $1

  echo "Assuming a database of features with 1304000 entries"
  python src/s_save_csv.py $CSV_FILE -i $5
fi
