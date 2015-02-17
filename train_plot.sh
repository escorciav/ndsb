#!/bin/bash
PLOT_TOOL_PATH="../caffe/tools/extra/plot_training_log.py.example"

if [ -z "$1"  ] || [ -z "$2" ]
then
  echo "Usage: train_plot.sh OUTPUT_PREFIX LOGFILE"
else
  for i in 0 2 6
  do
    $PLOT_TOOL_PATH $i $1_$i.png $2
  done
fi
