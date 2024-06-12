#!/bin/bash

RECORDS=$(cat records.txt)
FILES=$(cat files.txt)
SIZE=xs #CHANGEME

while read record;
do
  while read file;
  do
    # Skip base.py file
    if [[ "$file" == *"base.py" ]]; then
    echo "Skipping benchmark for $file"
    continue
    fi

    # Extract the module path from the file path
    MODULE_PATH=$(echo "$file" | sed 's/\//./g' | sed 's/\.py$//')

    # Find and list all classes in the module
    echo "Running benchmarks for classes in $MODULE_PATH"
    CLASSES=$(python3 -c "import sys; import inspect; from importlib import import_module; mod = import_module('$MODULE_PATH'); print(' '.join(cls for cls, obj in inspect.getmembers(mod, inspect.isclass) if obj.__module__ == mod.__name__ and not cls.startswith('Base')))")
    IFS=' ' read -ra CLS <<< "$CLASSES"

    for cls in "${CLS[@]}"
    do
      CLASS_PATH="$MODULE_PATH.$cls"
      FILENAME="benchmark_results_${cls}.txt"
      IFS='/' read -r -a array <<< "$record"
      FILEDIR=$(echo "${array[-1]}")
      mkdir -p "/waveform/results-$SIZE/$FILEDIR"
      echo "Benchmarking $CLASS_PATH for $FILEDIR"
      ./waveform_benchmark.py -r "$record" -f "$CLASS_PATH" > "/waveform/results-$SIZE/$FILEDIR/$FILENAME"
    done
  done < files.txt
done < records.txt
