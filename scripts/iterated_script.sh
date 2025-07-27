#!/bin/bash

# declare -a PARAMS=(40)
# declare -a PARAM_NAME=("gamma")
# declare -a PARAM_SET=("param_set_1")

# for PARAM in "${PARAMS[@]}"
# do
#     sbatch --export=ALL,PARAM="$PARAM",PARAM_NAME="$PARAM_NAME",PARAM_SET=$PARAM_SET scripts/mpi_test.sh
# done

script_to_run=''

while getopts "sv" opt; do
  case $opt in
    s)
      echo "Running search ..."
      script_to_run='mpi_test.sh'
      ;;
    v)
      echo "Running visualization ..."
      script_to_run='visualization.sh'
      ;;
    *)
      echo "Usage: $0 [-s | -v]"
      exit 1
      ;;
  esac
done


input_file="scripts/params.txt"

# Read the first two lines as strings
{ IFS= read -r PARAM_SET && IFS= read -r PARAM_NAME; }  < "$input_file"

echo "Parameter Set: $PARAM_SET"
echo "Parameter Name: $PARAM_NAME"

# Now read the rest as floats
tail -n +3 "$input_file" | while read -r PARAM; do
    echo "Processing $PARAM_NAME: $PARAM"
    sbatch --export=ALL,PARAM=$PARAM,PARAM_NAME=$PARAM_NAME,PARAM_SET=$PARAM_SET scripts/$script_to_run
done

