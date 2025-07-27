#!/bin/bash

while IFS=' ' read -r PARAM_SET PARAM_NAME PARAM_VALUE; do
    echo "PARAM_SET=$PARAM_SET"
    echo "PARAM_NAME=$PARAM_NAME"
    echo "PARAM_VALUE=$PARAM_VALUE"
    echo "--------------------------"

    echo "Submitting job array"  
    sbatch --array=0-39:10 --export=ALL,PARAM_SET=$PARAM_SET,PARAM_NAME=$PARAM_NAME,PARAM_VALUE=$PARAM_VALUE scripts/run_array.sh
    # Sleep to not overload the job scheduler
    sleep 60
done < scripts/params.txt