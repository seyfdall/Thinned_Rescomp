#!/bin/bash

script_to_run=''

while getopts "sv" opt; do
    case $opt in
        s)
            echo "Running search ..."
            while IFS=' ' read -r PARAM_SET PARAM_NAME PARAM_VALUE; do
                echo "PARAM_SET=$PARAM_SET"
                echo "PARAM_NAME=$PARAM_NAME"
                echo "PARAM_VALUE=$PARAM_VALUE"
                echo "--------------------------"

                echo "Submitting job array"  
                sbatch --array=0-39:10 --export=ALL,PARAM_SET=$PARAM_SET,PARAM_NAME=$PARAM_NAME,PARAM_VALUE=$PARAM_VALUE scripts/simulations_array.sh
                # Sleep to not overload the job scheduler
                sleep 60
            done < scripts/params.txt
            ;;

        v)
            echo "Running visualization ..."
            while IFS=' ' read -r PARAM_SET PARAM_NAME PARAM_VALUE; do
                echo "PARAM_SET=$PARAM_SET"
                echo "PARAM_NAME=$PARAM_NAME"
                echo "PARAM_VALUE=$PARAM_VALUE"
                echo "--------------------------"

                echo "Submitting job array"  
                sbatch --export=ALL,PARAM_SET=$PARAM_SET,PARAM_NAME=$PARAM_NAME,PARAM_VALUE=$PARAM_VALUE scripts/visualization.sh
                # Sleep to not overload the job scheduler
                sleep 60
            done < scripts/params.txt
            ;;

        *)
            echo "Usage: $0 [-s | -v]"
            exit 1
            ;;
    esac
done

