#!/bin/bash

script_to_run=''

while getopts "sv" opt; do
    case $opt in
        s)
            echo "Running search ..."
            while IFS=' ' read -r NETWORK_TYPE RHO_P_THIN_SET PARAM_SET PARAM_NAME PARAM_VALUE; do
                echo "NETWORK_TYPE=$NETWORK_TYPE"
                echo "RHO_P_THIN_SET=$RHO_P_THIN_SET"
                echo "PARAM_SET=$PARAM_SET"
                echo "PARAM_NAME=$PARAM_NAME"
                echo "PARAM_VALUE=$PARAM_VALUE"
                echo "--------------------------"

                json_path="./utils/rho_p_thin_sets/$RHO_P_THIN_SET.json"
                if [[ ! -f "$json_path" ]]; then
                    echo "Error: $RHO_P_THIN_SET json file not found at $json_path"
                    continue
                fi

                # Get column indices for 'rho' and 'p_thin' (legacy csv)
                # rho_col=$(awk -F',' 'NR==1 {for (i=1;i<=NF;i++) if ($i=="rho") print i}' "$csv_path")
                # pthin_col=$(awk -F',' 'NR==1 {for (i=1;i<=NF;i++) if ($i=="p_thin") print i}' "$csv_path")

                # Count unique non-empty entries in each column
                num_rho=$(jq '.rho | length' "$json_path")
                num_pthin=$(jq '.p_thin | length' "$json_path")

                if [[ "$num_rho" -eq 0 || "$num_pthin" -eq 0 ]]; then
                    echo "Error: rho or p_thin arrays are empty in $json_path"
                    continue
                fi

                total_jobs=$((num_rho * num_pthin))

                echo "Found $num_rho rho values and $num_pthin p_thin values."
                echo "Submitting job array of size $total_jobs (=$num_rho×$num_pthin) at increments of 10."

                sbatch \
                --array=0-$((total_jobs - 1)):10 \
                --export=ALL,NETWORK_TYPE=$NETWORK_TYPE,RHO_P_THIN_SET=$RHO_P_THIN_SET,PARAM_SET=$PARAM_SET,PARAM_NAME=$PARAM_NAME,PARAM_VALUE=$PARAM_VALUE \
                scripts/simulations_array.sh
                # Sleep to not overload the job scheduler
                sleep 60
            done < scripts/vars.txt
            ;;

        v)
            echo "Running visualization ..."
            while IFS=' ' read -r NETWORK_TYPE RHO_P_THIN_SET PARAM_SET PARAM_NAME PARAM_VALUE; do
                echo "NETWORK_TYPE=$NETWORK_TYPE"
                echo "RHO_P_THIN_SET=$RHO_P_THIN_SET"
                echo "PARAM_SET=$PARAM_SET"
                echo "PARAM_NAME=$PARAM_NAME"
                echo "PARAM_VALUE=$PARAM_VALUE"
                echo "--------------------------"

                echo "Submitting job array"  
                sbatch --export=ALL,NETWORK_TYPE=$NETWORK_TYPE,RHO_P_THIN_SET=$RHO_P_THIN_SET,PARAM_SET=$PARAM_SET,PARAM_NAME=$PARAM_NAME,PARAM_VALUE=$PARAM_VALUE scripts/visualization.sh
                # Sleep to not overload the job scheduler
                sleep 60
            done < scripts/vars.txt
            ;;

        *)
            echo "Usage: $0 [-s | -v]"
            exit 1
            ;;
    esac
done

