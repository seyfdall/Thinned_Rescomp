#!/bin/bash

script_to_run=''

while getopts "sv" opt; do
    case $opt in
        s)
            echo "Running search ..."
            while IFS=' ' read -r RHO_P_THIN_SET PARAM_SET PARAM_NAME PARAM_VALUE; do
                echo "RHO_P_THIN_SET=$RHO_P_THIN_SET"
                echo "PARAM_SET=$PARAM_SET"
                echo "PARAM_NAME=$PARAM_NAME"
                echo "PARAM_VALUE=$PARAM_VALUE"
                echo "--------------------------"

                csv_path="./utils/rho_p_thin_sets/$RHO_P_THIN_SET.csv"
                if [[ ! -f "$csv_path" ]]; then
                    echo "Error: $RHO_P_THIN_SET csv file not found at $csv_path"
                    continue
                fi

                # Get column indices for 'rho' and 'p_thin'
                rho_col=$(awk -F',' 'NR==1 {for (i=1;i<=NF;i++) if ($i=="rho") print i}' "$csv_path")
                pthin_col=$(awk -F',' 'NR==1 {for (i=1;i<=NF;i++) if ($i=="p_thin") print i}' "$csv_path")

                if [[ -z "$rho_col" || -z "$pthin_col" ]]; then
                    echo "Error: Could not find required columns 'rho' and 'p_thin' in $csv_path"
                    continue
                fi

                # Count unique non-empty entries in each column
                num_rho=$(awk -F',' -v col="$rho_col" 'NR>1 && $col != "" {seen[$col]++} END {print length(seen)}' "$csv_path")
                num_pthin=$(awk -F',' -v col="$pthin_col" 'NR>1 && $col != "" {seen[$col]++} END {print length(seen)}' "$csv_path")

                total_jobs=$((num_rho * num_pthin))

                echo "Found $num_rho unique rho values and $num_pthin unique p_thin values."
                echo "Submitting job array of size $total_jobs (=$num_rho×$num_pthin) at increments of 10."

                echo "Submitting job array"  
                sbatch --array=0-$((total_jobs - 1)):10 --export=ALL,RHO_P_THIN_SET=$RHO_P_THIN_SET,PARAM_SET=$PARAM_SET,PARAM_NAME=$PARAM_NAME,PARAM_VALUE=$PARAM_VALUE scripts/simulations_array.sh
                # Sleep to not overload the job scheduler
                sleep 60
            done < scripts/vars.txt
            ;;

        v)
            echo "Running visualization ..."
            while IFS=' ' read -r RHO_P_THIN_SET PARAM_SET PARAM_NAME PARAM_VALUE; do
                echo "RHO_P_THIN_SET=$RHO_P_THIN_SET"
                echo "PARAM_SET=$PARAM_SET"
                echo "PARAM_NAME=$PARAM_NAME"
                echo "PARAM_VALUE=$PARAM_VALUE"
                echo "--------------------------"

                echo "Submitting job array"  
                sbatch --export=ALL,RHO_P_THIN_SET=$RHO_P_THIN_SET,PARAM_SET=$PARAM_SET,PARAM_NAME=$PARAM_NAME,PARAM_VALUE=$PARAM_VALUE scripts/visualization.sh
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

