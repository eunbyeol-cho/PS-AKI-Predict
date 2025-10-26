#!/bin/bash

# Define the list of studies
# studies=('SNUH' 'SNUBH' 'BRMH-KNUH-CAUH' 'AMC')
studies=('BRMH' 'KNUH' 'CAUH')

# Loop through each study and process data
for study in "${studies[@]}"
do
    echo "Processing $study..."
    # Run the Python script with appropriate parameters
    python mmtg/datamodules/preprocess_six_hosp.py \
        --study "$study" \
        --output_path "/home/data_storage/mimic3/snuh/20240609/$study" \
        --boxcox_transformation

    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "$study processed successfully."
    else
        echo "Error processing $study. Check logs for details."
    fi
done

echo "All processing complete."
