#!/bin/bash

python mmtg/datamodules/preprocess.py \
    --study "$study" \
    --output_path $output_path/$study \
    --boxcox_transformation
