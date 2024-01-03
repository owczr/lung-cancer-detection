#!/bin/bash

input_dir="/home/jo-engineers-thesis/dataset/images/images/LIDC-IDRI"
output_dir="/mnt/data/images/images/LIDC-IDRI"

#mv "$input_dir"/* "$output_dir"/
for file in "$input_dir"/*; do
	mv $file "$output_dir"/
done
