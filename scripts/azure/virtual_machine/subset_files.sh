#!/bin/bash

SOURCE_DIR=$1
DESTINATION_DIR=$2
NUMBER_OF_FILES=$3

echo "Creating a subset of $NUMBER_OF_FILES files from $SOURCE_DIR in $DESTINATION_DIR"
find "$SOURCE_DIR" -type f | shuf -n "$NUMBER_OF_FILES" | xargs -I {} cp {} "$DESTINATION_DIR"
echo "Done!"
