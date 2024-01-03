#!/bin/bash

LOCAL_DATA_PATH=$1
STORAGE_ACCOUNT_NAME=$2
CONTAINER_NAME=$3
SAS_TOKEN=$4

REMOTE_STORAGE_PATH="https://$STORAGE_ACCOUNT_NAME.blob.core.windows.net/$CONTAINER_NAME"

echo "Uploading $LOCAL_DATA_PATH to $REMOTE_STORAGE_PATH"
echo "Uploading train directory..."
azcopy copy "$LOCAL_DATA_PATH/train" "$REMOTE_STORAGE_PATH?$SAS_TOKEN" --recursive=true 

echo "Uploading test directory..."
azcopy copy "$LOCAL_DATA_PATH/test" "$REMOTE_STORAGE_PATH?$SAS_TOKEN" --recursive=true

echo "Upload complete!"

