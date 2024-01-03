#!/bin/bash

LOCAL_DATA_PATH=$1
STORAGE_ACCOUNT_NAME=$2
CONTAINER_NAME=$3
BLOB_DIRECTORY_NAME=$4

upload_directory() {
    local dir_path=$1
    local blob_dir_path=$2
    for filepath in "$dir_path"/*; do
	if [ -d "$filepath" ]; then
	    upload_directory "$filepath" "$blob_dir_path/$(basename "$filepath")"
    	elif [ -f "$filepath" ]; then
            echo "Uploading $filepath to $blob_dir_path/$(basename "$filepath")"
	    az storage blob upload --account-name $STORAGE_ACCOUNT_NAME \
				   --container-name $CONTAINER_NAME --file "$filepath" \
				   --name "$blob_dir_path/$(basename "$filepath")"
	fi
    done
}

echo "Started uplading $LOCAL_DATA_PATH to Azure Storage with arguments:"
echo "  --account-name $STORAGE_ACCOUNT_NAME"
echo "  --container-name $CONTAINER_NAME"

echo "Uploading train directory..."
upload_directory "$LOCAL_DATA_PATH/train" "$BLOB_DIRECTORY_NAME/train"
echo "Done!"

echo "Uploading test directory..."
upload_directory "$LOCAL_DATA_PATH/test" "$BLOB_DIRECTORY_NAME/test"
echo "Done!"

echo "Upload complete!"

