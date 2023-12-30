#!/bin/bash
# This script downloads the LIDC-IDRI dataset from the TCIA website.

# URL variables
nbia_url="https://cbiit-download.nci.nih.gov/nbia/releases/ForTCIA/NBIADataRetriever_4.4.1/nbia-data-retriever-4.4.1.deb"
images_url="https://wiki.cancerimagingarchive.net/download/attachments/1966254/TCIA_LIDC-IDRI_20200921.tcia?version=1&modificationDate=1600709265077&api=v2"
annotations_url="https://wiki.cancerimagingarchive.net/download/attachments/1966254/LIDC-XML-only.zip?version=1&modificationDate=1530215018015&api=v2"

# Make directory for the dataset
dataset_dir="/mnt/data"
echo -e "-- Using directory $dataset_dir\n"

# Download the NBIA Data Retriever
nbia_file_path="$dataset_dir/nbia-data-retriever.deb"

echo -e "-- Downloading the NBIA Data Retriever...\n"
wget -O $nbia_file_path $nbia_url
echo -e "-- Downloaded to: $nbia_file_path\n"

# Download the manifest file
images_file_path="$dataset_dir/images.tcia"

echo -e "-- Downloading the manifest file...\n"
wget -O $images_file_path $images_url
echo -e "-- Downloaded to: $images_file_path\n"

# Download the annotations file
annotations_file_path="$dataset_dir/annotations.zip"

echo -e "-- Downloading the annotations file...\n"
wget -O $annotations_file_path $annotations_url
echo -e "-- Downloaded to: $annotations_file_path\n"

# Ensure java is installed
echo -e "-- Checking if java is installed...\n"
if ! command -v java &> /dev/null
then
    echo -e "-- Java is not installed. Installing...\n"
    sudo -S apt-get install default-jre
    echo -e "-- Java installed.\n"
else
    echo -e "-- Java is installed.\n"
fi

# Install the NBIA Data Retriever
echo "-- Installing the NBIA Data Retriever...\n"
sudo -S dpkg -r $nbia_file_path; sudo -S dpkg -i $nbia_file_path
echo -e "-- Installed.\n"

# Download the dataset
output_dir="$dataset_dir/images"
manifest_path="$images_file_path"

echo -e "-- Downloading the dataset...\n"
/opt/nbia-data-retriever/nbia-data-retriever --cli $manifest_path -d $output_dir -v -f 
echo -e "-- Downloaded to: $output_dir\n" 

