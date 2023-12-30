#!/bin/bash
# This script downloads the LIDC-IDRI dataset from the TCIA website.

# URL variables
nbia_url="https://cbiit-download.nci.nih.gov/nbia/releases/ForTCIA/NBIADataRetriever_4.4.1/nbia-data-retriever-4.4.1.deb"
images_url="https://wiki.cancerimagingarchive.net/download/attachments/1966254/TCIA_LIDC-IDRI_20200921.tcia?version=1&modificationDate=1600709265077&api=v2"
annotations_url="https://wiki.cancerimagingarchive.net/download/attachments/1966254/LIDC-XML-only.zip?version=1&modificationDate=1530215018015&api=v2"

# Make directory for the dataset
dataset_dir="~/dataset"

echo -e "-- Making directory for the dataset...\n"
mkdir -p $dataset_dir
echo -e "-- Directory made.\n"

# Download the NBIA Data Retriever
nbia_filename="nbia-data-retriever.deb"

echo -e "-- Downloading the NBIA Data Retriever...\n"
wget -P $dataset_dir -O $nbia_filename $nbia_url
echo -e "-- Downloaded.\n"

# Download the manifest file
images_filename="images.tcia"

echo -e "-- Downloading the manifest file...\n"
wget -P $dataset_dir -O $images_filename $images_url
echo -e "-- Downloaded.\n"

# Download the annotations file
annotations_filename="annotations.zip"

echo -e "-- Downloading the annotations file...\n"
wget -P $dataset_dir -O $annotations_filename $annotations_url
echo -e "-- Downloaded.\n"

# Download the dataset
output_dir="~/dataset/images"

echo -e "-- Downloading the dataset...\n"
/opt/nbia-data-retriever/nbia-data-retriever --cli --manifestPath $images_filename --outputDirectory $output_dir
echo -e "-- Downloaded.\n" 

# Change directory to the dataset
cd $dataset_dir
echo -e "-- Changed directory to the dataset.\n"

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
sudo -S dpkg -r $nbia_filename; sudo -S dpkg -i $nbia_filename
echo -e "-- Installed.\n"
