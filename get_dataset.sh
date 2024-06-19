#!/bin/bash

DEFAULT_DIRECTORY="./data"

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: download_dataset.sh [DIRECTORY]"
    echo "Downloads the dataset and extracts it to the given [DIRECTORY]"
    echo ""
    echo "  [DIRECTORY] The directory to extract the dataset to. Default: $DEFAULT_DIRECTORY"
    exit 0
fi

DIRECTORY="${1:-$DEFAULT_DIRECTORY}"
FILES=(
    "_README.md"
    "metadata.zip"
    "quebec_trees_dataset_2021-05-28.zip"
    "quebec_trees_dataset_2021-06-17.zip"
    "quebec_trees_dataset_2021-07-21.zip"
    "quebec_trees_dataset_2021-08-18.zip"
    "quebec_trees_dataset_2021-09-02.zip"
    "quebec_trees_dataset_2021-09-28.zip"
    "quebec_trees_dataset_2021-10-07.zip"
)

mkdir -p "$DIRECTORY"

for FILE in "${FILES[@]}"; do
    # Download
    if [ ! -f "$DIRECTORY/$FILE" ]; then
        URL=$(printf "https://zenodo.org/records/8148479/files/%s?download=1" "$FILE")
        echo "Downloading $FILE ..."
        wget -O "$DIRECTORY/$FILE" "$URL" || echo "Failed to download $FILE" && continue
        echo "... done"
    else
        echo "$FILE already exists, skipping download"
    fi
    
    # Extract
    if [[ "$FILE" == *.zip ]]; then
        if [ -d "$DIRECTORY/$(basename -s .zip "$FILE")" ]; then
            echo "Directory $(basename -s .zip "$FILE") already exists, skipping extraction"
        else
            echo "Extracting $FILE ..."
            unzip -o "$DIRECTORY/$FILE" -d "$DIRECTORY" || echo "Failed to extract $FILE" && continue
            echo "... done"
        fi
        
        # Delete
        echo "Deleting $FILE ..."
        rm "$DIRECTORY/$FILE" || echo "Failed to delete $FILE"
        echo "... done"
    fi
done