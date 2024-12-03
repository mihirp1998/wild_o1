#!/bin/bash

# Base directory path
BASE_DIR="/grogu/user/mprabhud/papers_o1/attention_papers/"

# Function to process each directory
process_directory() {
    local dir="$1"
    echo "Processing directory: $dir"
    
    # Initialize counters
    available=0
    not_available=0

    available_pdf=0
    not_available_pdf=0


    # Loop through all files in the directory
    for file in "$dir"/*; do
        if [ -d "$file" ]; then
            # If it's a directory, recursively process it
            # Check if main.md exists in subdirectory
            if [ -f "$file/main.md" ]; then
                ((available++))
            else
                ((not_available++))
                echo "No main.md found in: $file"
            fi
            if [ -f "$file/main.pdf" ]; then
                ((available_pdf++))
            else
                ((not_available_pdf++))
                echo "No main.pdf found in: $file"
            fi            
        fi
    done

    echo "Summary for $dir:"
    echo "Available main.md files: $available"
    echo "Missing main.md files: $not_available"
    echo "Available main.pdf files: $available_pdf"
    echo "Missing main.pdf files: $not_available_pdf"
}

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory $BASE_DIR does not exist"
    exit 1
fi

# Start processing from base directory
process_directory "$BASE_DIR"
