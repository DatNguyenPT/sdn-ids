#!/bin/bash

# Upload artifacts to Nexus repository
# Usage: ./upload_to_nexus.sh <file_path> <repository_name> <nexus_url> <username> <password>

if [ $# -lt 5 ]; then
    echo "Usage: $0 <file_path> <repository_name> <nexus_url> <username> <password>"
    echo "Example: $0 ./model.h5 models-hosted http://localhost:8081 jenkins mypassword"
    exit 1
fi

FILE_PATH="$1"
REPOSITORY="$2"
NEXUS_URL="$3"
USERNAME="$4"
PASSWORD="$5"

if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File not found: $FILE_PATH"
    exit 1
fi

FILENAME=$(basename "$FILE_PATH")
UPLOAD_URL="${NEXUS_URL}/repository/${REPOSITORY}/${FILENAME}"

echo "=========================================="
echo "Uploading to Nexus Repository"
echo "=========================================="
echo "File: $FILE_PATH"
echo "Repository: $REPOSITORY"
echo "Target URL: $UPLOAD_URL"
echo ""

curl -v -u "${USERNAME}:${PASSWORD}" \
    --upload-file "$FILE_PATH" \
    "$UPLOAD_URL"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Upload successful!"
    echo "Artifact available at: $UPLOAD_URL"
else
    echo ""
    echo "✗ Upload failed!"
    exit 1
fi
