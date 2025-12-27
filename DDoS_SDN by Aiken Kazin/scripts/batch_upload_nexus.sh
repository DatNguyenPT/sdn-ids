#!/bin/bash

# Batch upload artifacts to Nexus
# Usage: ./batch_upload_nexus.sh <config_file>
# Config file format:
# file_path:repository:nexus_url:username:password
# Example:
# ./models/LSTM_FL.h5:models-hosted:http://localhost:8081:jenkins:password
# ./logs.tar.gz:raw-hosted:http://localhost:8081:jenkins:password

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file>"
    echo ""
    echo "Config file format (one per line):"
    echo "file_path:repository:nexus_url:username:password"
    echo ""
    echo "Example config.txt:"
    echo "  ./models/LSTM_FL.h5:models-hosted:http://localhost:8081:jenkins:password"
    echo "  ./logs.tar.gz:raw-hosted:http://localhost:8081:jenkins:password"
    exit 1
fi

CONFIG_FILE="$1"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Batch Uploading to Nexus"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo ""

SUCCESS_COUNT=0
FAILED_COUNT=0

while IFS=':' read -r FILE_PATH REPOSITORY NEXUS_URL USERNAME PASSWORD; do
    # Skip empty lines and comments
    [[ -z "$FILE_PATH" ]] && continue
    [[ "$FILE_PATH" =~ ^# ]] && continue
    
    # Trim whitespace
    FILE_PATH=$(echo "$FILE_PATH" | xargs)
    REPOSITORY=$(echo "$REPOSITORY" | xargs)
    NEXUS_URL=$(echo "$NEXUS_URL" | xargs)
    USERNAME=$(echo "$USERNAME" | xargs)
    PASSWORD=$(echo "$PASSWORD" | xargs)
    
    if [ ! -f "$FILE_PATH" ]; then
        echo "✗ File not found: $FILE_PATH"
        ((FAILED_COUNT++))
        continue
    fi
    
    FILENAME=$(basename "$FILE_PATH")
    UPLOAD_URL="${NEXUS_URL}/repository/${REPOSITORY}/${FILENAME}"
    
    echo "Uploading: $FILENAME to $REPOSITORY"
    
    if curl -s -u "${USERNAME}:${PASSWORD}" \
        --upload-file "$FILE_PATH" \
        "$UPLOAD_URL" > /dev/null 2>&1; then
        echo "✓ Success: $FILENAME"
        ((SUCCESS_COUNT++))
    else
        echo "✗ Failed: $FILENAME"
        ((FAILED_COUNT++))
    fi
    
done < "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "Upload Summary"
echo "=========================================="
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAILED_COUNT"
echo ""

if [ $FAILED_COUNT -eq 0 ]; then
    echo "✓ All uploads completed successfully"
    exit 0
else
    echo "✗ Some uploads failed"
    exit 1
fi
