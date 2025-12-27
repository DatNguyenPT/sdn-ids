#!/bin/bash

# Sign artifacts and create checksums
# Usage: ./sign_artifacts.sh <artifact_directory> <output_directory>

if [ $# -lt 2 ]; then
    echo "Usage: $0 <artifact_directory> <output_directory>"
    echo "Example: $0 ./models ./signatures"
    exit 1
fi

ARTIFACT_DIR="$1"
OUTPUT_DIR="$2"

if [ ! -d "$ARTIFACT_DIR" ]; then
    echo "Error: Artifact directory not found: $ARTIFACT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Signing Artifacts"
echo "=========================================="
echo "Source: $ARTIFACT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Create SHA256 checksums
echo "Creating SHA256 checksums..."
cd "$ARTIFACT_DIR"
sha256sum * > "${OUTPUT_DIR}/checksums.sha256"
cd - > /dev/null

echo "✓ Checksums created"
echo ""
cat "${OUTPUT_DIR}/checksums.sha256"
echo ""

# Sign with GPG if available
if command -v gpg &> /dev/null; then
    echo "Signing with GPG..."
    for file in "$ARTIFACT_DIR"/*; do
        if [ -f "$file" ]; then
            echo "Signing: $(basename "$file")"
            gpg --batch --armor --sign --detach-sign "$file" \
                -o "${OUTPUT_DIR}/$(basename "$file").asc" || echo "GPG signing skipped"
        fi
    done
    echo "✓ GPG signatures created"
else
    echo "GPG not available, skipping GPG signing"
fi

echo ""
echo "✓ Artifact signing complete"
ls -lh "$OUTPUT_DIR"
