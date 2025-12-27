#!/bin/bash

# Configure and validate Nexus connection
# Usage: ./nexus_config.sh <nexus_url> <username> <password>

if [ $# -lt 3 ]; then
    echo "Usage: $0 <nexus_url> <username> <password>"
    echo "Example: $0 http://localhost:8081 jenkins mypassword"
    exit 1
fi

NEXUS_URL="$1"
USERNAME="$2"
PASSWORD="$3"

echo "=========================================="
echo "Nexus Configuration Validator"
echo "=========================================="
echo "URL: $NEXUS_URL"
echo ""

# Test connection
echo "1. Testing Nexus connectivity..."
if curl -s -f "$NEXUS_URL" > /dev/null 2>&1; then
    echo "✓ Nexus is reachable"
else
    echo "✗ Cannot reach Nexus at $NEXUS_URL"
    exit 1
fi

echo ""
echo "2. Testing authentication..."
if curl -s -u "${USERNAME}:${PASSWORD}" "$NEXUS_URL/service/rest/v1/status" > /dev/null 2>&1; then
    echo "✓ Authentication successful"
else
    echo "✗ Authentication failed"
    exit 1
fi

echo ""
echo "3. Checking repositories..."
REPOS=$(curl -s -u "${USERNAME}:${PASSWORD}" \
    "$NEXUS_URL/service/rest/v1/repositories" 2>/dev/null | grep -o '"name":"[^"]*' | cut -d'"' -f4)

if [ -z "$REPOS" ]; then
    echo "✗ No repositories found or unable to retrieve"
    exit 1
fi

echo "✓ Found repositories:"
echo "$REPOS" | while read -r repo; do
    echo "  - $repo"
done

echo ""
echo "4. Checking critical repositories..."
for repo in "raw-hosted" "models-hosted" "docker-hosted"; do
    if echo "$REPOS" | grep -q "$repo"; then
        echo "✓ $repo exists"
    else
        echo "⚠ $repo NOT found - you may need to create it"
    fi
done

echo ""
echo "=========================================="
echo "✓ Configuration validation complete"
echo "=========================================="
