#!/bin/bash
# Fix permissions for two-user environment (jupyter + petergrabowski_google_com)
# Run this whenever you encounter permission errors

cd /home/jupyter/research/directed_kvcache

echo "Fixing file permissions (making world-writable)..."
find . -type f -exec chmod 666 {} \; 2>/dev/null
echo "Fixing directory permissions (making world-writable + executable)..."
find . -type d -exec chmod 777 {} \; 2>/dev/null

echo "Done. Current permissions:"
ls -la
ls -la results/
