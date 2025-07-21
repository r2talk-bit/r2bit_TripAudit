#!/bin/bash

# Exit on error
set -e

# Build the Docker image
echo "Building Docker image..."
docker build -t gcr.io/wa-agentic/r2bittripaudit:latest .

# Push to Google Cloud Registry
echo "Pushing to Google Cloud Registry..."
docker push gcr.io/wa-agentic/r2bittripaudit:latest

echo "Done! Image successfully built and pushed to gcr.io/wa-agentic/r2bittripaudit:latest"
