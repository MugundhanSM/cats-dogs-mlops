#!/bin/bash

set -e

echo "Building Docker image..."
docker build -t cats-dogs-mlops .

echo "Stopping old containers..."
docker compose down || true

echo "Starting new container..."
docker compose up -d

echo "Deployment completed."
