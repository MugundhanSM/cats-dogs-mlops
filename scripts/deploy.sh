#!/bin/bash

set -e

IMAGE="ghcr.io/mugundhansm/cats-dogs-mlops:latest"

echo "Pulling latest image from registry..."
docker pull $IMAGE

echo "Stopping old containers..."
docker compose down || true

echo "Starting new container..."
docker compose up -d

echo "Deployment completed."

docker ps