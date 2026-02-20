#!/bin/bash

docker pull mugundhansm/cats-dogs-mlops:latest
docker compose down
docker compose up -d
