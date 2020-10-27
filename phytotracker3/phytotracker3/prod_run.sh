#!/bin/bash
echo "Docker building phytotracker3..."

# Make sure we aren't running...
echo "Bringing Docker down..."
docker-compose down

# Pull the latest
echo "Pulling the latest from GitHub..."
git pull

# Build image
echo "Docker building..."
docker build -t phytotracker3 .

# Up time
echo "PRODUCTION ENVIRONMENT ENABLED!"
export UID=${UID}
export GID=${GID}
docker-compose -f docker-compose.yml up -d
