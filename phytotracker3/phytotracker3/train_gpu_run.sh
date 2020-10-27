#!/bin/bash
echo "Docker building GPU-enabled version..."
docker build -t phytotracker3 .

# Up time
docker run  \
        -u $(id -u):$(id -g) \
        -e DISPLAY=unix$DISPLAY \
        -it \
        --runtime=nvidia \
        --gpus all \
        --rm \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v ~/src/apps/phytotracker3:/phytotracker3 \
        -v /data:/data \
       phytotracker3
