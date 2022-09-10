#!/bin/bash
# Build image
docker build -t robertdcurrier/ubuntu_node_python3 .
docker push robertdcurrier/ubuntu_node_python3
