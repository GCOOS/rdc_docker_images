#!/bin/bash
# Build image
docker build -t robertdcurrier/ubuntu_tensorflow-gpu_python3 .
docker push robertdcurrier/ubuntu_tensorflow-gpu_python3

