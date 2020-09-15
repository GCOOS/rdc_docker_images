#!/bin/bash
# Build image
docker build -t robertdcurrier/tensorflow-gpu_python3 .
docker push robertdcurrier/tensorflow-gpu_python3

