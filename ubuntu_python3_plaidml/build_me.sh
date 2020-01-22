#!/bin/bash
# Build image
docker build -t robertdcurrier/ubuntu_python3_plaidml .
docker push robertdcurrier/ubuntu_python3_plaidml
