#!/bin/bash
# Build image
docker build -t robertdcurrier/ubuntu_apache_python3_m2 .
docker push robertdcurrier/ubuntu_apache_python3_m2

