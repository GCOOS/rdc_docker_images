#!/bin/bash
# Build image
docker build -t robertdcurrier/ubuntu_apache_opencv .
docker push robertdcurrier/ubuntu_apache_opencv

