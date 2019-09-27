#!/bin/bash

# Build image
docker build -t robertdcurrier/ubuntu_gncutils .
docker push robertdcurrier/ubuntu_gncutils

