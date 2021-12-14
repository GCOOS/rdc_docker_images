#!/bin/bash

# Build image
docker build -t robertdcurrier/ubuntu_gncutils_2021 .
docker push robertdcurrier/ubuntu_gncutils_2021

