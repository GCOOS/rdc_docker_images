#!/bin/bash

# Build image
docker build -t robertdcurrier/ubuntu_gncutils_2023 .
docker push robertdcurrier/ubuntu_gncutils_2023

