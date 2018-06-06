#!/bin/bash

# Clone latest repo
echo "Cloning gncutils..."
if [ ! -d src/gncutils ]; then
  git clone https://github.com/kerfoot/gncutils.git src/gncutils
else
  cd src/gncutils
  git pull
fi

# Back to top level
cd ~/src/docker/docker_images/gncutils

# Build image
docker build -t robertdcurrier/gncutils .

