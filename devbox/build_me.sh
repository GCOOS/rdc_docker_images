#!/bin/bash
# Build image
docker build -t robertdcurrier/devbox .
docker push robertdcurrier/devbox
