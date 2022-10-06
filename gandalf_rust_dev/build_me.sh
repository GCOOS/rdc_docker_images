#!/bin/bash
# Build image
docker build -t robertdcurrier/gandalf_rust_dev .
docker push robertdcurrier/gandalf_rust_dev

