#!/bin/bash
# Build image
docker build -t robertdcurrier/gandalf_rust_dist .
docker push robertdcurrier/gandalf_rust_dist

