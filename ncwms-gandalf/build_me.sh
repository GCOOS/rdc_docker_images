#!/bin/bash
# Build image
docker build -t robertdcurrier/ncwms2-gandalf .
docker push robertdcurrier/ncwms2-gandalf
