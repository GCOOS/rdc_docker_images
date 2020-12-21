# GCOOS DMAC Application Docker File
FROM continuumio/anaconda3
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2020-12-21

# Set the working directory to the name of the app
WORKDIR /pyifcb
COPY requirements.txt /pyficb
COPY pyifcb /pyifcb
RUN python setup.py install
