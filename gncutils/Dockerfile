# GCOOS DMAC Kerfoot gncutils Docker File
FROM continuumio/anaconda3
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2018-06-06

RUN apt-get install -y mlocate \
&& apt-get install -y cron \
&& apt-get install -y gcc \
&& apt-get clean

# Add Kerfoot's tools
COPY src/gncutils /opt/gncutils

WORKDIR /opt/gncutils
RUN python setup.py build \
&& python setup.py install

