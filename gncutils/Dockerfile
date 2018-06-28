# GCOOS DMAC Kerfoot gncutils Docker File
FROM continuumio/anaconda3
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2018-06-06

# We no longer use cron as it borked -- we use internal scheduler
RUN apt-get install -y mlocate \
&& apt-get install -y gcc \
&& apt-get clean

# Add Kerfoot's tools -- we do this here as we don't want
# to go through the whole build process each time we restart
# But it does mean we need to rebuild the image and push if the src changes
COPY src/gncutils /opt/gncutils

WORKDIR /opt/gncutils
RUN python setup.py build \
&& python setup.py install
# Add seawater and geojson
RUN conda install -y -c conda-forge seawater geojson
