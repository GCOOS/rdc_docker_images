# GCOOS DMAC Kerfoot gncutils Docker File
FROM ubuntu:18.04
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2019-09-27

ADD . /tmp
WORKDIR /tmp

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y gcc \
&& apt-get install -y vim \
&& apt-get install -y gnupg2 \
&& apt-get install -y python \
&& apt-get install -y python-setuptools \
&& apt-get install -y python-pip \
&& apt-get clean

RUN pip install -r requirements.txt 

# Add Kerfoot's tools -- we do this here as we don't want
# to go through the whole build process each time we restart
# But it does mean we need to rebuild the image and push if the src changes
COPY src/gncutils /opt/gncutils

WORKDIR /opt/gncutils
RUN python setup.py build \
&& python setup.py install
