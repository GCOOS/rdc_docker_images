# GCOOS DMAC Kerfoot gncutils Docker File
FROM ubuntu:18.04
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2021-07-06

ADD . /tmp
WORKDIR /tmp

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y vim \
&& apt-get install -y python3 \
&& apt-get install -y python3-setuptools \
&& apt-get install -y python3-pip \
&& apt-get install -y libgeos-dev \
&& apt-get clean \
&& cd /usr/local/bin \
&& ln -s /usr/bin/python3 python  \
&& pip3 install --upgrade pip

COPY requirements.txt /tmp
RUN pip3 install -r requirements.txt

# Add Kerfoot's tools -- we do this here as we don't want
# to go through the whole build process each time we restart
# But it does mean we need to rebuild the image and push if the src changes
COPY src/gncutils /opt/gncutils

WORKDIR /opt/gncutils
RUN python3 setup.py build \
&& python3 setup.py install
