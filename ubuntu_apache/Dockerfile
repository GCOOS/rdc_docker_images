# GCOOS DMAC Application Docker File
FROM ubuntu:16.04
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2018-05-23

ADD . /gcoos
WORKDIR /gcoos

# System tools
RUN apt-get update \
&& apt-get install -y \
        vim \
        mlocate \
        curl \
        apache2 \
        libapache2-mod-wsgi \
        python \
        python-dev \
        python-pip \
%% pip install --upgrade pip

# Install all the necessary components via pip
RUN pip install -r requirements.txt

