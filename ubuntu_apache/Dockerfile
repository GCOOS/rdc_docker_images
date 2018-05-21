# GCOOS DMAC Application Docker File
FROM ubuntu:16.04
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2018-05-17

ADD . /gcoos
WORKDIR /gcoos

# System tools
RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y mlocate

# Install Python, Python-dev, pip
RUN apt-get install -y python
RUN apt-get install -y python-dev
RUN apt-get install -y python-pip
RUN pip install  --upgrade pip
RUN apt-get install -y python-setuptools
RUN apt-get install -y curl

# Install all the necessary components via pip
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install -y apache2
RUN apt-get install -y libapache2-mod-wsgi

