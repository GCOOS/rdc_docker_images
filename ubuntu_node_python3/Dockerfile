# Ubuntu 18.04 w/apache, nodejs and Python3
FROM ubuntu:18.04
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2018-06-01

ADD . /tmp
WORKDIR /tmp

# System tools
RUN apt-get update \
&& apt-get install -y \
        mlocate \
        curl \
        apache2 \
        libapache2-mod-wsgi-py3 \
        python3 \
        python3-dev \
        python3-pip \
        build-essential \
        pkg-config \
        vim \
&& pip3 install --upgrade pip

# Install all the necessary components via pip
RUN pip3 install -r requirements.txt

# Node stuff
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -  \
&& apt-get install -y nodejs \
&& npm install -g pm2
