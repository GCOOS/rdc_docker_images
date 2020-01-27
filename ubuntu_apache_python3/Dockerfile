# Ubuntu 18.04 w/apache and mod_wsgi; Python3 only
FROM ubuntu:18.04
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2020-01-27

ADD . /tmp
WORKDIR /tmp

# System tools -- added wget and ssh for gandalf_harvest
RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y \
        mlocate \
        curl \
        apache2 \
        libapache2-mod-wsgi-py3 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        build-essential \
        pkg-config \
        vim \
        wget \
        ssh \
        rsync \
&& pip3 install --upgrade pip

# Install all the necessary components via pip
RUN pip3 install -r requirements.txt
