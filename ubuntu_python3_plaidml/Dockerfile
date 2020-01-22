# Ubuntu 19.04 Disco so we get Python3.7 with Unicode
FROM ubuntu:disco
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2020-01-20

ADD . /tmp
WORKDIR /tmp

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y \
        python3 \
        python3-dev \
        python3-pip \
        libsm6 libxext6 libxrender-dev 

RUN pip3 install -r requirements.txt
COPY .plaidml /root/.plaidml
