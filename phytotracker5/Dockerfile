# TENSORFLOW with GPU support
From tensorflow/tensorflow:2.3.2-gpu
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2022-01-26

WORKDIR /phytotracker5
RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y tzdata \
        ffmpeg \
        vim \
&& pip3 install --upgrade pip

COPY requirements.txt /phytotracker5
RUN pip3 install -r requirements.txt
