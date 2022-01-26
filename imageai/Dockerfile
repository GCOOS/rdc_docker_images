# TENSORFLOW with GPU support
From tensorflow/tensorflow:devel-gpu
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2022-01-24

WORKDIR /imageai
RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y tzdata \
        apt-utils \
        python3 \
        python3-dev \
        python3-pip \
        build-essential \
        cmake \
        pkg-config \
        libjpeg8-dev \
        libtiff5-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libgtk2.0 \
        libgtk2.0-dev \
        libatlas-base-dev \
        gfortran \
        ffmpeg \
        libv4l-dev \
        vim \
        ssh \
        rsync \
&& pip3 install --upgrade pip

COPY requirements.txt /imageai
RUN pip3 install -r requirements.txt
