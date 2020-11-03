# TENSORFLOW with GPU support
From tensorflow/tensorflow:2.0.0-gpu-py3
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2020-11-03

WORKDIR /phytotracker3
RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y tzdata \
        apt-utils \
        python3 \
        python3-dev \
        python3-pip \
        build-essential \
        cmake \
        git \
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

COPY requirements.txt /phytotracker3
RUN pip3 install -r requirements.txt

# OpenCV added 2020-11-03
WORKDIR /tmp
RUN git clone https://github.com/Itseez/opencv.git 
RUN git clone https://github.com/Itseez/opencv_contrib

RUN mkdir /tmp/opencv/build
WORKDIR /tmp/opencv/build/

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=OFF \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
	-D BUILD_EXAMPLES=OFF \
        -D WITH_CUDA=OFF \
        -D USE_V4L=ON .. \
&& make -j4 \
&& make install \
&& rm -rf /tmp/opencv \
&& rm -rf /tmp/opencv_contrib
