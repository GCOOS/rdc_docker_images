# TENSORFLOW with GPU support
From tensorflow/tensorflow:2.0.0-gpu-py3
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2021-12-14
WORKDIR /habcam
COPY requirements.txt /habcam
COPY focus-stack_1.1-19-g0ee97d3_amd64.deb /habcam
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
        libcanberra-gtk-module \
        libatlas-base-dev \
        gfortran \
        ffmpeg \
        libv4l-dev \
        vim \
        ssh \
        rsync \
	ttf-mscorefonts-installer \
        libopencv-core3.2 \
        libopencv-video3.2 \
        libopencv-imgcodecs3.2 \
        libopencv-photo3.2 \
        libopencv-imgproc3.2 \
&& pip3 install --upgrade pip
RUN dpkg -i focus-stack_1.1-19-g0ee97d3_amd64.deb

COPY requirements.txt /phytotracker3
RUN pip3 install -r requirements.txt
RUN fc-cache -f -v

# OpenCV added 2020-11-03
WORKDIR /tmp
RUN git clone https://github.com/Itseez/opencv.git
RUN git clone https://github.com/Itseez/opencv_contrib

WORKDIR /tmp/opencv
RUN git checkout 4.4.0
WORKDIR /tmp/opencv_contrib
RUN git checkout 4.4.0
RUN mkdir /tmp/opencv/build

WORKDIR /tmp/opencv/build/

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
        -D BUILD_EXAMPLES=OFF \
        -D WITH_CUDA=OFF \
        -D USE_GTK=ON \
        -D USE_V4L=ON .. \
&& make -j4 \
&& make install \
&& rm -rf /tmp/opencv \
&& rm -rf /tmp/opencv_contrib
WORKDIR /habcam
