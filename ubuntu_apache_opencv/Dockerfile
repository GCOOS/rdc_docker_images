# GCOOS HABscope applicaton Docker File
FROM robertdcurrier/ubuntu_apache
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2018-05-22

WORKDIR /tmp

# system stuff
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python-pip \
    git \
    build-essential \
    cmake \
    pkg-config \
    libjpeg8-dev \
    libtiff5-dev \
    libjasper-dev \
    libpng12-dev \
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
    nodejs \
    nodejs-legacy \
    npm \
&&  npm install -g pm2

# Always upgrade pip
RUN pip2 install --upgrade pip
RUN pip3 install --upgrade pip
# Install dependencies
RUN pip2 install pillow ffmpy numpy pandas==0.17.1  configparser tensorflow keras watchdog dateparser matplotlib
RUN pip3 install geopy geojson wget pandas dateparser pymongo
# OpenCV
WORKDIR /tmp
RUN git clone https://github.com/Itseez/opencv.git 
RUN git clone https://github.com/Itseez/opencv_contrib

WORKDIR /tmp/opencv 
RUN git checkout 3.1.0 
WORKDIR /tmp/opencv_contrib 
RUN git checkout 3.1.0 
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
