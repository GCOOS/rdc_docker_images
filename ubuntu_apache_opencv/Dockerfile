# GCOOS HABscope applicaton Docker File
FROM robertdcurrier/apache_ubuntu:16.04
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2018-05-22

WORKDIR /tmp
ADD . /tmp
# Update
RUN apt-get -y update

# system stuff
RUN apt-get install -y git
RUN apt-get install -y vim

# Python
RUN apt-get install -y python python-dev

# pip
RUN apt-get install -y python-pip
RUN pip install --upgrade pip

# Now we build the HABscope-specific stuff
RUN apt-get install -y build-essential cmake pkg-config
RUN apt-get install -y libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt-get install -y libgtk2.0 libgtk2.0-dev
RUN apt-get install -y libdc1394-22-dev
RUN apt-get install -y libatlas-base-dev gfortran

#FFMPEG
RUN apt-get install -y ffmpeg

#Video 4 Linux
RUN apt-get install -y libv4l-dev

# Install dependencies
RUN pip install numpy
RUN pip install scipy
RUN pip install pillow
RUN pip install pandas
RUN pip install matplotlib
RUN pip install configparser
RUN pip install tensorflow
RUN pip install keras

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
        -D USE_V4L=ON ..
RUN make -j4
RUN make install

WORKDIR /tmp
RUN rm -rf /tmp/opencv
RUN rm -rf /tmp/opencv_contrib
