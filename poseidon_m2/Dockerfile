# Ubuntu latest w/apache and mod_wsgi; Python3 only
FROM ubuntu:22.04
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2024-03-12
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ADD . /tmp
WORKDIR /tmp

RUN  apt-get update && apt-get install -y \
	gdal-bin \
	libgdal-dev \
        tzdata \
        mlocate \
        curl \
        apache2 \
        libapache2-mod-wsgi-py3 \
        python3 \
        python3-dev \
        python3-pip \
        pkg-config \
        vim \
        wget \
        ssh \
        rsync \
        sqlite3 \
&& pip3 install --upgrade pip

# Install all the necessary components via pip
RUN pip3 install -r requirements.txt
