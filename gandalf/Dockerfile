# Ubuntu latest w/apache and mod_wsgi; Python3 only
FROM ubuntu:18.04
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2022-06-30
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ADD . /tmp
WORKDIR /tmp

# System tools -- added wget and ssh for gandalf_harvest
#RUN apt-get update \
#&& apt-get upgrade -y \
#&&
RUN  apt-get update && apt-get install -y \
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
