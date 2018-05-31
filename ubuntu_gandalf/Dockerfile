# GCOOS HABscope applicaton Docker File
FROM robertdcurrier/ubuntu_apache
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2018-05-31

WORKDIR /tmp

# system stuff
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    pkg-config \
    nodejs \
    nodejs-legacy \
    npm \
&&  npm install -g pm2 \
&& pip install --upgrade pip

# Install dependencies
RUN pip install numpy pandas==0.17.1 matplotlib configparser geojson seawater netCDF4
