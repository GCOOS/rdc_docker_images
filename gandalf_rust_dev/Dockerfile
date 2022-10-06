# Start with rust container and add vim
FROM rust
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2022-10-05
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ADD . /tmp
WORKDIR /tmp

RUN  apt-get update && apt-get install -y \
        tzdata \
        mlocate \
        vim
