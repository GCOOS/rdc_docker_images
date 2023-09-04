# Tomcat 9 for ncWMS2 -- DO appears to no longer support older TC versions
FROM tomcat:9.0.5-jre8
MAINTAINER Bob Currier <robertdcurrier@gmail.com>
ENV REFRESHED_AT 2023-09-03
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ADD ./ncWMS2.war /usr/local/tomcat/webapps
