

FROM ubuntu:latest
WORKDIR /src
COPY . /src/

RUN cd /src \
    && apt-get update && apt-get install -y python3 python3-pip sudo \
    && pip3 install --upgrade pip \
    && pip3 install -r requirements.txt 

EXPOSE 8080
ENTRYPOINT python3 main.py












