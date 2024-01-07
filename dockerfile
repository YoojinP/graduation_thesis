FROM python:3.7
FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
WORKDIR /ussr/src/app

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
apt-get install libglib2.0-0

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["/bin/bash"]