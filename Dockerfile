FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
MAINTAINER Or Shabtay <or@dataloop.ai>

RUN apt-get update
RUN apt-get install -y \
    libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
    libjasper-dev libavformat-dev libpq-dev libxine2-dev libglew-dev \
    libtiff5-dev zlib1g-dev libjpeg-dev libpng12-dev libjasper-dev \
    libavcodec-dev libavformat-dev libavutil-dev libpostproc-dev \
    libswscale-dev libeigen3-dev libtbb-dev libgtk2.0-dev locales
    # libcudnn7=7.1.4.18-1+cuda9.0

RUN apt-get update -y  && apt-get install -y bzip2 wget
# install 3.6
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh

# Set path to conda
ENV PATH /root/anaconda3/bin:$PATH
# export PATH="/root/anaconda3:$PATH"

# Updating Anaconda packages
RUN conda update conda
# RUN conda update anaconda
# RUN conda update --all

RUN  apt-get update -y  && \
        apt-get install -y build-essential python3-pip && \
        apt-get install -y git  && \
        # update pip
        python3.6 -m pip install pip --upgrade && \
        python3.6 -m pip install wheel \
        apt-get -y install rsyslog

RUN mkdir -p /src
ENV PYTHONPATH="$PYTHONPATH:/src"
# fix for other languages issues
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

RUN pip --no-cache-dir install \
    nms==0.1.6 \
    imgaug==0.2.9 \
    ffmpeg-python \
    tornado==6.0.2 \
    hdf5storage \
    h5py \
    py3nvml \
    tensorflow-gpu==1.9.0 \
    keras==2.1.6 \
    opencv_python==3.4.2.17 \
    Pillow \
    jwt==0.6.1 \
    psutil==5.6.7 \
    torch==1.1.0 \
    numpy==1.18 \
    scipy==1.1.0 \
    scikit-image==0.14.2 \
    scikit-learn==0.21.3

WORKDIR /home