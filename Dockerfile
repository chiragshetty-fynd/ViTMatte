FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES=all
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get upgrade && apt-get update -y
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda create -n envd python=3.8.8

ENV ENVD_PREFIX=/root/miniconda3/envs/envd/bin

RUN update-alternatives --install /usr/bin/python python ${ENVD_PREFIX}/python 1 && \
    update-alternatives --install /usr/bin/python3 python3 ${ENVD_PREFIX}/python3 1 && \
    update-alternatives --install /usr/bin/pip pip ${ENVD_PREFIX}/pip 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 ${ENVD_PREFIX}/pip3 1

RUN apt-get upgrade && apt-get update -y 
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y python3-opencv git 

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENV TORCH_CUDA_ARCH_LIST=8.0
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

WORKDIR /workspace
COPY . .

CMD ["python", "main.py"]