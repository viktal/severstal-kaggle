FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

# Install dependencies
RUN apt-get update -y && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    libbz2-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install \
    torch \
    torchvision

RUN python3 -m pip install \
    numpy \
    matplotlib \
    seaborn \
    tqdm \
    opencv-python \
    albumentations \
    pandas \
    scikit-learn \
    scikit-multilearn \
    arff

RUN python3 -m pip install requests