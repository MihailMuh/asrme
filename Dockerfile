FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home
COPY . .


RUN apt update && \
    apt install -y libsndfile1 \
    software-properties-common \
    ffmpeg \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    gnupg2 \
    wget \
    git \
    curl \
    gdb \
    openmpi-bin \
    libopenmpi-dev \
    libffi-dev \
    libssl-dev \
    python3-pip \
    libbz2-dev \
    python3-dev \
    liblzma-dev \
    libsqlite3-dev --no-install-recommends && \
    pip3 install --no-cache-dir --upgrade pip wheel setuptools && \
    pip3 install --no-cache-dir $(cat requirements.txt | grep numpy) $(cat requirements.txt | grep Cython) $(cat requirements.txt | grep typing_extensions) && \
    pip3 install --no-cache-dir -r requirements.txt && \
    apt purge -y python3-pip && \
    apt autoremove -y --purge && \
    apt clean && \
    rm -rf /tmp/* && \
    rm -rf /var/tmp/* && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 60002/tcp

ENTRYPOINT ["python3", "main.py"]
