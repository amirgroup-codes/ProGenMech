FROM --platform=linux/x86_64 mosaicml/pytorch:2.5.1_cu124-python3.11-ubuntu22.04

SHELL ["/bin/bash", "-c"]
WORKDIR /root

# Install base utilities
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        wget \
        curl \
        git \
        vim \
        libxml2 \
        apt-transport-https \
        ca-certificates \
        gnupg \
        unzip && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.7 8.9 9.0"
RUN pip install "megablocks[gg]==0.7.0"
RUN MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation
