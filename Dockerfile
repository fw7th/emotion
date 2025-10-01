FROM ubuntu:20.04

# Set environment variables to avoid some tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install deps
RUN apt-get update && apt-get install -y \
    build-essential \
    time \
    htop \
    procps \
    sysstat \
    cmake \
    pkg-config \
    libopencv-dev \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /workspace

# Copy project source into container
COPY . /workspace
