#!/bin/bash

# Define your workspace container
CONTAINER_NAME=lowtier-iot
IMAGE_NAME=lowtier-iot:latest
MOUNT_PATH=$(pwd)

# Allow local Docker containers to access the host X server
xhost +local:docker

# If container exists, start it; otherwise, run a new one
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker start -ai "$CONTAINER_NAME"
else
    docker run -it \
        --name "$CONTAINER_NAME" \
        -v "$MOUNT_PATH":/app \
	-e NO_AT_BRIDGE=1 \
        -e DISPLAY=$DISPLAY \
        -e QT_X11_NO_MITSHM=1 \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /run/user/1000/at-spi:/run/user/1000/at-spi \
        "$IMAGE_NAME"
fi


