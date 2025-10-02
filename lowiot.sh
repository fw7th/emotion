# Simulates a low-tier IoT device

docker run -it --rm \
  --cpus="1.0" \
  -m 512m \
  --device /dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  lowiot:latest
