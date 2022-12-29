xhost + local:root

 docker run -it \
     --runtime=nvidia \
     --env="CONTAINER_NAME" \
     --env="DISPLAY" \
     --env="NVIDIA_VISIBLE_DEVICES=all" \
     --env="NVIDIA_DRIVER_CAPABILITIES=all" \
     --volume="/tmp/.X11-unix:/tmp/.X11-unix:ro" \
     --volume="$(pwd):/demo" \
     --privileged \
     --network=host \
     pkic/ros_tf

