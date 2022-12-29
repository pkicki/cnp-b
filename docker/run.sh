xhost + local:root

docker run -it \
    --env="CONTAINER_NAME" \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:ro" \
    --volume="$(pwd):/demo" \
    --privileged \
    --network=host \
    pkic/ros_tf
