FROM ubuntu:focal
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# install tools
RUN apt-get update
RUN apt-get install -y python3-pip curl lsb-release wget git
# install ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update
RUN apt-get install -y ros-noetic-desktop-full
RUN apt-get install -y coinor-libclp-dev ros-noetic-joint-trajectory-controller ros-noetic-rqt-joint-trajectory-controller ros-noetic-pinocchio
# install TensorFlow
RUN pip3 install tensorflow tensorflow-graphics

RUN mkdir -p /catkin_ws/src
# iiwas core & dependencies
RUN cd /catkin_ws/src/ && git clone https://github.com/PuzeLiu/iiwas_core.git && cd iiwas_core && git checkout manifold_planning
RUN git clone --recursive https://github.com/osqp/osqp.git && cd osqp && mkdir build && cd build && cmake -G "Unix Makefiles" .. && cmake --build . && cmake --build . --target install
RUN git clone https://github.com/robotology/osqp-eigen.git && cd osqp-eigen && mkdir build && cd build && cmake .. && make && make install
RUN git clone https://github.com/stevengj/nlopt.git && cd nlopt && mkdir build && cd build && cmake .. && make && make install
RUN git clone --recursive https://github.com/mherb/kalman.git && cd kalman && mkdir build && cd build && cmake .. && make && make install
RUN wget https://github.com/msteinbeck/tinyspline/releases/download/v0.5.0/tinyspline-0.5.0-Linux.deb && dpkg -i tinyspline-0.5.0-Linux.deb
# kinodynamic_stack
RUN cd /catkin_ws/src/ && git clone https://github.com/pkicki/kinodynamic_stack.git && cd kinodynamic_stack/kinodynamic_neural_planner/scripts && git submodule update --init --recursive && cd manifold_planning && ./download_models.sh
# airhockey
RUN cd /catkin_ws/src/ && git clone https://github.com/PuzeLiu/air_hockey_stack.git && cd air_hockey_stack && git checkout neural_planning && cd air_hockey_neural_planner/scripts && git submodule update --init --recursive
# CNP-B
RUN git clone https://github.com/pkicki/cnp-b.git && cd cnp-b && git submodule update --init --recursive
RUN source /opt/ros/noetic/setup.bash && cd catkin_ws && catkin_make -DCMAKE_BUILD_TYPE=Release
RUN ln -s /usr/bin/python3 /usr/bin/python
