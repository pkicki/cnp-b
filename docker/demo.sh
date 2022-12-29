#!/usr/bin/bash
. /catkin_ws/devel/setup.bash
roslaunch kinodynamic_gazebo kinodynamic_gazebo.launch &
sleep 3
roslaunch kinodynamic_neural_planner planner.launch &
sleep 3
gzclient &
sleep 3
rosrun kinodynamic_neural_planner planners_evaluation.py
