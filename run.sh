#!/bin/bash

colcon build
source install/setup.bash
ros2 launch autorace_2023 autorace_2023.launch.py
rm -rf build install log
