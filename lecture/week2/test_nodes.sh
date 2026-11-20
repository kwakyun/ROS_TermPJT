#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /mnt/d/WorkSpace/ROS/week2/install/setup.bash

echo "=== 1. Starting hello_talker ==="
ros2 run week2_ros2_pc hello_talker &
TALKER_PID=$!

echo "=== 2. Starting hello_listener ==="
ros2 run week2_ros2_pc hello_listener &
LISTENER_PID=$!

sleep 3

echo "=== 3. ros2 node list ==="
ros2 node list

echo "=== 4. ros2 topic list ==="
ros2 topic list

echo "=== 5. ros2 topic info /hello_topic ==="
ros2 topic info /hello_topic

echo "=== 6. ros2 topic echo /hello_topic --once ==="
ros2 topic echo /hello_topic --once

echo "=== Terminating nodes ==="
kill $TALKER_PID $LISTENER_PID 2>/dev/null || true
wait $TALKER_PID 2>/dev/null || true
wait $LISTENER_PID 2>/dev/null || true

echo "=== Verification test passed successfully! ==="
