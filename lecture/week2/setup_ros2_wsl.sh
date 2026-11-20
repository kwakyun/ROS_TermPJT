#!/bin/bash
set -e

echo "==> 1. Adding ROS 2 apt repository key..."
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "==> 2. Adding ROS 2 apt repository..."
. /etc/os-release
ARCH=$(dpkg --print-architecture)
echo "deb [arch=${ARCH} signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${UBUNTU_CODENAME} main" > /etc/apt/sources.list.d/ros2.list

echo "==> 3. Updating package index..."
apt-get update

echo "==> 4. Installing ROS 2 Humble Desktop and dev tools..."
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ros-humble-desktop \
  ros-dev-tools \
  python3-colcon-common-extensions

echo "==> 5. Adding ROS 2 source to user bashrc if not present..."
if ! grep -q "source /opt/ros/humble/setup.bash" /home/user/.bashrc; then
  echo "source /opt/ros/humble/setup.bash" >> /home/user/.bashrc
fi

echo "==> ROS 2 Humble installation complete!"
