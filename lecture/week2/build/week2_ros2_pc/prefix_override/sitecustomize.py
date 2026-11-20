import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/mnt/d/WorkSpace/ROS/week2/install/week2_ros2_pc'
