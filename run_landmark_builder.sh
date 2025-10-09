#!/bin/bash

# Landmark Graph Builder启动脚本
# 启动地标图构建节点

echo "🚀 Starting Landmark Graph Builder..."

# 确保环境已设置
source /opt/ros/humble/setup.bash
source /home/vln/ros2_ws/install/setup.bash

# 启动地标图构建器
ros2 run bev landmarker_graph_builder.py

echo "🛑 Landmark Graph Builder stopped."
