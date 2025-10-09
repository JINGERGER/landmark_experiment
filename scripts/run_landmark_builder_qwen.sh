#!/bin/bash
# 启动带Qwen VLM支持的Landmark Graph Builder

# 1. 设置阿里云百炼API Key（如已在~/.bashrc设置可省略）
export DASHSCOPE_API_KEY="<你的API_KEY>"

# 2. 检查依赖
if ! python3 -c "import openai" 2>/dev/null; then
  echo "[ERROR] openai (dashscope-openai) Python包未安装，请先运行: pip install dashscope-openai"
  exit 1
fi

# 3. ROS2环境
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# 4. 启动Landmark Graph Builder，指定Qwen模型
ros2 run bev landmarker_graph_builder \
  --ros-args \
  -p enable_vlm_association:=true \
  -p vlm_model_name:=qwen3-vl-plus \
  -p vlm_confidence_threshold:=0.7

# 5. 日志输出
# 可选: tail -f ~/.ros/log/latest_build/log/latest_build.log
