#!/bin/bash

# YOLO Detection with RViz2 Visualization Setup Script
# Usage: ./run_yolo_with_rviz.sh

echo "🚀 Starting YOLO Detection with RViz2 Visualization"
echo "=================================================="

# Check if ROS2 environment is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "❌ ROS2 environment not found. Please source your ROS2 setup.bash"
    exit 1
fi

echo "✅ ROS2 Distribution: $ROS_DISTRO"

# Source the workspace
if [ -f "/home/vln/ros2_ws/install/setup.bash" ]; then
    source /home/vln/ros2_ws/install/setup.bash
    echo "✅ Workspace sourced"
else
    echo "⚠️  Workspace not built. Building now..."
    cd /home/vln/ros2_ws
    colcon build --packages-select bev
    source install/setup.bash
fi

# Function to kill background processes on exit
cleanup() {
    echo "🛑 Shutting down..."
    kill $YOLO_PID $RVIZ_PID 2>/dev/null
    wait
    echo "✅ All processes stopped"
}

trap cleanup EXIT

echo ""
echo "📋 Available Commands:"
echo "  1. Start YOLO Detection Node"
echo "  2. Start RViz2 with YOLO configuration"
echo "  3. Start both (recommended)"
echo "  4. View live detection results"
echo "  5. Exit"
echo ""

read -p "Choose option (1-5): " choice

case $choice in
    1)
        echo "🎯 Starting YOLO Detection Node..."
        python3 /home/vln/ros2_ws/src/bev/src/d435i_semantic.py
        ;;
    2)
        echo "🖥️  Starting RViz2..."
        rviz2 -d /home/vln/ros2_ws/src/bev/rviz2_yolo_config.rviz
        ;;
    3)
        echo "🎯 Starting YOLO Detection Node in background..."
        python3 /home/vln/ros2_ws/src/bev/src/d435i_semantic.py &
        YOLO_PID=$!
        
        echo "⏳ Waiting for YOLO node to initialize..."
        sleep 3
        
        echo "🖥️  Starting RViz2..."
        rviz2 -d /home/vln/ros2_ws/src/bev/rviz2_yolo_config.rviz &
        RVIZ_PID=$!
        
        echo ""
        echo "✅ Both nodes started!"
        echo "📋 RViz2 Display Configuration:"
        echo "   - Original Camera Image: /camera/camera/color/image_raw"
        echo "   - YOLO Detections: /yolo/detections/image"
        echo "   - Detection Markers: /yolo/detections/markers"
        echo ""
        echo "💡 Tips:"
        echo "   - Switch between different views in RViz2"
        echo "   - Toggle visibility of different displays"
        echo "   - Use the Camera view to see overlaid markers"
        echo ""
        echo "Press Ctrl+C to stop all nodes..."
        wait
        ;;
    4)
        echo "📺 Available visualization topics:"
        echo ""
        echo "🖼️  Image Topics:"
        echo "   ros2 run rqt_image_view rqt_image_view /camera/camera/color/image_raw"
        echo "   ros2 run rqt_image_view rqt_image_view /yolo/detections/image"
        echo ""
        echo "📊 Detection Data:"
        echo "   ros2 topic echo /yolo/detections/bbox"
        echo ""
        echo "🎯 RViz Markers:"
        echo "   ros2 topic echo /yolo/detections/markers"
        echo ""
        echo "📈 Topic Information:"
        echo "   ros2 topic list | grep yolo"
        echo "   ros2 topic info /yolo/detections/image"
        echo ""
        read -p "Press Enter to continue..."
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac
