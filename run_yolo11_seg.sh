#!/bin/bash

# YOLO11 Segmentation with RViz2 Visualization Setup Script
# Usage: ./run_yolo11_seg.sh

echo "🚀 Starting YOLO11 Instance Segmentation"
echo "========================================"

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
echo "  1. Start YOLO11 Segmentation Node"
echo "  2. Start RViz2 for visualization"
echo "  3. Start both (recommended)"
echo "  4. View live segmentation results"
echo "  5. Test YOLO11 installation"
echo "  6. Exit"
echo ""

read -p "Choose option (1-6): " choice

case $choice in
    1)
        echo "🎯 Starting YOLO11 Segmentation Node..."
        python3 /home/vln/ros2_ws/src/bev/src/d435i_seg.py
        ;;
    2)
        echo "🖥️  Starting RViz2..."
        echo "💡 Add these topics in RViz2:"
        echo "   - /camera/camera/color/image_raw (sensor_msgs/Image)"
        echo "   - /yolo11/segmentation/image (sensor_msgs/Image)"
        echo "   - /yolo11/segmentation/masks (sensor_msgs/Image)"
        rviz2
        ;;
    3)
        echo "🎯 Starting YOLO11 Segmentation Node in background..."
        python3 /home/vln/ros2_ws/src/bev/src/d435i_seg.py &
        YOLO_PID=$!
        
        echo "⏳ Waiting for YOLO11 node to initialize..."
        sleep 5
        
        echo "🖥️  Starting RViz2..."
        rviz2 &
        RVIZ_PID=$!
        
        echo ""
        echo "✅ Both nodes started!"
        echo "📋 Published Topics:"
        echo "   - /yolo11/segmentation/image (annotated image with masks)"
        echo "   - /yolo11/segmentation/masks (segmentation masks only)"
        echo "   - /yolo11/segmentation/data (JSON segmentation data)"
        echo ""
        echo "💡 RViz2 Setup Tips:"
        echo "   1. Add Image displays for the topics above"
        echo "   2. Set Fixed Frame to 'camera_color_optical_frame'"
        echo "   3. Toggle between different image views"
        echo ""
        echo "Press Ctrl+C to stop all nodes..."
        wait
        ;;
    4)
        echo "📺 Available visualization commands:"
        echo ""
        echo "🖼️  Image Topics:"
        echo "   ros2 run rqt_image_view rqt_image_view /camera/camera/color/image_raw"
        echo "   ros2 run rqt_image_view rqt_image_view /yolo11/segmentation/image"
        echo "   ros2 run rqt_image_view rqt_image_view /yolo11/segmentation/masks"
        echo ""
        echo "📊 Segmentation Data:"
        echo "   ros2 topic echo /yolo11/segmentation/data"
        echo ""
        echo "📈 Topic Information:"
        echo "   ros2 topic list | grep yolo11"
        echo "   ros2 topic info /yolo11/segmentation/image"
        echo "   ros2 topic hz /yolo11/segmentation/image"
        echo ""
        echo "🎛️  Parameter Control:"
        echo "   ros2 param list /yolo11_segmentation_node"
        echo "   ros2 param set /yolo11_segmentation_node confidence_threshold 0.6"
        echo "   ros2 param set /yolo11_segmentation_node mask_alpha 0.3"
        echo ""
        read -p "Press Enter to continue..."
        ;;
    5)
        echo "🔧 Testing YOLO11 installation..."
        python3 -c "
import sys
try:
    from ultralytics import YOLO
    print('✅ Ultralytics installed successfully')
    
    # Test YOLO11 model loading
    print('🔄 Testing YOLO11n-seg model download/loading...')
    model = YOLO('yolo11n-seg.pt')
    print(f'✅ YOLO11n-seg model loaded successfully')
    print(f'📋 Model can detect {len(model.names)} classes')
    print(f'🏷️  Classes: {list(model.names.values())[:10]}...')  # Show first 10 classes
    
except ImportError as e:
    print(f'❌ Import failed: {e}')
    print('💡 Install with: pip install ultralytics')
    sys.exit(1)
except Exception as e:
    print(f'❌ Model loading failed: {e}')
    print('💡 Check internet connection for model download')
    sys.exit(1)
"
        if [ $? -eq 0 ]; then
            echo ""
            echo "🎉 YOLO11 is ready to use!"
        else
            echo ""
            echo "❌ YOLO11 setup incomplete. Please fix the issues above."
        fi
        read -p "Press Enter to continue..."
        ;;
    6)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac
