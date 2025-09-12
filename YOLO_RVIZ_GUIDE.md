# YOLO Detection RViz2 Visualization Guide

## 📋 Overview
This setup allows you to visualize YOLO object detection results in RViz2 with multiple display options.

## 🚀 Quick Start

### Option 1: Use the automated script
```bash
cd /home/vln/ros2_ws/src/bev
./run_yolo_with_rviz.sh
```

### Option 2: Manual startup

1. **Build the workspace** (if not already built):
```bash
cd /home/vln/ros2_ws
colcon build --packages-select bev
source install/setup.bash
```

2. **Start the YOLO detection node**:
```bash
ros2 run bev d435i_semantic.py
```

3. **Start RViz2** (in another terminal):
```bash
rviz2 -d /home/vln/ros2_ws/src/bev/rviz2_yolo_config.rviz
```

## 📊 Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/yolo/detections/image` | `sensor_msgs/Image` | Annotated image with bounding boxes |
| `/yolo/detections/bbox` | `std_msgs/String` | JSON detection data |
| `/yolo/detections/markers` | `visualization_msgs/MarkerArray` | RViz markers for 3D visualization |

## 🎯 RViz2 Display Configuration

The provided RViz config includes:

### 1. **Original Camera Image**
- Topic: `/camera/camera/color/image_raw`
- Shows the raw camera feed

### 2. **YOLO Detections**
- Topic: `/yolo/detections/image`
- Shows annotated image with bounding boxes, labels, and confidence scores

### 3. **Detection Markers**
- Topic: `/yolo/detections/markers`
- 3D markers overlaid on the camera view
- Includes text labels and bounding box outlines

### 4. **Camera View**
- Interactive 3D camera perspective
- Shows markers overlaid on the camera image
- Best for understanding spatial relationships

## ⚙️ Configuration Parameters

You can customize the YOLO node with these parameters:

```bash
ros2 run bev d435i_semantic.py --ros-args \
  -p model_name:=yolov8s.pt \
  -p confidence_threshold:=0.6 \
  -p publish_markers:=true \
  -p save_results:=true \
  -p show_fps:=true
```

### Available Parameters:
- `model_name`: YOLO model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
- `confidence_threshold`: Minimum confidence for detections (0.0-1.0)
- `publish_markers`: Enable/disable RViz markers (true/false)
- `save_results`: Save detection results to files (true/false)
- `show_fps`: Display FPS on image (true/false)
- `save_path`: Directory to save results (default: /tmp/yolo_results)

## 🔧 Troubleshooting

### YOLO Import Issues
If you get import errors:
```bash
pip install ultralytics
```

### Camera Topic Not Available
Check available camera topics:
```bash
ros2 topic list | grep camera
```

Common D435i topics:
- `/camera/camera/color/image_raw`
- `/camera/color/image_raw`
- `/camera/rgb/image_raw`

### RViz2 Not Showing Markers
1. Check if markers are being published:
```bash
ros2 topic echo /yolo/detections/markers
```

2. Verify the frame_id in RViz2 matches the camera frame:
   - Fixed Frame should be: `camera_color_optical_frame`

3. Check marker lifetime - they expire after 1 second

### No Detections Showing
1. Verify confidence threshold isn't too high
2. Check if camera is working:
```bash
ros2 run rqt_image_view rqt_image_view /camera/camera/color/image_raw
```

## 💡 Tips for Better Visualization

1. **Adjust View in RViz2**:
   - Use "Camera" view for best marker overlay
   - Switch to "Orbit" view for 3D perspective

2. **Toggle Displays**:
   - Turn off displays you don't need
   - Adjust alpha values for transparency

3. **Performance Optimization**:
   - Use yolov8n.pt for fastest inference
   - Reduce image resolution if needed
   - Increase confidence threshold to reduce false positives

4. **Save and Load Configurations**:
   - Save your custom RViz config: File → Save Config As
   - Load saved configs: File → Open Config

## 📈 Performance Monitoring

Monitor performance with:
```bash
ros2 topic hz /yolo/detections/image
ros2 topic bw /yolo/detections/image
```

Expected performance:
- YOLOv8n: 20-30 FPS on typical hardware
- YOLOv8s: 15-25 FPS
- YOLOv8m: 10-20 FPS

## 🔗 Integration with BEV

This YOLO detection system can be combined with your BEV projection system:
1. Use detection results to identify objects in the scene
2. Project detected objects to BEV coordinate system
3. Create semantic BEV maps with object labels

Example integration workflow:
1. YOLO detects objects in camera image
2. Use depth information to get 3D positions
3. Transform to robot coordinate system
4. Project to BEV view with object labels
