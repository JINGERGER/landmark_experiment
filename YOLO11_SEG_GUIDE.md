# YOLO11 Instance Segmentation ROS2 Node Guide

## 📋 Overview
This ROS2 node provides real-time instance segmentation using YOLO11n-seg.pt model on camera images from `/camera/camera/color/image_raw`.

## 🚀 Features

### 🎯 **Core Capabilities**:
- **Instance Segmentation**: Pixel-level object segmentation
- **Real-time Processing**: Optimized for live camera feeds
- **Multi-output**: Annotated images, mask visualizations, and JSON data
- **Configurable Parameters**: Adjustable confidence, IoU thresholds, and visualization options

### 📤 **Published Topics**:
| Topic | Type | Description |
|-------|------|-------------|
| `/yolo11/segmentation/image` | `sensor_msgs/Image` | Annotated image with masks and bounding boxes |
| `/yolo11/segmentation/masks` | `sensor_msgs/Image` | Segmentation masks visualization |
| `/yolo11/segmentation/data` | `std_msgs/String` | JSON data with detection and segmentation info |

### 📥 **Subscribed Topics**:
| Topic | Type | Description |
|-------|------|-------------|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | Input camera image |

## 🛠️ Installation & Setup

### 1. **Install Dependencies**:
```bash
pip install ultralytics
```

### 2. **Build Workspace**:
```bash
cd /home/vln/ros2_ws
colcon build --packages-select bev
source install/setup.bash
```

## 🎮 Quick Start

### Option 1: Use automated script
```bash
cd /home/vln/ros2_ws/src/bev
./run_yolo11_seg.sh
```

### Option 2: Manual startup
```bash
# Terminal 1: Start segmentation node
python3 /home/vln/ros2_ws/src/bev/src/d435i_seg.py

# Terminal 2: View results
ros2 run rqt_image_view rqt_image_view /yolo11/segmentation/image
```

## ⚙️ Configuration Parameters

### **Available Parameters**:
```bash
ros2 run python3 /home/vln/ros2_ws/src/bev/src/d435i_seg.py --ros-args \
  -p model_name:=yolo11n-seg.pt \
  -p confidence_threshold:=0.5 \
  -p iou_threshold:=0.5 \
  -p mask_alpha:=0.4 \
  -p save_results:=true \
  -p show_masks:=true
```

### **Parameter Descriptions**:
- `model_name`: YOLO11 segmentation model (yolo11n-seg.pt, yolo11s-seg.pt, etc.)
- `confidence_threshold`: Minimum confidence for detections (0.0-1.0)
- `iou_threshold`: IoU threshold for Non-Maximum Suppression (0.0-1.0)
- `mask_alpha`: Transparency for mask overlay (0.0-1.0)
- `save_results`: Save results to files (true/false)
- `show_masks`: Show segmentation masks on image (true/false)
- `show_fps`: Display FPS counter (true/false)
- `save_path`: Directory to save results (default: /tmp/yolo11_seg_results)

## 🎨 Visualization Options

### **1. RViz2 Visualization**:
```bash
rviz2
```
Add these displays:
- **Image Display** → `/camera/camera/color/image_raw`
- **Image Display** → `/yolo11/segmentation/image`
- **Image Display** → `/yolo11/segmentation/masks`

### **2. rqt_image_view**:
```bash
# View annotated results
ros2 run rqt_image_view rqt_image_view /yolo11/segmentation/image

# View masks only
ros2 run rqt_image_view rqt_image_view /yolo11/segmentation/masks

# View original camera
ros2 run rqt_image_view rqt_image_view /camera/camera/color/image_raw
```

### **3. Terminal Data Monitoring**:
```bash
# View segmentation data
ros2 topic echo /yolo11/segmentation/data

# Monitor performance
ros2 topic hz /yolo11/segmentation/image
ros2 topic bw /yolo11/segmentation/image
```

## 📊 Output Data Format

### **JSON Segmentation Data**:
```json
{
  "timestamp": "2025-09-11T10:30:00.123456",
  "image_size": {"width": 640, "height": 480},
  "model": "yolo11n-seg.pt",
  "confidence_threshold": 0.5,
  "iou_threshold": 0.5,
  "detections": [
    {
      "id": 0,
      "class": "person",
      "confidence": 0.87,
      "bbox": {
        "x1": 100, "y1": 50,
        "x2": 300, "y2": 400,
        "width": 200, "height": 350,
        "center_x": 200, "center_y": 225
      },
      "segmentation": {
        "area": 45678,
        "polygon": [[100,50], [300,50], [300,400], [100,400]],
        "mask_shape": [480, 640]
      }
    }
  ]
}
```

## 🔧 Performance Optimization

### **Model Selection**:
- `yolo11n-seg.pt`: Fastest, lowest accuracy (~30 FPS)
- `yolo11s-seg.pt`: Balanced (~20 FPS)
- `yolo11m-seg.pt`: Better accuracy (~15 FPS)
- `yolo11l-seg.pt`: High accuracy (~10 FPS)
- `yolo11x-seg.pt`: Best accuracy (~5 FPS)

### **Parameter Tuning**:
```bash
# For speed (reduce quality)
-p confidence_threshold:=0.7 -p iou_threshold:=0.6

# For accuracy (reduce speed)
-p confidence_threshold:=0.3 -p iou_threshold:=0.4

# Reduce visualization overhead
-p show_masks:=false -p save_results:=false
```

## 🚨 Troubleshooting

### **Common Issues**:

1. **YOLO Import Error**:
```bash
pip install ultralytics
```

2. **Model Download Issues**:
- Check internet connection
- The model will auto-download on first use
- Download location: `~/.cache/ultralytics/`

3. **Camera Topic Not Found**:
```bash
# Check available topics
ros2 topic list | grep camera

# Common alternatives
-p image_topic:=/camera/color/image_raw
-p image_topic:=/camera/rgb/image_raw
```

4. **Low FPS Performance**:
- Use smaller model (yolo11n-seg.pt)
- Increase confidence threshold
- Disable mask visualization
- Reduce image resolution

5. **RViz2 Not Showing Images**:
- Check topic names match
- Verify image encoding (should be bgr8)
- Set correct Fixed Frame

## 📈 Expected Performance

### **YOLO11n-seg Performance**:
- **Speed**: 20-30 FPS on typical hardware
- **Accuracy**: Good for real-time applications
- **Memory**: ~2GB GPU memory
- **Classes**: 80 COCO classes (person, car, chair, etc.)

### **Supported Object Classes**:
The model can segment 80 different object classes including:
- People, animals
- Vehicles (car, truck, bus, motorcycle, etc.)
- Furniture (chair, table, sofa, etc.)
- Electronics (laptop, phone, TV, etc.)
- Sports equipment, kitchen items, and more

## 🔗 Integration Examples

### **Combine with BEV Projection**:
```python
# Use segmentation results in BEV system
segmentation_data = json.loads(msg.data)
for detection in segmentation_data['detections']:
    class_name = detection['class']
    polygon = detection['segmentation']['polygon']
    # Project polygon to BEV coordinates
```

### **Object Tracking**:
```python
# Use detection IDs for tracking
detection_id = detection['id']
bbox = detection['bbox']
# Implement tracking logic
```

### **Dynamic Parameter Updates**:
```bash
# Adjust parameters during runtime
ros2 param set /yolo11_segmentation_node confidence_threshold 0.6
ros2 param set /yolo11_segmentation_node mask_alpha 0.3
```

## 💡 Advanced Usage

### **Custom Model Usage**:
```bash
# Use custom trained model
-p model_name:=/path/to/your/custom_model.pt
```

### **Selective Class Detection**:
Modify the code to filter specific classes:
```python
# In process_segmentation function
if class_name in ['person', 'car', 'truck']:  # Only detect specific classes
    # Process detection
```

### **Save Results with Timestamps**:
Results are automatically saved to `/tmp/yolo11_seg_results/` with format:
- `yolo11_seg_YYYYMMDD_HHMMSS_NNN_XXXXXX_annotated.jpg`
- `yolo11_seg_YYYYMMDD_HHMMSS_NNN_XXXXXX_masks.jpg`
- `yolo11_seg_YYYYMMDD_HHMMSS_NNN_XXXXXX_data.json`
