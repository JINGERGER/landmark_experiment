#!/usr/bin/env python3
"""
ROS2 Node for YOLO object detection on camera images
Subscribes to: /camera/camera/color/image_raw
Publishes to: /yolo/detections/image (annotated image)
              /yolo/detections/bbox (detection results)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from datetime import datetime
import os

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    print(f"❌ YOLO import failed: {e}")
    print("Please install ultralytics: pip install ultralytics")
    YOLO_AVAILABLE = False

class YOLODetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')
        
        if not YOLO_AVAILABLE:
            self.get_logger().error("YOLO is not available. Please install ultralytics package.")
            return
        
        # Declare parameters
        self.declare_parameter('model_name', 'yolov8n.pt')  # YOLOv8 nano model (fastest)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('output_image_topic', '/yolo/detections/image')
        self.declare_parameter('output_bbox_topic', '/yolo/detections/bbox')
        self.declare_parameter('output_markers_topic', '/yolo/detections/markers')
        self.declare_parameter('save_results', True)
        self.declare_parameter('save_path', '/tmp/yolo_results')
        self.declare_parameter('show_fps', True)
        self.declare_parameter('publish_markers', True)
        
        # Get parameters
        self.model_name = self.get_parameter('model_name').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.image_topic = self.get_parameter('image_topic').value
        self.output_image_topic = self.get_parameter('output_image_topic').value
        self.output_bbox_topic = self.get_parameter('output_bbox_topic').value
        self.output_markers_topic = self.get_parameter('output_markers_topic').value
        self.save_results = self.get_parameter('save_results').value
        self.save_path = self.get_parameter('save_path').value
        self.show_fps = self.get_parameter('show_fps').value
        self.publish_markers = self.get_parameter('publish_markers').value
        
        # Initialize YOLO model
        self.get_logger().info(f"Loading YOLO model: {self.model_name}")
        try:
            self.yolo_model = YOLO(self.model_name)
            self.get_logger().info("✅ YOLO model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load YOLO model: {e}")
            return
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Create save directory if needed
        if self.save_results:
            os.makedirs(self.save_path, exist_ok=True)
            self.get_logger().info(f"Results will be saved to: {self.save_path}")
            self.image_counter = 0
        
        # FPS calculation
        self.fps_counter = 0
        self.fps_start_time = self.get_clock().now()
        
        # Create subscriber
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        
        # Create publishers
        self.image_pub = self.create_publisher(
            Image,
            self.output_image_topic,
            10
        )
        
        self.bbox_pub = self.create_publisher(
            String,
            self.output_bbox_topic,
            10
        )
        
        self.markers_pub = self.create_publisher(
            MarkerArray,
            self.output_markers_topic,
            10
        )
        
        self.get_logger().info(f"🚀 YOLO Detection Node initialized")
        self.get_logger().info(f"📥 Subscribing to: {self.image_topic}")
        self.get_logger().info(f"📤 Publishing annotated images to: {self.output_image_topic}")
        self.get_logger().info(f"📤 Publishing detections to: {self.output_bbox_topic}")
        if self.publish_markers:
            self.get_logger().info(f"📤 Publishing RViz markers to: {self.output_markers_topic}")

    def image_callback(self, msg):
        """Process incoming camera images with YOLO detection"""
        if not YOLO_AVAILABLE:
            return
            
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run YOLO detection
            results = self.yolo_model(cv_image, conf=self.confidence_threshold, verbose=False)
            
            # Process detections
            annotated_image, detections_data = self.process_detections(cv_image, results[0])
            
            # Add FPS information if enabled
            if self.show_fps:
                self.fps_counter += 1
                current_time = self.get_clock().now()
                time_diff = (current_time - self.fps_start_time).nanoseconds / 1e9
                
                if time_diff >= 1.0:  # Update FPS every second
                    fps = self.fps_counter / time_diff
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                    
                    # Add FPS text to image
                    cv2.putText(annotated_image, f'FPS: {fps:.1f}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Publish annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            annotated_msg.header = msg.header
            self.image_pub.publish(annotated_msg)
            
            # Publish detection data as JSON
            bbox_msg = String()
            bbox_msg.data = json.dumps(detections_data)
            self.bbox_pub.publish(bbox_msg)
            
            # Publish RViz markers if enabled
            if self.publish_markers:
                markers_msg = self.create_detection_markers(detections_data, msg.header)
                self.markers_pub.publish(markers_msg)
            
            # Save results if enabled
            if self.save_results:
                self.save_detection_results(annotated_image, detections_data, msg.header.stamp)
            
            # Log detection summary
            num_detections = len(detections_data['detections'])
            if num_detections > 0:
                classes = [det['class'] for det in detections_data['detections']]
                class_counts = {}
                for cls in classes:
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                summary = ', '.join([f"{count} {cls}" for cls, count in class_counts.items()])
                self.get_logger().info(f"🎯 Detected: {summary}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def process_detections(self, image, results):
        """Process YOLO detection results and annotate image"""
        annotated_image = image.copy()
        height, width = image.shape[:2]
        
        # Prepare detection data
        detections_data = {
            'timestamp': datetime.now().isoformat(),
            'image_size': {'width': width, 'height': height},
            'model': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'detections': []
        }
        
        # Process each detection
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = box.astype(int)
                class_name = self.yolo_model.names[int(cls)]
                
                # Add to detection data
                detection = {
                    'id': i,
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': {
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1),
                        'center_x': int((x1 + x2) / 2),
                        'center_y': int((y1 + y2) / 2)
                    }
                }
                detections_data['detections'].append(detection)
                
                # Draw bounding box
                color = self.get_class_color(int(cls))
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f"{class_name}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y1, label_size[1] + 10)
                
                # Background rectangle for text
                cv2.rectangle(annotated_image, 
                             (x1, label_y - label_size[1] - 10),
                             (x1 + label_size[0], label_y + 5),
                             color, -1)
                
                # Text
                cv2.putText(annotated_image, label, (x1, label_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw center point
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(annotated_image, (center_x, center_y), 5, color, -1)
        
        # Add detection count
        num_detections = len(detections_data['detections'])
        cv2.putText(annotated_image, f'Objects: {num_detections}', (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_image, detections_data

    def get_class_color(self, class_id):
        """Get consistent color for each class"""
        # Generate color based on class ID
        np.random.seed(class_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color

    def create_detection_markers(self, detections_data, header):
        """Create RViz markers for detected objects"""
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header = header
        clear_marker.ns = "yolo_detections"
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        # Create markers for each detection
        for i, detection in enumerate(detections_data['detections']):
            # Text marker for class name and confidence
            text_marker = Marker()
            text_marker.header = header
            text_marker.header.frame_id = "camera_color_optical_frame"  # Camera frame
            text_marker.ns = "yolo_detections"
            text_marker.id = i * 2  # Even IDs for text
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            # Position the text marker in image coordinates (normalized)
            bbox = detection['bbox']
            # Convert pixel coordinates to normalized coordinates (-1 to 1)
            img_width = detections_data['image_size']['width']
            img_height = detections_data['image_size']['height']
            
            # Position text at top of bounding box
            norm_x = (bbox['center_x'] / img_width - 0.5) * 2.0
            norm_y = (bbox['y1'] / img_height - 0.5) * 2.0
            
            text_marker.pose.position.x = norm_x
            text_marker.pose.position.y = norm_y
            text_marker.pose.position.z = 0.0
            text_marker.pose.orientation.w = 1.0
            
            # Set text content
            text_marker.text = f"{detection['class']}: {detection['confidence']:.2f}"
            
            # Set marker properties
            text_marker.scale.z = 0.1  # Text height
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.lifetime.sec = 1  # Marker expires after 1 second
            
            marker_array.markers.append(text_marker)
            
            # Bounding box marker (as a cube outline)
            bbox_marker = Marker()
            bbox_marker.header = header
            bbox_marker.header.frame_id = "camera_color_optical_frame"
            bbox_marker.ns = "yolo_detections"
            bbox_marker.id = i * 2 + 1  # Odd IDs for bounding boxes
            bbox_marker.type = Marker.LINE_STRIP
            bbox_marker.action = Marker.ADD
            
            # Create bounding box corners in normalized coordinates
            x1_norm = (bbox['x1'] / img_width - 0.5) * 2.0
            x2_norm = (bbox['x2'] / img_width - 0.5) * 2.0
            y1_norm = (bbox['y1'] / img_height - 0.5) * 2.0
            y2_norm = (bbox['y2'] / img_height - 0.5) * 2.0
            
            # Define rectangle corners
            corners = [
                Point(x=x1_norm, y=y1_norm, z=0.0),  # Top-left
                Point(x=x2_norm, y=y1_norm, z=0.0),  # Top-right
                Point(x=x2_norm, y=y2_norm, z=0.0),  # Bottom-right
                Point(x=x1_norm, y=y2_norm, z=0.0),  # Bottom-left
                Point(x=x1_norm, y=y1_norm, z=0.0),  # Close the rectangle
            ]
            
            bbox_marker.points = corners
            
            # Set line properties
            bbox_marker.scale.x = 0.005  # Line width
            
            # Set color based on class
            class_id = hash(detection['class']) % 100  # Simple hash for consistent colors
            np.random.seed(class_id)
            color = np.random.rand(3)
            bbox_marker.color.r = float(color[0])
            bbox_marker.color.g = float(color[1])
            bbox_marker.color.b = float(color[2])
            bbox_marker.color.a = 1.0
            bbox_marker.lifetime.sec = 1
            
            marker_array.markers.append(bbox_marker)
        
        return marker_array

    def save_detection_results(self, annotated_image, detections_data, timestamp):
        """Save detection results to files"""
        try:
            # Create filename with timestamp
            timestamp_sec = timestamp.sec
            timestamp_nsec = timestamp.nanosec
            dt = datetime.fromtimestamp(timestamp_sec + timestamp_nsec / 1e9)
            base_filename = f"yolo_{dt.strftime('%Y%m%d_%H%M%S')}_{timestamp_nsec//1000000:03d}_{self.image_counter:06d}"
            
            # Save annotated image
            image_path = os.path.join(self.save_path, f"{base_filename}.jpg")
            cv2.imwrite(image_path, annotated_image)
            
            # Save detection data as JSON
            json_path = os.path.join(self.save_path, f"{base_filename}.json")
            with open(json_path, 'w') as f:
                json.dump(detections_data, f, indent=2)
            
            self.image_counter += 1
            self.get_logger().debug(f"Saved results: {base_filename}")
            
        except Exception as e:
            self.get_logger().error(f"Error saving results: {str(e)}")

def main(args=None):
    if not YOLO_AVAILABLE:
        print("❌ Cannot start YOLO detection node - ultralytics not available")
        return
    
    rclpy.init(args=args)
    node = YOLODetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("� YOLO Detection Node shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
