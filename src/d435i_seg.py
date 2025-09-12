#!/usr/bin/env python3
"""
ROS2 Node for YOLO11 instance segmentation on depth-aligned camera images
Subscribes to: /camera/camera/aligned_depth_to_color/image_raw (depth image)
               /camera/camera/color/image_raw (color image for visualization)
               /camera/camera/depth/camera_info (camera intrinsics)
Publishes to: /yolo11/segmentation/image (annotated image with masks)
              /yolo11/segmentation/masks (segmentation masks)
              /yolo11/segmentation/data (detection and segmentation data)
              /yolo11/bev/detections (BEV projected detections)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray
from cv_bridge import CvBridge
from bev.msg import BevObject, BevObjectArray
import cv2
import numpy as np
import json
import time
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

class YOLO11SegmentationNode(Node):
    def __init__(self):
        super().__init__('yolo11_segmentation_node')
        
        if not YOLO_AVAILABLE:
            self.get_logger().error("YOLO is not available. Please install ultralytics package.")
            return
        
        # Declare parameters
        self.declare_parameter('model_name', 'yolo11n-seg.pt')  # YOLO11 nano segmentation model
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('output_image_topic', '/yolo11/segmentation/image')
        self.declare_parameter('output_masks_topic', '/yolo11/segmentation/masks')
        self.declare_parameter('output_data_topic', '/yolo11/segmentation/data')
        self.declare_parameter('output_bev_topic', '/yolo11/bev/detections')
        self.declare_parameter('output_bev_data_topic', '/yolo11/bev/objects')  # New topic for structured BEV data
        self.declare_parameter('save_results', True)
        self.declare_parameter('save_path', '/tmp/yolo11_seg_results')
        self.declare_parameter('show_fps', True)
        self.declare_parameter('show_masks', True)
        self.declare_parameter('mask_alpha', 0.4)  # Transparency for mask overlay
        # BEV parameters
        self.declare_parameter('bev_width', 800)
        self.declare_parameter('bev_height', 600)
        self.declare_parameter('bev_x_range', 4.0)  # BEV X range in meters (forward)
        self.declare_parameter('bev_y_range', 3.0)  # BEV Y range in meters (left-right)
        self.declare_parameter('min_distance', 0.1) # Minimum distance in meters (changed from 0.5 to 0.1)
        self.declare_parameter('max_distance', 10.0)# Maximum distance in meters
        
        # Get parameters
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.output_image_topic = self.get_parameter('output_image_topic').get_parameter_value().string_value
        self.output_masks_topic = self.get_parameter('output_masks_topic').get_parameter_value().string_value
        self.output_data_topic = self.get_parameter('output_data_topic').get_parameter_value().string_value
        self.output_bev_topic = self.get_parameter('output_bev_topic').get_parameter_value().string_value
        self.output_bev_data_topic = self.get_parameter('output_bev_data_topic').get_parameter_value().string_value
        self.save_results = self.get_parameter('save_results').get_parameter_value().bool_value
        self.save_path = self.get_parameter('save_path').get_parameter_value().string_value
        self.show_fps = self.get_parameter('show_fps').get_parameter_value().bool_value
        self.show_masks = self.get_parameter('show_masks').get_parameter_value().bool_value
        self.mask_alpha = self.get_parameter('mask_alpha').get_parameter_value().double_value
        
        # BEV parameters
        self.bev_width = self.get_parameter('bev_width').get_parameter_value().integer_value
        self.bev_height = self.get_parameter('bev_height').get_parameter_value().integer_value
        self.bev_x_range = self.get_parameter('bev_x_range').get_parameter_value().double_value
        self.bev_y_range = self.get_parameter('bev_y_range').get_parameter_value().double_value
        self.min_distance = self.get_parameter('min_distance').get_parameter_value().double_value
        self.max_distance = self.get_parameter('max_distance').get_parameter_value().double_value
        
        # Initialize YOLO11 segmentation model
        self.get_logger().info(f"Loading YOLO11 segmentation model: {self.model_name}")
        try:
            self.yolo_model = YOLO(self.model_name)
            self.get_logger().info("✅ YOLO11 segmentation model loaded successfully!")
            
            # Log model info
            if hasattr(self.yolo_model, 'names'):
                self.get_logger().info(f"📋 Model can detect {len(self.yolo_model.names)} classes")
                
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load YOLO11 model: {e}")
            return
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Initialize camera intrinsics
        self.camera_info = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        # Synchronization
        self.depth_image = None
        self.color_image = None
        self.depth_image_timestamp = None
        self.color_image_timestamp = None
        
        # Create save directory if needed
        if self.save_results:
            os.makedirs(self.save_path, exist_ok=True)
            self.get_logger().info(f"Results will be saved to: {self.save_path}")
            self.image_counter = 0
        
        # FPS calculation
        self.fps_counter = 0
        self.fps_start_time = self.get_clock().now()
        self.current_fps = 0.0
        
        # Create subscribers
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            10
        )
        
        self.color_sub = self.create_subscription(
            Image,
            self.color_topic,
            self.color_callback,
            10
        )
            
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )
        
        # Create publishers
        self.image_pub = self.create_publisher(
            Image,
            self.output_image_topic,
            10
        )
        
        self.masks_pub = self.create_publisher(
            Image,
            self.output_masks_topic,
            10
        )
        
        self.data_pub = self.create_publisher(
            String,
            self.output_data_topic,
            10
        )
        
        self.bev_pub = self.create_publisher(
            Image,
            self.output_bev_topic,
            10
        )
        
        self.bev_data_pub = self.create_publisher(
            BevObjectArray,  # 使用自定义消息类型代替String
            self.output_bev_data_topic,
            10
        )
        
        self.get_logger().info(f"🚀 YOLO11 Segmentation Node initialized")
        self.get_logger().info(f"📥 Subscribing to depth: {self.depth_topic}")
        self.get_logger().info(f"📥 Subscribing to color: {self.color_topic}")
        self.get_logger().info(f"📥 Subscribing to camera info: {self.camera_info_topic}")
        self.get_logger().info(f"📤 Publishing annotated images to: {self.output_image_topic}")
        self.get_logger().info(f"📤 Publishing segmentation masks to: {self.output_masks_topic}")
        self.get_logger().info(f"📤 Publishing segmentation data to: {self.output_data_topic}")
        self.get_logger().info(f"📤 Publishing BEV detections to: {self.output_bev_topic}")
        self.get_logger().info(f"📤 Publishing BEV object data to: {self.output_bev_data_topic}")

    def camera_info_callback(self, msg):
        """Store camera intrinsics for depth-to-3D conversion"""
        if self.camera_info is None:
            self.camera_info = msg
            self.fx = msg.k[0]  # Focal length X
            self.fy = msg.k[4]  # Focal length Y
            self.cx = msg.k[2]  # Principal point X
            self.cy = msg.k[5]  # Principal point Y
            self.get_logger().info(f"📷 Camera intrinsics received: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")

    def depth_callback(self, msg):
        """Store depth image for synchronization"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_image_timestamp = msg.header.stamp
            self.process_synchronized_data()
        except Exception as e:
            self.get_logger().error(f"❌ Error processing depth image: {e}")

    def color_callback(self, msg):
        """Store color image for synchronization"""
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.color_image_timestamp = msg.header.stamp
            self.process_synchronized_data()
        except Exception as e:
            self.get_logger().error(f"❌ Error processing color image: {e}")

    def process_synchronized_data(self):
        """Process data when both depth and color images are available"""
        if (self.depth_image is None or self.color_image is None or 
            self.camera_info is None or not YOLO_AVAILABLE):
            return
            
        # Simple timestamp synchronization (within 50ms)
        if (self.depth_image_timestamp is not None and 
            self.color_image_timestamp is not None):
            time_diff = abs((self.depth_image_timestamp.sec + self.depth_image_timestamp.nanosec * 1e-9) - 
                          (self.color_image_timestamp.sec + self.color_image_timestamp.nanosec * 1e-9))
            if time_diff > 0.05:  # 50ms threshold
                return
        
        try:
            # Run YOLO11 segmentation on color image
            results = self.yolo_model(self.color_image, 
                                    conf=self.confidence_threshold,
                                    iou=self.iou_threshold,
                                    verbose=False)
            
            # Process segmentation results with depth data
            annotated_image, mask_image, segmentation_data, bev_image = self.process_segmentation_with_depth(
                self.color_image, self.depth_image, results[0])
            
            # Update FPS
            self.update_fps()
            
            # Publish results
            self.publish_results(annotated_image, mask_image, segmentation_data, bev_image)
            
            # Save results if enabled
            if self.save_results:
                self.save_result_images(annotated_image, mask_image, bev_image)
                
        except Exception as e:
            self.get_logger().error(f"❌ Error in segmentation processing: {e}")

    def image_callback(self, msg):
        """Legacy callback - now replaced by synchronized processing"""
        pass  # Keeping for compatibility, actual processing moved to process_synchronized_data

    def update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = self.get_clock().now()
        time_diff = (current_time - self.fps_start_time).nanoseconds / 1e9
        
        if time_diff >= 1.0:  # Update FPS every second
            self.current_fps = self.fps_counter / time_diff
            self.fps_counter = 0
            self.fps_start_time = current_time

    def process_segmentation(self, image, results):
        """Process YOLO11 segmentation results and create visualizations"""
        annotated_image = image.copy()
        height, width = image.shape[:2]
        
        # Create mask visualization image
        mask_image = np.zeros_like(image)
        
        # Prepare segmentation data
        segmentation_data = {
            'timestamp': datetime.now().isoformat(),
            'image_size': {'width': width, 'height': height},
            'model': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'detections': []
        }
        
        # Process detections and segmentation masks
        if results.boxes is not None and results.masks is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            masks = results.masks.data.cpu().numpy()  # segmentation masks
            
            for i, (box, conf, cls, mask) in enumerate(zip(boxes, confidences, classes, masks)):
                x1, y1, x2, y2 = box.astype(int)
                class_name = self.yolo_model.names[int(cls)]
                
                # Resize mask to match image size
                mask_resized = cv2.resize(mask, (width, height))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Calculate mask area and contours
                mask_area = np.sum(mask_binary)
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Get mask polygon points
                polygon_points = []
                if len(contours) > 0:
                    # Get the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Simplify contour
                    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                    polygon_points = simplified_contour.reshape(-1, 2).tolist()
                
                # Add to segmentation data
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
                    },
                    'segmentation': {
                        'area': int(mask_area),
                        'polygon': polygon_points,
                        'mask_shape': mask_resized.shape
                    }
                }
                segmentation_data['detections'].append(detection)
                
                # Generate color for this class
                color = self.get_class_color(int(cls))
                
                if self.show_masks:
                    # Draw segmentation mask overlay on annotated image
                    mask_colored = np.zeros_like(image)
                    mask_colored[mask_binary == 1] = color
                    annotated_image = cv2.addWeighted(annotated_image, 1-self.mask_alpha, 
                                                    mask_colored, self.mask_alpha, 0)
                    
                    # Draw mask contours
                    cv2.drawContours(annotated_image, contours, -1, color, 2)
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence and area
                label = f"{class_name}: {conf:.2f} (Area: {mask_area}px)"
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
                
                # Add mask to mask visualization image
                mask_image[mask_binary == 1] = color
        
        # Add detection count and info
        num_detections = len(segmentation_data['detections'])
        cv2.putText(annotated_image, f'Objects: {num_detections}', (10, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_image, f'Model: {self.model_name}', (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotated_image, mask_image, segmentation_data

    def get_class_color(self, class_id):
        """Get consistent color for each class"""
        # Generate bright, distinguishable colors based on class ID
        np.random.seed(class_id)
        # Use brighter colors that stand out on dark background
        color_options = [
            (0, 255, 255),    # 青色 - Cyan
            (255, 0, 255),    # 紫色 - Magenta  
            (0, 255, 0),      # 绿色 - Green
            (255, 255, 0),    # 黄色 - Yellow
            (0, 0, 255),      # 红色 - Red
            (255, 128, 0),    # 橙色 - Orange
            (128, 255, 0),    # 亮绿 - Bright Green
            (255, 0, 128),    # 粉色 - Pink
            (0, 128, 255),    # 天蓝 - Sky Blue
            (128, 0, 255),    # 紫罗兰 - Violet
        ]
        return color_options[class_id % len(color_options)]

    def save_segmentation_results(self, annotated_image, mask_image, segmentation_data, timestamp):
        """Save segmentation results to files"""
        try:
            # Create filename with timestamp
            timestamp_sec = timestamp.sec
            timestamp_nsec = timestamp.nanosec
            dt = datetime.fromtimestamp(timestamp_sec + timestamp_nsec / 1e9)
            base_filename = f"yolo11_seg_{dt.strftime('%Y%m%d_%H%M%S')}_{timestamp_nsec//1000000:03d}_{self.image_counter:06d}"
            
            # Save annotated image
            annotated_path = os.path.join(self.save_path, f"{base_filename}_annotated.jpg")
            cv2.imwrite(annotated_path, annotated_image)
            
            # Save mask image
            mask_path = os.path.join(self.save_path, f"{base_filename}_masks.jpg")
            cv2.imwrite(mask_path, mask_image)
            
            # Save segmentation data as JSON
            json_path = os.path.join(self.save_path, f"{base_filename}_data.json")
            with open(json_path, 'w') as f:
                json.dump(segmentation_data, f, indent=2)
            
            self.image_counter += 1
            self.get_logger().debug(f"Saved segmentation results: {base_filename}")
            
        except Exception as e:
            self.get_logger().error(f"Error saving segmentation results: {str(e)}")

    def process_segmentation_with_depth(self, color_image, depth_image, results):
        """Process YOLO11 segmentation results with depth information and BEV projection"""
        
        # First process normal segmentation
        annotated_image, mask_image, segmentation_data = self.process_segmentation(color_image, results)
        
        # Create BEV image
        bev_image = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        
        # Add grid lines to BEV
        self.draw_bev_grid(bev_image)
        
        # Process each detection with depth
        if results.boxes is not None and results.masks is not None:
            self.get_logger().info(f"🔍 Processing {len(results.boxes)} detections for BEV projection")
            bev_objects_count = 0
            
            for i, (box, mask) in enumerate(zip(results.boxes, results.masks)):
                if box.conf[0] < self.confidence_threshold:
                    continue
                    
                # Get detection info
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = self.yolo_model.names[class_id]
                conf = box.conf[0]
                
                # Get mask
                mask_array = mask.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_array, (color_image.shape[1], color_image.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Get depth values for the mask region
                depth_roi = depth_image * mask_binary
                valid_depths = depth_roi[depth_roi > 0]
                
                if len(valid_depths) > 0:
                    # Calculate minimum depth in mm (D435i outputs depth in mm)
                    min_depth_mm = np.min(valid_depths)
                    min_depth_m = min_depth_mm / 1000.0  # Convert to meters
                    
                    # Debug: Log depth information for each object
                    self.get_logger().info(f"🔍 {class_name} depth: {min_depth_m:.2f}m (range: {self.min_distance}-{self.max_distance}m), valid_depths: {len(valid_depths)}")
                    
                    # Filter by distance range
                    if self.min_distance <= min_depth_m <= self.max_distance:
                        # Get center point of bounding box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Convert pixel coordinates to 3D camera coordinates
                        x_3d = (center_x - self.cx) * min_depth_m / self.fx
                        y_3d = (center_y - self.cy) * min_depth_m / self.fy
                        z_3d = min_depth_m
                        
                        # Convert to BEV coordinates
                        # Camera coordinate system: X-right, Y-down, Z-forward
                        # BEV coordinate system: X-forward, Y-left
                        bev_x = z_3d  # Camera Z becomes BEV X (forward)
                        bev_y = -x_3d  # Camera -X becomes BEV Y (left)
                        
                        # Convert to pixel coordinates in BEV image
                        # New coordinate system: origin at bottom center
                        bev_pixel_x = int(self.bev_width/2 + (bev_y / self.bev_y_range) * self.bev_width)
                        bev_pixel_y = int(self.bev_height - (bev_x / self.bev_x_range) * self.bev_height)
                        
                        # Check if point is within BEV image bounds
                        if (0 <= bev_pixel_x < self.bev_width and 
                            0 <= bev_pixel_y < self.bev_height):
                            
                            # Debug: Log successful BEV projection
                            self.get_logger().info(f"✅ {class_name} projected to BEV at pixel({bev_pixel_x}, {bev_pixel_y})")
                            
                            # Calculate object size in BEV first
                            # Get object dimensions in pixels
                            obj_width_px = x2 - x1
                            obj_height_px = y2 - y1
                            
                            # Convert pixel size to real-world size using depth
                            # Object width in meters = (pixel_width * depth) / focal_length
                            obj_width_m = (obj_width_px * min_depth_m) / self.fx
                            obj_height_m = (obj_height_px * min_depth_m) / self.fy
                            
                            # Use mask area for better size estimation
                            mask_area_px = np.sum(mask_binary)
                            # Estimate equivalent circle radius from mask area
                            equivalent_radius_px = np.sqrt(mask_area_px / np.pi)
                            equivalent_radius_m = (equivalent_radius_px * min_depth_m) / ((self.fx + self.fy) / 2)
                            
                            # Convert real-world size to BEV pixel size
                            bev_radius_px = max(3, int((equivalent_radius_m / self.bev_x_range) * self.bev_width * 0.5))
                            bev_radius_px = min(bev_radius_px, 25)  # Limit maximum size
                            
                            bev_objects_count += 1
                            # Debug info
                            self.get_logger().debug(f"🎯 BEV: {class_name} at ({bev_x:.2f}m, {bev_y:.2f}m) -> pixel({bev_pixel_x}, {bev_pixel_y}), radius={bev_radius_px}px")
                            
                            # Draw detection on BEV
                            color = self.get_class_color(class_id)
                            
                            # Draw object with size proportional to its real size
                            cv2.circle(bev_image, (bev_pixel_x, bev_pixel_y), bev_radius_px, color, -1)
                            cv2.circle(bev_image, (bev_pixel_x, bev_pixel_y), bev_radius_px + 3, (255, 255, 255), 3)
                            
                            # Add center point
                            cv2.circle(bev_image, (bev_pixel_x, bev_pixel_y), 3, (0, 0, 0), -1)
                            
                            # Draw label
                            label = f"{class_name}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                            label_x = max(5, min(bev_pixel_x - label_size[0]//2, self.bev_width - label_size[0] - 5))
                            label_y = max(20, bev_pixel_y - bev_radius_px - 25)
                            
                            # Background for text
                            cv2.rectangle(bev_image, 
                                        (label_x - 3, label_y - 15),
                                        (label_x + label_size[0] + 3, label_y + 3),
                                        (0, 0, 0), -1)
                            cv2.rectangle(bev_image, 
                                        (label_x - 3, label_y - 15),
                                        (label_x + label_size[0] + 3, label_y + 3),
                                        (255, 255, 255), 1)
                            cv2.putText(bev_image, label, (label_x, label_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            # Add distance and size info
                            dist_label = f"{min_depth_m:.1f}m"
                            size_label = f"{equivalent_radius_m*2:.2f}m"
                            info_text = f"{dist_label} {size_label}"
                            
                            info_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                            info_x = max(5, min(bev_pixel_x - info_size[0]//2, self.bev_width - info_size[0] - 5))
                            info_y = min(self.bev_height - 10, bev_pixel_y + bev_radius_px + 20)
                            
                            cv2.rectangle(bev_image,
                                        (info_x - 2, info_y - 12),
                                        (info_x + info_size[0] + 2, info_y + 2),
                                        (0, 0, 0), -1)
                            cv2.rectangle(bev_image,
                                        (info_x - 2, info_y - 12),
                                        (info_x + info_size[0] + 2, info_y + 2),
                                        (0, 255, 255), 1)
                            cv2.putText(bev_image, info_text, (info_x, info_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
                        
                        # Update segmentation data with depth info
                        for detection in segmentation_data['detections']:
                            if detection['id'] == i:
                                detection['depth'] = {
                                    'min_depth_m': float(min_depth_m),
                                    'min_depth_mm': float(min_depth_mm),
                                    'camera_3d': {'x': float(x_3d), 'y': float(y_3d), 'z': float(z_3d)},
                                    'bev_coords': {'x': float(bev_x), 'y': float(bev_y)},
                                    'bev_pixels': {'x': int(bev_pixel_x), 'y': int(bev_pixel_y)},
                                    'size_info': {
                                        'width_m': float(obj_width_m),
                                        'height_m': float(obj_height_m),
                                        'equivalent_radius_m': float(equivalent_radius_m),
                                        'equivalent_diameter_m': float(equivalent_radius_m * 2),
                                        'mask_area_px': int(mask_area_px),
                                        'bev_radius_px': int(bev_radius_px)
                                    }
                                }
                                break
                        else:
                            # Debug: Log why object was not projected to BEV
                            self.get_logger().warn(f"❌ {class_name} NOT projected: pixel({bev_pixel_x}, {bev_pixel_y}) outside BEV bounds (0-{self.bev_width-1}, 0-{self.bev_height-1})")
                    else:
                        # Debug: Log why object was filtered out by distance
                        self.get_logger().warn(f"❌ {class_name} filtered by distance: {min_depth_m:.2f}m outside range ({self.min_distance}-{self.max_distance}m)")
                else:
                    # Debug: Log objects with no valid depth
                    self.get_logger().warn(f"❌ {class_name} has no valid depth values from mask region")
        
        return annotated_image, mask_image, segmentation_data, bev_image
    
    def draw_bev_grid(self, bev_image):
        """Draw grid lines on BEV image with origin at bottom center"""
        # Grid spacing in meters
        grid_spacing = 0.25  # 25cm
        
        # Vertical lines (distance markers) - forward direction
        for x_m in np.arange(0, self.bev_x_range + grid_spacing, grid_spacing):
            y_pixel = int(self.bev_height - (x_m / self.bev_x_range) * self.bev_height)
            if y_pixel >= 0:
                # Different line weights for major/minor grid lines
                if x_m % 1.0 == 0:  # Major lines every 1m
                    cv2.line(bev_image, (0, y_pixel), (self.bev_width, y_pixel), (100, 100, 100), 2)
                    # Add distance label
                    if x_m > 0:
                        label = f"{x_m:.0f}m"
                        cv2.putText(bev_image, label, (5, y_pixel - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                else:  # Minor lines every 0.25m
                    cv2.line(bev_image, (0, y_pixel), (self.bev_width, y_pixel), (60, 60, 60), 1)
        
        # Horizontal lines (left-right markers)
        for y_m in np.arange(-self.bev_y_range/2, self.bev_y_range/2 + grid_spacing, grid_spacing):
            x_pixel = int(self.bev_width/2 + (y_m / self.bev_y_range) * self.bev_width)
            if 0 <= x_pixel < self.bev_width:
                if abs(y_m) % 1.0 == 0:  # Major lines every 1m
                    cv2.line(bev_image, (x_pixel, 0), (x_pixel, self.bev_height), (100, 100, 100), 2)
                    # Add distance label for non-zero lines
                    if abs(y_m) > 0.1:
                        label = f"{y_m:+.0f}m"
                        cv2.putText(bev_image, label, (x_pixel + 5, self.bev_height - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                else:  # Minor lines
                    cv2.line(bev_image, (x_pixel, 0), (x_pixel, self.bev_height), (60, 60, 60), 1)
        
        # Center line (camera position) - now at bottom center
        center_x = self.bev_width // 2
        cv2.line(bev_image, (center_x, 0), (center_x, self.bev_height), (0, 255, 0), 2)
        cv2.putText(bev_image, "Camera", (center_x + 10, self.bev_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add coordinate system indicator
        cv2.circle(bev_image, (center_x, self.bev_height - 5), 8, (0, 255, 0), -1)
        cv2.putText(bev_image, "Origin", (center_x - 25, self.bev_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Add size reference legend (move to top right)
        legend_x = self.bev_width - 250
        legend_y = 30
        cv2.putText(bev_image, "Size Reference:", (legend_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw reference circles for different sizes
        ref_sizes = [0.1, 0.3, 0.5, 1.0]  # meters
        for i, size_m in enumerate(ref_sizes):
            ref_radius_px = max(2, int((size_m / self.bev_x_range) * self.bev_width * 0.5))
            ref_x = legend_x + 20 + i * 50
            ref_y = legend_y + 25
            
            cv2.circle(bev_image, (ref_x, ref_y), ref_radius_px, (150, 150, 150), 1)
            cv2.putText(bev_image, f"{size_m}m", (ref_x - 10, ref_y + ref_radius_px + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    
    def publish_results(self, annotated_image, mask_image, segmentation_data, bev_image):
        """Publish all results"""
        try:
            header = self.get_clock().now().to_msg()
            
            # Add FPS to annotated image
            if self.show_fps:
                cv2.putText(annotated_image, f'FPS: {self.current_fps:.1f}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Publish annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            annotated_msg.header.stamp = header
            self.image_pub.publish(annotated_msg)
            
            # Publish mask image
            mask_msg = self.bridge.cv2_to_imgmsg(mask_image, encoding='bgr8')
            mask_msg.header.stamp = header
            self.masks_pub.publish(mask_msg)
            
            # Publish structured BEV data using custom message format
            if len(segmentation_data['detections']) > 0:
                bev_objects_with_depth = []
                
                for detection in segmentation_data['detections']:
                    # Only include objects that have depth and BEV coordinates
                    if 'depth' in detection and 'bev_coords' in detection['depth']:
                        bev_coords = detection['depth']['bev_coords']
                        camera_3d = detection['depth']['camera_3d']
                        size_info = detection['depth']['size_info']
                        
                        # Create BevObject message
                        bev_obj = BevObject()
                        bev_obj.class_name = detection['class']
                        bev_obj.confidence = float(detection['confidence'])
                        
                        # 3D position in camera coordinates
                        bev_obj.camera_3d_position = Point()
                        bev_obj.camera_3d_position.x = float(camera_3d['x'])
                        bev_obj.camera_3d_position.y = float(camera_3d['y'])
                        bev_obj.camera_3d_position.z = float(camera_3d['z'])
                        
                        # BEV position in meters
                        bev_obj.bev_position = Point()
                        bev_obj.bev_position.x = float(bev_coords['x'])  # forward
                        bev_obj.bev_position.y = float(bev_coords['y'])  # left
                        bev_obj.bev_position.z = 0.0
                        
                        # BEV pixel position
                        bev_obj.bev_pixel_position = Point()
                        bev_obj.bev_pixel_position.x = float(detection['depth']['bev_pixels']['x'])
                        bev_obj.bev_pixel_position.y = float(detection['depth']['bev_pixels']['y'])
                        bev_obj.bev_pixel_position.z = 0.0
                        
                        # Size and distance information
                        bev_obj.equivalent_diameter_m = float(size_info['equivalent_diameter_m'])
                        bev_obj.depth_m = float(detection['depth']['min_depth_m'])
                        bev_obj.area_pixels = int(detection['segmentation']['area'])
                        
                        bev_objects_with_depth.append(bev_obj)
                
                if len(bev_objects_with_depth) > 0:
                    # Create BevObjectArray message
                    bev_array = BevObjectArray()
                    bev_array.header.stamp = header
                    bev_array.header.frame_id = "camera_bev"
                    bev_array.objects = bev_objects_with_depth
                    bev_array.total_objects = len(bev_objects_with_depth)
                    
                    # BEV range information
                    bev_array.bev_range = Point()
                    bev_array.bev_range.x = float(self.bev_x_range)
                    bev_array.bev_range.y = float(self.bev_y_range)
                    bev_array.bev_range.z = 0.0
                    
                    # Publish the custom message
                    self.bev_data_pub.publish(bev_array)
                    
                    self.get_logger().info(f"📍 Published BEV structured data for {len(bev_objects_with_depth)} objects using custom message format")
            
            # Publish BEV image (existing code)
            bev_msg = self.bridge.cv2_to_imgmsg(bev_image, encoding='bgr8')
            bev_msg.header.stamp = header
            self.bev_pub.publish(bev_msg)
            
            # Publish segmentation data as JSON
            data_msg = String()
            data_msg.data = json.dumps(segmentation_data)
            self.data_pub.publish(data_msg)
            
            # Log summary
            num_objects = len(segmentation_data['detections'])
            if num_objects > 0:
                classes = [det['class'] for det in segmentation_data['detections']]
                class_counts = {}
                for cls in classes:
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                summary = ', '.join([f"{count} {cls}" for cls, count in class_counts.items()])
                self.get_logger().info(f"🎯 Segmented with BEV: {summary}")
                
        except Exception as e:
            self.get_logger().error(f"❌ Error publishing results: {e}")
    
    def save_result_images(self, annotated_image, mask_image, bev_image):
        """Save result images"""
        try:
            timestamp = self.get_clock().now().to_msg()
            timestamp_sec = timestamp.sec
            timestamp_nsec = timestamp.nanosec
            dt = datetime.fromtimestamp(timestamp_sec + timestamp_nsec / 1e9)
            base_filename = f"yolo11_seg_{dt.strftime('%Y%m%d_%H%M%S')}_{timestamp_nsec//1000000:03d}_{self.image_counter:06d}"
            
            # Save annotated image
            annotated_path = os.path.join(self.save_path, f"{base_filename}_annotated.jpg")
            cv2.imwrite(annotated_path, annotated_image)
            
            # Save mask image
            mask_path = os.path.join(self.save_path, f"{base_filename}_masks.jpg")
            cv2.imwrite(mask_path, mask_image)
            
            # Save BEV image
            bev_path = os.path.join(self.save_path, f"{base_filename}_bev.jpg")
            cv2.imwrite(bev_path, bev_image)
            
            self.image_counter += 1
            
        except Exception as e:
            self.get_logger().error(f"❌ Error saving images: {e}")
def main(args=None):
    if not YOLO_AVAILABLE:
        print("❌ Cannot start YOLO11 segmentation node - ultralytics not available")
        print("💡 Install with: pip install ultralytics")
        return
    
    rclpy.init(args=args)
    node = YOLO11SegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 YOLO11 Segmentation Node shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
