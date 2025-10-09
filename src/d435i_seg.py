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
        # Add configurable save interval (seconds)
        self.declare_parameter('save_interval_sec', 5.0)
        # BEV parameters
        self.declare_parameter('bev_width', 800)
        self.declare_parameter('bev_height', 600)
        self.declare_parameter('bev_x_range', 4.0)  # BEV X range in meters (forward)
        self.declare_parameter('bev_y_range', 3.0)  # BEV Y range in meters (left-right)
        self.declare_parameter('min_distance', 0.1) # Minimum distance in meters (changed from 0.5 to 0.1)
        self.declare_parameter('max_distance', 10.0)# Maximum distance in meters
        self.declare_parameter('bev_size_scale', 1.0)  # New parameter for BEV size scaling
        self.declare_parameter('bev_size_reference_values', '0.1,0.3,0.5,1.0')  # meters (diameters)
        # BEV label/layout options
        self.declare_parameter('bev_draw_connectors', True)   # draw leader lines from text boxes to object
        self.declare_parameter('bev_number_labels', True)     # prefix labels with object id
        self.declare_parameter('bev_label_border_object_color', True)  # use object color for text box border
        # NEW: BEV font configuration (larger defaults)
        self.declare_parameter('bev_label_font_scale', 0.9)
        self.declare_parameter('bev_info_font_scale', 0.8)
        self.declare_parameter('bev_label_thickness', 2)
        self.declare_parameter('bev_info_thickness', 2)
        # ... existing code ...

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
        # Get configurable save interval
        self.save_interval_sec = self.get_parameter('save_interval_sec').get_parameter_value().double_value
        
        # BEV parameters
        self.bev_width = self.get_parameter('bev_width').get_parameter_value().integer_value
        self.bev_height = self.get_parameter('bev_height').get_parameter_value().integer_value
        self.bev_x_range = self.get_parameter('bev_x_range').get_parameter_value().double_value
        self.bev_y_range = self.get_parameter('bev_y_range').get_parameter_value().double_value
        self.min_distance = self.get_parameter('min_distance').get_parameter_value().double_value
        self.max_distance = self.get_parameter('max_distance').get_parameter_value().double_value
        self.bev_size_scale = self.get_parameter('bev_size_scale').get_parameter_value().double_value
        ref_str = self.get_parameter('bev_size_reference_values').get_parameter_value().string_value
        try:
            self.bev_size_ref_values = [float(v.strip()) for v in ref_str.split(',') if v.strip()]
        except Exception:
            self.bev_size_ref_values = [0.1, 0.3, 0.5, 1.0]
        # Read BEV label/layout options
        self.bev_draw_connectors = self.get_parameter('bev_draw_connectors').get_parameter_value().bool_value
        self.bev_number_labels = self.get_parameter('bev_number_labels').get_parameter_value().bool_value
        self.bev_label_border_object_color = self.get_parameter('bev_label_border_object_color').get_parameter_value().bool_value
        # NEW: Read BEV font configuration
        self.bev_label_font_scale = self.get_parameter('bev_label_font_scale').get_parameter_value().double_value
        self.bev_info_font_scale = self.get_parameter('bev_info_font_scale').get_parameter_value().double_value
        self.bev_label_thickness = self.get_parameter('bev_label_thickness').get_parameter_value().integer_value
        self.bev_info_thickness = self.get_parameter('bev_info_thickness').get_parameter_value().integer_value
        
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
            # Initialize last save time
            self.last_save_time = None
        
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
        };
        
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
        # Track all drawn rectangles to avoid overlaps (ellipse, labels, info boxes)
        overlay_rects = []
        if results.boxes is not None and results.masks is not None:
            self.get_logger().info(f"🔍 Processing {len(results.boxes)} detections for BEV projection")
            bev_objects_count = 0
            
            for i, (box, mask) in enumerate(zip(results.boxes, results.masks)):
                if box.conf[0] < self.confidence_threshold:
                    continue

                # ---- Initialize to avoid UnboundLocalError on any branch ----
                obj_width_m = 0.0
                obj_height_m = 0.0
                equivalent_radius_m = 0.0
                footprint_diameter_m = 0.0
                radius_m = 0.0
                bev_radius_px = 0
                bev_width_px = 0
                bev_height_px = 0
                mask_area_px = 0
                depth_for_size_m = 0.0
                min_depth_m = 0.0
                min_depth_mm = 0.0
                avg_depth_m = 0.0
                x_3d = y_3d = z_3d = 0.0
                bev_x = bev_y = 0.0
                bev_pixel_x = bev_pixel_y = -1
                has_depth = False
                has_size = False
                # -------------------------------------------------------------

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
                    # Depth stats
                    min_depth_mm = np.min(valid_depths)
                    avg_depth_mm = float(np.mean(valid_depths))
                    min_depth_m = min_depth_mm / 1000.0
                    avg_depth_m = avg_depth_mm / 1000.0
                    depth_for_size_m = avg_depth_m
                    has_depth = True

                    self.get_logger().info(
                        f"🔍 {class_name} depth: {min_depth_m:.2f}m (range: {self.min_distance}-{self.max_distance}m), valid_depths: {len(valid_depths)}"
                    )
                    
                    # Distance filter
                    if self.min_distance <= min_depth_m <= self.max_distance:
                        # Center of bbox
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Camera 3D (use min depth for position)
                        x_3d = (center_x - self.cx) * min_depth_m / self.fx if self.fx else 0.0
                        y_3d = (center_y - self.cy) * min_depth_m / self.fy if self.fy else 0.0
                        z_3d = min_depth_m
                        
                        # BEV meters
                        bev_x = z_3d
                        bev_y = -x_3d
                        
                        # BEV pixels (origin: bottom-center)
                        bev_pixel_x = int(self.bev_width/2 + (bev_y / self.bev_y_range) * self.bev_width)
                        bev_pixel_y = int(self.bev_height - (bev_x / self.bev_x_range) * self.bev_height)

                        # Scales
                        scale_lateral = self.bev_width / self.bev_y_range   # px/m
                        scale_forward = self.bev_height / self.bev_x_range  # px/m

                        # Compute metric size using avg depth (more stable)
                        obj_width_px = max(1, x2 - x1)
                        obj_height_px = max(1, y2 - y1)
                        if depth_for_size_m > 0 and self.fx and self.fy:
                            obj_width_m = (obj_width_px * depth_for_size_m) / self.fx
                            obj_height_m = (obj_height_px * depth_for_size_m) / self.fy

                        # Mask → equivalent radius in meters
                        mask_area_px = int(np.sum(mask_binary))
                        if mask_area_px > 0 and depth_for_size_m > 0:
                            equivalent_radius_px = np.sqrt(mask_area_px / np.pi)
                            f_mean = (self.fx + self.fy) / 2.0 if (self.fx and self.fy) else (self.fx or 1.0)
                            equivalent_radius_m = (equivalent_radius_px * depth_for_size_m) / f_mean
                        else:
                            equivalent_radius_m = 0.0
                        
                        # Footprint diameter (fallback to equivalent circle)
                        footprint_diameter_m = max(obj_width_m, obj_height_m)
                        if footprint_diameter_m <= 0 and equivalent_radius_m > 0:
                            footprint_diameter_m = equivalent_radius_m * 2.0
                        if footprint_diameter_m <= 0:
                            footprint_diameter_m = 0.10  # final fallback

                        radius_m = max(0.05, footprint_diameter_m / 2.0)
                        isotropic_scale = (scale_lateral + scale_forward) / 2.0
                        bev_radius_px = int(radius_m * isotropic_scale * self.bev_size_scale)
                        bev_radius_px = max(3, min(bev_radius_px, 40))

                        bev_width_px = int(obj_width_m * scale_lateral * self.bev_size_scale)
                        bev_height_px = int(obj_height_m * scale_forward * self.bev_size_scale)
                        if bev_width_px <= 0:  bev_width_px = bev_radius_px * 2
                        if bev_height_px <= 0: bev_height_px = bev_radius_px * 2
                        bev_width_px = max(6, min(bev_width_px, 80))
                        bev_height_px = max(6, min(bev_height_px, 80))
                        has_size = True

                        # Draw only if inside BEV bounds
                        if (0 <= bev_pixel_x < self.bev_width and 0 <= bev_pixel_y < self.bev_height):
                            self.get_logger().info(
                                f"✅ {class_name} projected to BEV at pixel({bev_pixel_x}, {bev_pixel_y})"
                            )
                            color = self.get_class_color(class_id)
                            center = (bev_pixel_x, bev_pixel_y)
                            axes = (bev_width_px // 2, bev_height_px // 2)
                            cv2.ellipse(bev_image, center, axes, 0, 0, 360, color, -1)
                            cv2.ellipse(bev_image, center, axes, 0, 0, 360, (255, 255, 255), 2)
                            cv2.circle(bev_image, center, 3, (0, 0, 0), -1)

                            # Mark ellipse bounding rect as occupied（避免标签遮盖椭圆）
                            ell_rect = [
                                max(0, center[0] - axes[0] - 2),
                                max(0, center[1] - axes[1] - 2),
                                min(self.bev_width - 1,  center[0] + axes[0] + 2),
                                min(self.bev_height - 1, center[1] + axes[1] + 2)
                            ]
                            overlay_rects.append(ell_rect)

                            # Labels（类名）
                            obj_id = int(i)
                            label = f"#{obj_id} {class_name}" if self.bev_number_labels else f"{class_name}"
                            (lw, lh), lbase = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.bev_label_font_scale, self.bev_label_thickness)
                            lpad = max(3, int(4 * self.bev_label_font_scale))
                            lbox_w = lw + lpad * 2
                            lbox_h = lh + lpad * 2
                            # candidates: above center, above-left, above-right, right, left, below
                            cand_label = [
                                (bev_pixel_x - lbox_w // 2, bev_pixel_y - axes[1] - lbox_h - 8),
                                (bev_pixel_x - axes[0] - lbox_w - 8, bev_pixel_y - axes[1] - lbox_h - 8),
                                (bev_pixel_x + axes[0] + 8,        bev_pixel_y - axes[1] - lbox_h - 8),
                                (bev_pixel_x + axes[0] + 8,        bev_pixel_y - lbox_h // 2),
                                (bev_pixel_x - axes[0] - lbox_w - 8, bev_pixel_y - lbox_h // 2),
                                (bev_pixel_x - lbox_w // 2,        bev_pixel_y + axes[1] + 8),
                            ]
                            lrect = self._place_box_no_overlap(lbox_w, lbox_h, cand_label, overlay_rects, self.bev_width, self.bev_height)
                            cv2.rectangle(bev_image, (lrect[0], lrect[1]), (lrect[2], lrect[3]), (0, 0, 0), -1)
                            border_label = color if self.bev_label_border_object_color else (255, 255, 255)
                            cv2.rectangle(bev_image, (lrect[0], lrect[1]), (lrect[2], lrect[3]), border_label, 1)
                            cv2.putText(
                                bev_image,
                                label,
                                (lrect[0] + lpad, lrect[1] + lpad + lh),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                self.bev_label_font_scale,
                                (255, 255, 255),
                                self.bev_label_thickness,
                            )
                            overlay_rects.append(lrect)

                            # Info box（距离与尺寸，三位小数）
                            dist_label = f"{min_depth_m:.3f}m"
                            size_label = f"{equivalent_radius_m*2:.3f}m"
                            info_text = f"#{obj_id} {dist_label} {size_label}" if self.bev_number_labels else f"{dist_label} {size_label}"
                            (tw, th), tbase = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, self.bev_info_font_scale, self.bev_info_thickness)
                            tpad = max(2, int(3 * self.bev_info_font_scale))
                            tbox_w = tw + tpad * 2
                            tbox_h = th + tpad * 2
                            # candidates: below center, below-right, below-left, above center, right, left
                            cand_info = [
                                (bev_pixel_x - tbox_w // 2, bev_pixel_y + axes[1] + 8),
                                (bev_pixel_x + axes[0] + 8, bev_pixel_y + axes[1] + 8),
                                (bev_pixel_x - axes[0] - tbox_w - 8, bev_pixel_y + axes[1] + 8),
                                (bev_pixel_x - tbox_w // 2, bev_pixel_y - axes[1] - tbox_h - 8),
                                (bev_pixel_x + axes[0] + 8, bev_pixel_y - tbox_h // 2),
                                (bev_pixel_x - axes[0] - tbox_w - 8, bev_pixel_y - tbox_h // 2),
                            ]
                            trect = self._place_box_no_overlap(tbox_w, tbox_h, cand_info, overlay_rects, self.bev_width, self.bev_height)
                            cv2.rectangle(bev_image, (trect[0], trect[1]), (trect[2], trect[3]), (0, 0, 0), -1)
                            border_info = color if self.bev_label_border_object_color else (0, 255, 255)
                            cv2.rectangle(bev_image, (trect[0], trect[1]), (trect[2], trect[3]), border_info, 1)
                            cv2.putText(
                                bev_image,
                                info_text,
                                (trect[0] + tpad, trect[1] + tpad + th),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                self.bev_info_font_scale,
                                (0, 255, 255),
                                self.bev_info_thickness,
                            )
                            overlay_rects.append(trect)

                            # Optional: draw connectors from boxes to object center
                            if self.bev_draw_connectors:
                                lcx = (lrect[0] + lrect[2]) // 2
                                lcy = (lrect[1] + lrect[3]) // 2
                                tcx = (trect[0] + trect[2]) // 2
                                tcy = (trect[1] + trect[3]) // 2
                                cv2.line(bev_image, (lcx, lcy), center, color, 1, cv2.LINE_AA)
                                cv2.line(bev_image, (tcx, tcy), center, color, 1, cv2.LINE_AA)
                                cv2.circle(bev_image, (lcx, lcy), 2, color, -1)
                                cv2.circle(bev_image, (tcx, tcy), 2, color, -1)
                        else:
                            self.get_logger().warn(
                                f"❌ {class_name} NOT projected: pixel({bev_pixel_x},{bev_pixel_y}) out of BEV bounds"
                            )
                    else:
                        self.get_logger().warn(
                            f"❌ {class_name} filtered by distance: {min_depth_m:.2f}m "
                            f"(allowed {self.min_distance}-{self.max_distance}m)"
                        )
                else:
                    self.get_logger().warn(f"❌ {class_name} has no valid depth values")

                # Write back only if depth computed
                if has_depth:
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
                                    'footprint_diameter_m': float(footprint_diameter_m),
                                    'visual_radius_m': float(radius_m),
                                    'bev_radius_px': int(bev_radius_px),
                                    'mask_area_px': int(mask_area_px),
                                    'depth_basis': 'avg',
                                    'size_depth_m': float(depth_for_size_m)
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
        """Draw grid lines on BEV image with origin at bottom center (updated size reference)"""
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
        """Save result images with a configurable minimum interval."""
        try:
            # Respect runtime parameter toggle as well
            if not getattr(self, 'save_results', True):
                return

            # Ensure directory exists even if toggled at runtime
            os.makedirs(self.save_path, exist_ok=True)

            current_time = self.get_clock().now().to_msg()
            last_t = getattr(self, 'last_save_time', None)
            if last_t is not None:
                time_diff = (current_time.sec - last_t.sec) + (current_time.nanosec - last_t.nanosec) / 1e9
                if time_diff < max(0.0, float(getattr(self, 'save_interval_sec', 1.0))):
                    return  # Skip saving if interval not reached

            # Update last save time
            self.last_save_time = current_time

            # Build filename
            timestamp_sec = current_time.sec
            timestamp_nsec = current_time.nanosec
            dt = datetime.fromtimestamp(timestamp_sec + timestamp_nsec / 1e9)
            base_filename = f"yolo11_seg_{dt.strftime('%Y%m%d_%H%M%S')}_{timestamp_nsec//1000000:03d}_{getattr(self, 'image_counter', 0):06d}"

            # Write files
            annotated_path = os.path.join(self.save_path, f"{base_filename}_annotated.jpg")
            mask_path = os.path.join(self.save_path, f"{base_filename}_masks.jpg")
            bev_path = os.path.join(self.save_path, f"{base_filename}_bev.jpg")

            cv2.imwrite(annotated_path, annotated_image)
            cv2.imwrite(mask_path, mask_image)
            cv2.imwrite(bev_path, bev_image)

            # Increment counter
            self.image_counter = getattr(self, 'image_counter', 0) + 1
            self.get_logger().debug(f"Saved images: {annotated_path}, {mask_path}, {bev_path}")

        except Exception as e:
            self.get_logger().error(f"❌ Error saving images: {e}")

    # ---------- Helpers for non-overlapping text boxes on BEV ----------
    def _intersects(self, a, b):
        # a,b: [x1,y1,x2,y2]
        return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

    def _place_box_no_overlap(self, box_w, box_h, candidates, taken_rects, canvas_w, canvas_h):
        # candidates: list of (x1, y1) top-left candidates
        pad = 0
        for (cx, cy) in candidates:
            # clamp to canvas
            x1 = max(0, min(cx, canvas_w - box_w))
            y1 = max(0, min(cy, canvas_h - box_h))
            rect = [x1, y1, x1 + box_w, y1 + box_h]
            # try small vertical shifts to clear collisions
            if any(self._intersects(rect, r) for r in taken_rects):
                step = 12
                # downwards
                for k in range(1, 8):
                    yk = min(canvas_h - box_h, y1 + k * step)
                    rect_k = [x1, yk, x1 + box_w, yk + box_h]
                    if not any(self._intersects(rect_k, r) for r in taken_rects):
                        return rect_k
                # upwards
                for k in range(1, 8):
                    yk = max(0, y1 - k * step)
                    rect_k = [x1, yk, x1 + box_w, yk + box_h]
                    if not any(self._intersects(rect_k, r) for r in taken_rects):
                        return rect_k
                # rightwards
                for k in range(1, 8):
                    xk = min(canvas_w - box_w, x1 + k * step)
                    rect_k = [xk, y1, xk + box_w, y1 + box_h]
                    if not any(self._intersects(rect_k, r) for r in taken_rects):
                        return rect_k
                # leftwards
                for k in range(1, 8):
                    xk = max(0, x1 - k * step)
                    rect_k = [xk, y1, xk + box_w, y1 + box_h]
                    if not any(self._intersects(rect_k, r) for r in taken_rects):
                        return rect_k
            else:
                return rect
        # fallback: bottom-left corner
        return [pad, canvas_h - box_h - pad, pad + box_w, canvas_h - pad]

def main():
    rclpy.init()
    node = YOLO11SegmentationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
