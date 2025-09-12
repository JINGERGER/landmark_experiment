import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import os
from datetime import datetime

class D435IBEVProjection(Node):
    def __init__(self):
        super().__init__('d435i_bev_projection')
        
        # Declare parameters with debug info
        self.get_logger().debug("Declaring node parameters...")
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('bev_topic', '/bev/image')
        self.declare_parameter('bev_width', 800)    # BEV image width in pixels
        self.declare_parameter('bev_height', 600)   # BEV image height in pixels
        self.declare_parameter('bev_x_range', 4.0)  # BEV X range in meters (forward)
        self.declare_parameter('bev_y_range', 3.0)  # BEV Y range in meters (left-right)
        self.declare_parameter('min_distance', 0.5) # Minimum distance in meters
        self.declare_parameter('max_distance', 10.0)# Maximum distance in meters
        self.declare_parameter('save_images', True) # Save BEV images to local files
        self.declare_parameter('save_path', '/tmp/bev_images')  # Path to save images
        
        # Get parameters with debug info
        self.get_logger().debug("Retrieving node parameters...")
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.bev_topic = self.get_parameter('bev_topic').value
        self.bev_width = self.get_parameter('bev_width').value
        self.bev_height = self.get_parameter('bev_height').value
        self.bev_x_range = self.get_parameter('bev_x_range').value
        self.bev_y_range = self.get_parameter('bev_y_range').value
        self.min_distance = self.get_parameter('min_distance').value
        self.max_distance = self.get_parameter('max_distance').value
        self.save_images = self.get_parameter('save_images').value
        self.save_path = self.get_parameter('save_path').value
        
        self.get_logger().debug(f"Parameters loaded - Depth topic: {self.depth_topic}, BEV topic: {self.bev_topic}")
        
        # Initialize variables
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create save directory if saving is enabled
        if self.save_images:
            os.makedirs(self.save_path, exist_ok=True)
            self.get_logger().info(f'BEV images will be saved to: {self.save_path}')
            self.image_counter = 0
        
        # Create subscribers
        self.get_logger().debug("Creating subscribers...")
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            10)
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10)
        
        # Create publisher
        self.get_logger().debug("Creating BEV image publisher...")
        self.bev_pub = self.create_publisher(
            Image,
            self.bev_topic,
            10)
        
        # Set logger level to DEBUG
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        
        self.get_logger().info('D435I BEV Projection node initialized successfully')

    def camera_info_callback(self, msg):
        """Process camera intrinsic parameters"""
        self.get_logger().debug("Received camera information message")
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
        
        # Log camera parameters for debugging
        self.get_logger().debug(f"Camera matrix received:\n{self.camera_matrix}")
        self.get_logger().debug(f"Distortion coefficients: {self.dist_coeffs}")
        
        # Only need camera info once
        self.destroy_subscription(self.camera_info_sub)
        self.get_logger().info('Camera calibration data received and stored')

    def depth_callback(self, msg):
        """Process depth image and project to BEV"""
        self.get_logger().debug(f"Received depth image with timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        
        if self.camera_matrix is None:
            self.get_logger().warn('Waiting for camera calibration data before processing depth images...')
            return
        
        try:
            # Convert ROS image to OpenCV format
            self.get_logger().debug("Converting depth image from ROS message to OpenCV format")
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.get_logger().debug(f"Depth image dimensions: {depth_image.shape[0]}x{depth_image.shape[1]}")
            
            # Convert depth image to point cloud
            self.get_logger().debug("Starting conversion from depth image to point cloud")
            point_cloud = self.depth_to_point_cloud(depth_image)
            self.get_logger().debug(f"Generated point cloud with {len(point_cloud)} valid points")
            
            # Transform point cloud if transform is available
            transformed = False
            try:
                self.get_logger().debug(f"Attempting to get transform from {msg.header.frame_id} to base_link")
                transform = self.tf_buffer.lookup_transform(
                    'base_link',
                    msg.header.frame_id,
                    rclpy.time.Time())
                point_cloud = self.transform_point_cloud(point_cloud, transform)
                transformed = True
                self.get_logger().debug("Successfully transformed point cloud to base_link coordinate system")
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f'Transform lookup failed: {str(e)}. Using camera coordinate system with manual conversion.')
                # Apply manual coordinate transformation from camera to robot frame
                point_cloud = self.camera_to_robot_transform(point_cloud)
                transformed = True
            
            # Project to BEV
            self.get_logger().debug("Projecting point cloud to BEV image")
            bev_image = self.point_cloud_to_bev(point_cloud)
            
            # Save BEV image to local file if enabled
            if self.save_images:
                self.save_bev_image(bev_image, msg.header.stamp)
                # Also save the original depth image for comparison
                self.save_depth_image(depth_image, msg.header.stamp)
            
            # Publish BEV image
            self.get_logger().debug("Converting BEV image to ROS message and publishing")
            bev_msg = self.bridge.cv2_to_imgmsg(bev_image, encoding='bgr8')
            bev_msg.header = msg.header
            bev_msg.header.frame_id = 'base_link' if transformed else msg.header.frame_id
            self.bev_pub.publish(bev_msg)
            self.get_logger().debug(f"BEV image published to {self.bev_topic} topic")
            
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def depth_to_point_cloud(self, depth_image):
        """Convert depth image to 3D point cloud in camera coordinates"""
        height, width = depth_image.shape
        self.get_logger().debug(f"Processing depth image of size {width}x{height}")
        
        # Create pixel coordinate grid
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u.flatten()
        v = v.flatten()
        
        # Get depth values and convert to meters
        z = depth_image.flatten() / 1000.0
        self.get_logger().debug(f"Depth values range: min={np.min(z)}, max={np.max(z)} (before filtering)")
        
        # Filter invalid depth values
        valid = (z > self.min_distance) & (z < self.max_distance) & (z > 0)
        valid_count = np.sum(valid)
        invalid_count = len(z) - valid_count
        self.get_logger().debug(f"Filtered {invalid_count} invalid depth points. Keeping {valid_count} valid points.")
        
        u = u[valid]
        v = v[valid]
        z = z[valid]
        
        if valid_count == 0:
            self.get_logger().warn("No valid depth points remaining after filtering")
            return np.array([])
        
        # Camera intrinsics
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        self.get_logger().debug(f"Using camera intrinsics - fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
        
        # Convert to 3D points in camera coordinates
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Return point cloud (x, y, z)
        return np.column_stack((x, y, z))

    def transform_point_cloud(self, point_cloud, transform):
        """Transform point cloud from camera coordinates to target coordinates"""
        if len(point_cloud) == 0:
            self.get_logger().debug("No points to transform - returning empty point cloud")
            return point_cloud
            
        # Extract transform parameters
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        tz = transform.transform.translation.z
        
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w
        
        self.get_logger().debug(f"Applying transform - translation: ({tx}, {ty}, {tz}), rotation quaternion: ({qx}, {qy}, {qz}, {qw})")
        
        # Convert quaternion to rotation matrix
        def quaternion_to_rotation_matrix(q):
            x, y, z, w = q
            return np.array([
                [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
                [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
                [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
            ])
        
        rotation_matrix = quaternion_to_rotation_matrix([qx, qy, qz, qw])
        self.get_logger().debug(f"Rotation matrix computed:\n{rotation_matrix}")
        
        # Apply rotation and translation
        transformed_points = []
        for point in point_cloud:
            # Rotate
            rotated = rotation_matrix @ point
            # Translate
            transformed = rotated + [tx, ty, tz]
            transformed_points.append(transformed)
            
        return np.array(transformed_points)

    def camera_to_robot_transform(self, point_cloud):
        """Transform point cloud from camera coordinate system to robot coordinate system
        Camera coordinates: X-right, Y-down, Z-forward
        Robot coordinates: X-forward, Y-left, Z-up
        """
        if len(point_cloud) == 0:
            return point_cloud
            
        self.get_logger().debug("Applying manual camera to robot coordinate transformation")
        
        # Extract camera coordinates
        x_cam = point_cloud[:, 0]  # Right in camera frame
        y_cam = point_cloud[:, 1]  # Down in camera frame  
        z_cam = point_cloud[:, 2]  # Forward in camera frame
        
        # Transform to robot coordinates
        # Robot X (forward) = Camera Z (forward)
        # Robot Y (left) = -Camera X (right becomes left)
        # Robot Z (up) = -Camera Y (down becomes up)
        x_robot = z_cam    # Forward
        y_robot = -x_cam   # Left
        z_robot = -y_cam   # Up
        
        transformed_cloud = np.column_stack((x_robot, y_robot, z_robot))
        
        self.get_logger().debug(f"Coordinate transformation applied. Original range - Camera X: [{np.min(x_cam):.2f}, {np.max(x_cam):.2f}], Y: [{np.min(y_cam):.2f}, {np.max(y_cam):.2f}], Z: [{np.min(z_cam):.2f}, {np.max(z_cam):.2f}]")
        self.get_logger().debug(f"Transformed range - Robot X: [{np.min(x_robot):.2f}, {np.max(x_robot):.2f}], Y: [{np.min(y_robot):.2f}, {np.max(y_robot):.2f}], Z: [{np.min(z_robot):.2f}, {np.max(z_robot):.2f}]")
        
        # Debug: Show some sample transformations for verification
        self.log_sample_transformations(point_cloud, transformed_cloud)
        
        return transformed_cloud

    def log_sample_transformations(self, original_cloud, transformed_cloud):
        """Log sample point transformations for debugging coordinate conversion"""
        if len(original_cloud) > 10:  # Only if we have enough points
            # Take some sample points for verification
            sample_indices = [0, len(original_cloud)//4, len(original_cloud)//2, len(original_cloud)*3//4, -1]
            
            self.get_logger().debug("Sample coordinate transformations:")
            self.get_logger().debug("Camera (X_cam, Y_cam, Z_cam) -> Robot (X_robot, Y_robot, Z_robot)")
            
            for i in sample_indices:
                if i < len(original_cloud):
                    orig = original_cloud[i]
                    trans = transformed_cloud[i]
                    self.get_logger().debug(f"({orig[0]:.2f}, {orig[1]:.2f}, {orig[2]:.2f}) -> ({trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f})")
        
        # Also log the transformation rule being used
        self.get_logger().debug("Transformation rule: Camera->Robot")
        self.get_logger().debug("Robot_X (forward) = Camera_Z (forward)")
        self.get_logger().debug("Robot_Y (left) = -Camera_X (right->left)")  
        self.get_logger().debug("Robot_Z (up) = -Camera_Y (down->up)")

    def save_bev_image(self, bev_image, timestamp):
        """Save BEV image to local file"""
        try:
            # Create filename with timestamp
            timestamp_sec = timestamp.sec
            timestamp_nsec = timestamp.nanosec
            dt = datetime.fromtimestamp(timestamp_sec + timestamp_nsec / 1e9)
            filename = f"bev_{dt.strftime('%Y%m%d_%H%M%S')}_{timestamp_nsec//1000000:03d}_{self.image_counter:06d}.png"
            
            # Full path
            filepath = os.path.join(self.save_path, filename)
            
            # Save image
            success = cv2.imwrite(filepath, bev_image)
            if success:
                self.get_logger().debug(f"Saved BEV image to: {filepath}")
                self.image_counter += 1
            else:
                self.get_logger().error(f"Failed to save BEV image to: {filepath}")
                
        except Exception as e:
            self.get_logger().error(f"Error saving BEV image: {str(e)}")

    def save_depth_image(self, depth_image, timestamp):
        """Save original depth image to local file for comparison"""
        try:
            # Create filename with timestamp (matching BEV naming convention)
            timestamp_sec = timestamp.sec
            timestamp_nsec = timestamp.nanosec
            dt = datetime.fromtimestamp(timestamp_sec + timestamp_nsec / 1e9)
            filename = f"depth_{dt.strftime('%Y%m%d_%H%M%S')}_{timestamp_nsec//1000000:03d}_{self.image_counter:06d}.png"
            
            # Full path
            filepath = os.path.join(self.save_path, filename)
            
            # Convert depth image to a visualizable format
            # Normalize depth values to 0-255 range for better visualization
            depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
            
            # Only consider valid depth values for normalization
            valid_mask = (depth_image > 0) & (depth_image < 65535)  # Avoid invalid/max values
            if np.any(valid_mask):
                valid_depths = depth_image[valid_mask]
                min_depth = np.min(valid_depths)
                max_depth = np.max(valid_depths)
                
                if max_depth > min_depth:
                    # Normalize to 0-255 range
                    depth_normalized[valid_mask] = ((valid_depths - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
                else:
                    # If all depths are the same, set to middle gray
                    depth_normalized[valid_mask] = 128
            
            # Apply colormap for better visualization (JET colormap: blue=close, red=far)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Add depth information overlay
            self.add_depth_info_overlay(depth_colored, depth_image, min_depth if np.any(valid_mask) else 0, 
                                      max_depth if np.any(valid_mask) else 0)
            
            # Save image
            success = cv2.imwrite(filepath, depth_colored)
            if success:
                self.get_logger().debug(f"Saved depth image to: {filepath}")
            else:
                self.get_logger().error(f"Failed to save depth image to: {filepath}")
                
        except Exception as e:
            self.get_logger().error(f"Error saving depth image: {str(e)}")

    def add_depth_info_overlay(self, image, depth_image, min_depth, max_depth):
        """Add information overlay to depth image"""
        # Add title and depth range info
        cv2.putText(image, 'Original Depth Image', (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f'Depth Range: {min_depth/1000.0:.2f}m - {max_depth/1000.0:.2f}m', (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add colorbar legend
        legend_x, legend_y = 20, 100
        cv2.putText(image, 'Color Scale:', (legend_x, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, 'Blue = Close', (legend_x, legend_y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(image, 'Green = Medium', (legend_x, legend_y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(image, 'Red = Far', (legend_x, legend_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Add center cross for reference
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        cv2.line(image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
        cv2.line(image, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)
        cv2.putText(image, 'CENTER', (center_x - 30, center_y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Add directional markers for better correspondence with BEV
        # Left and Right markers
        cv2.arrowedLine(image, (50, center_y), (100, center_y), (0, 255, 0), 3, tipLength=0.3)
        cv2.putText(image, 'LEFT', (105, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.arrowedLine(image, (width - 50, center_y), (width - 100, center_y), (0, 255, 0), 3, tipLength=0.3)
        cv2.putText(image, 'RIGHT', (width - 150, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Forward direction marker
        cv2.arrowedLine(image, (center_x, 80), (center_x, 130), (0, 0, 255), 3, tipLength=0.3)
        cv2.putText(image, 'FORWARD', (center_x - 30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Add coordinate system info
        cv2.putText(image, 'Camera Frame:', (width - 200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, 'X: Right', (width - 200, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, 'Y: Down', (width - 200, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, 'Z: Forward', (width - 200, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def point_cloud_to_bev(self, point_cloud):
        """Project point cloud to BEV (Bird's Eye View) coordinate system"""
        if len(point_cloud) == 0:
            self.get_logger().warn("Received empty point cloud for BEV projection")
            return np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
            
        # Create BEV image with white background for better visualization
        bev_image = np.full((self.bev_height, self.bev_width, 3), 255, dtype=np.uint8)
        self.get_logger().debug(f"Created BEV image canvas with dimensions {self.bev_width}x{self.bev_height}")
        
        # Filter points for true bird's eye view (remove points too close to ground/camera)
        x_min, x_max = 0.0, self.bev_x_range  # X: 0 to forward range (forward direction)
        y_min, y_max = -self.bev_y_range/2, self.bev_y_range/2  # Y: symmetric around 0 (left-right)
        z_min_filter = -0.5  # Filter points below this height (ground plane)
        z_max_filter = 3.0   # Filter points above this height (ceiling/sky)
        
        mask = (point_cloud[:, 0] > x_min) & (point_cloud[:, 0] < x_max) & \
               (point_cloud[:, 1] > y_min) & (point_cloud[:, 1] < y_max) & \
               (point_cloud[:, 2] > z_min_filter) & (point_cloud[:, 2] < z_max_filter)
        points = point_cloud[mask]
        
        self.get_logger().debug(f"Filtered {len(point_cloud) - len(points)} points outside BEV range. Keeping {len(points)} points.")
        
        if len(points) == 0:
            self.get_logger().warn("No points remaining after BEV range filtering")
            return bev_image
        
        # Extract coordinates for bird's eye view projection
        x = points[:, 0]  # Forward direction (X in robot frame) 
        y = points[:, 1]  # Left direction (Y in robot frame)  
        z = points[:, 2]  # Height (Z in robot frame)
        
        self.get_logger().debug(f"Point coordinates range - X: [{np.min(x):.2f}, {np.max(x):.2f}], Y: [{np.min(y):.2f}, {np.max(y):.2f}], Z: [{np.min(z):.2f}, {np.max(z):.2f}]")
        
        # Convert to BEV pixel coordinates
        # X (forward) maps to image height (top=forward, bottom=backward)
        # Y (left) maps to image width (left=left, right=right)
        pixel_x = ((y - y_min) / (y_max - y_min) * (self.bev_width - 1)).astype(np.int32)
        pixel_y = (self.bev_height - 1 - ((x - x_min) / (x_max - x_min) * (self.bev_height - 1))).astype(np.int32)
        
        # Debug: Log pixel mapping for verification
        self.get_logger().debug(f"BEV mapping - Y range [{y_min:.2f}, {y_max:.2f}] -> pixel_x [0, {self.bev_width-1}]")
        self.get_logger().debug(f"BEV mapping - X range [{x_min:.2f}, {x_max:.2f}] -> pixel_y [0, {self.bev_height-1}] (inverted)")
        
        # Log some sample mappings
        if len(points) > 5:
            sample_indices = [0, len(points)//4, len(points)//2, len(points)*3//4, -1]
            self.get_logger().debug("Sample Robot->Pixel mappings (X=distance, Y=left/right, Z=height):")
            for i in sample_indices:
                if i < len(points):
                    distance = x[i]  # Distance from robot
                    self.get_logger().debug(f"Robot({x[i]:.2f}m, {y[i]:.2f}m, {z[i]:.2f}m) -> Pixel({pixel_x[i]}, {pixel_y[i]}) [Dist:{distance:.2f}m]")
        
        # Ensure pixel coordinates are within bounds
        pixel_x = np.clip(pixel_x, 0, self.bev_width - 1)
        pixel_y = np.clip(pixel_y, 0, self.bev_height - 1)
        
        # Create height-based and distance-based visualization
        z_range = np.max(z) - np.min(z)
        x_range = np.max(x) - np.min(x)  # Distance range (forward direction)
        
        self.get_logger().debug(f"Height range for coloring: [{np.min(z):.2f}m, {np.max(z):.2f}m], range: {z_range:.2f}m")
        self.get_logger().debug(f"Distance range for coloring: [{np.min(x):.2f}m, {np.max(x):.2f}m], range: {x_range:.2f}m")
        
        if z_range > 0.01 and x_range > 0.01:  # If there's meaningful height and distance variation
            # Combine distance and height information for color mapping
            # Distance (X) determines the base color intensity (closer = brighter)
            # Height (Z) determines the hue (lower = blue, higher = red)
            
            # Normalize distance to 0-1 range (closer = 1, farther = 0 for brightness)
            distance_normalized = 1.0 - ((x - np.min(x)) / x_range)  # Invert so closer is brighter
            
            # Normalize height to 0-255 range for hue mapping
            z_normalized = ((z - np.min(z)) / z_range * 255).astype(np.uint8)
            
            # Create colors based on height (hue) and distance (brightness)
            colors = []
            for i in range(len(points)):
                # Get base color from height using rainbow colormap
                base_color = cv2.applyColorMap(np.array([[z_normalized[i]]], dtype=np.uint8), cv2.COLORMAP_RAINBOW)[0, 0]
                
                # Adjust brightness based on distance (closer objects are brighter)
                brightness_factor = 0.3 + 0.7 * distance_normalized[i]  # Range from 0.3 to 1.0
                adjusted_color = (base_color * brightness_factor).astype(np.uint8)
                colors.append(adjusted_color)
            
            colors = np.array(colors)
            
        elif z_range > 0.01:  # Only height variation
            # Use rainbow colormap for height visualization (blue=low, red=high)
            z_normalized = ((z - np.min(z)) / z_range * 255).astype(np.uint8)
            colors = cv2.applyColorMap(z_normalized, cv2.COLORMAP_RAINBOW)
            colors = colors.reshape(-1, 3)
            
        elif x_range > 0.01:  # Only distance variation
            # Use grayscale for distance (closer = brighter)
            distance_normalized = 1.0 - ((x - np.min(x)) / x_range)  # Closer = brighter
            intensity = (distance_normalized * 255).astype(np.uint8)
            colors = np.column_stack([intensity, intensity, intensity])  # Grayscale
            
        else:
            # If all points at similar height and distance, use green
            colors = np.full((len(points), 3), [0, 128, 0], dtype=np.uint8)
        
        # Create occupancy grid for better visualization
        occupancy_grid = np.zeros((self.bev_height, self.bev_width), dtype=np.float32)
        height_grid = np.full((self.bev_height, self.bev_width), np.nan, dtype=np.float32)
        
        # Accumulate points in grid cells
        for i, (px, py) in enumerate(zip(pixel_x, pixel_y)):
            occupancy_grid[py, px] += 1
            if np.isnan(height_grid[py, px]) or z[i] > height_grid[py, px]:
                height_grid[py, px] = z[i]
                # Set color based on highest point in each cell
                bev_image[py, px] = colors[i]
        
        # Apply Gaussian blur for smoother visualization
        bev_image = cv2.GaussianBlur(bev_image, (3, 3), 0)
        
        # Add grid lines for better reference
        self.add_grid_lines(bev_image)
        
        # Add robot position indicator (robot is at origin of coordinate system)
        # Robot is at X=0 (bottom of image), Y=0 (center of image)
        robot_x = self.bev_width // 2  # Center horizontally (Y=0)
        robot_y = self.bev_height - 1  # Bottom of image (X=0, origin)
        cv2.circle(bev_image, (robot_x, robot_y), 8, (0, 0, 255), -1)  # Red circle for robot
        cv2.putText(bev_image, 'ROBOT', (robot_x - 25, robot_y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Add coordinate system reference frame (right-hand coordinate system)
        # X-forward (red), Y-left (green), Z-up (blue) - but Z is represented as a circle since it's out of plane
        self.add_coordinate_frame(bev_image, robot_x, robot_y)
        
        self.get_logger().debug("BEV image generation complete")
        return bev_image
    
    def add_grid_lines(self, image):
        """Add grid lines to BEV image for better spatial reference"""
        # Calculate the actual scale: meters per pixel
        meters_per_pixel_x = self.bev_x_range / self.bev_height  # X (forward) maps to image height
        meters_per_pixel_y = self.bev_y_range / self.bev_width   # Y (left-right) maps to image width
        
        self.get_logger().debug(f"BEV Scale - X direction: {meters_per_pixel_x:.3f} m/pixel, Y direction: {meters_per_pixel_y:.3f} m/pixel")
        
        # Add grid lines every 0.25 meters (25cm)
        grid_spacing = 0.25  # 25cm grid spacing
        
        # Add horizontal lines for X direction (distance markers every 25cm)
        for grid_step in np.arange(grid_spacing, self.bev_x_range + grid_spacing, grid_spacing):
            # X=grid_step corresponds to pixel_y position
            pixel_y = int(self.bev_height - 1 - (grid_step / self.bev_x_range * (self.bev_height - 1)))
            if 0 <= pixel_y < self.bev_height:
                # Use different line styles for different intervals
                if abs(grid_step % 1.0) < 0.01:  # Every 1 meter - thicker line
                    cv2.line(image, (0, pixel_y), (self.bev_width, pixel_y), (150, 150, 150), 2)
                    # Add distance labels for every meter
                    cv2.putText(image, f'{grid_step:.0f}m', (self.bev_width - 40, pixel_y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                else:  # Every 25cm - thinner line
                    cv2.line(image, (0, pixel_y), (self.bev_width, pixel_y), (220, 220, 220), 1)
        
        # Add vertical lines for Y direction (left-right markers every 25cm)
        for grid_step in np.arange(-self.bev_y_range/2, self.bev_y_range/2 + grid_spacing, grid_spacing):
            if abs(grid_step) < 0.01:  # Skip center line for now
                continue
            # Y=grid_step corresponds to pixel_x position
            pixel_x = int((grid_step + self.bev_y_range/2) / self.bev_y_range * (self.bev_width - 1))
            if 0 <= pixel_x < self.bev_width:
                # Use different line styles for different intervals
                if abs(grid_step % 0.5) < 0.01:  # Every 0.5 meter - thicker line
                    cv2.line(image, (pixel_x, 0), (pixel_x, self.bev_height), (150, 150, 150), 2)
                    # Add distance labels for every 0.5m
                    if abs(grid_step) > 0.01:  # Don't label the center
                        label = f'{grid_step:+.1f}m'
                        cv2.putText(image, label, (pixel_x - 15, 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
                else:  # Every 25cm - thinner line
                    cv2.line(image, (pixel_x, 0), (pixel_x, self.bev_height), (220, 220, 220), 1)
        
        # Center lines in different color with labels
        center_x = self.bev_width // 2   # Y=0 line (left-right center)
        origin_y = self.bev_height - 1   # X=0 line (robot position)
        
        # Y=0 center line (left-right)
        cv2.line(image, (center_x, 0), (center_x, self.bev_height), (100, 100, 100), 3)
        cv2.putText(image, 'Y=0', (center_x + 5, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # X=0 line (robot position)
        cv2.line(image, (0, origin_y), (self.bev_width, origin_y), (100, 100, 100), 3)
        cv2.putText(image, 'X=0 (Robot)', (10, origin_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Add scale information
        cv2.putText(image, f'Scale: {self.bev_x_range}m x {self.bev_y_range}m (25cm grid)', 
                   (self.bev_width - 200, self.bev_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    def add_coordinate_frame(self, image, center_x, center_y):
        """Add coordinate system reference frame to BEV image
        Right-hand coordinate system: X-forward (red), Y-left (green), Z-up (blue)
        """
        # Arrow length in pixels
        arrow_length = 40
        arrow_thickness = 3
        
        # X-axis (forward direction) - RED arrow pointing up in image (forward in robot frame)
        x_end_x = center_x
        x_end_y = center_y - arrow_length  # Up in image = forward in robot frame
        cv2.arrowedLine(image, (center_x, center_y), (x_end_x, x_end_y), 
                       (0, 0, 255), arrow_thickness, tipLength=0.3)  # Red for X
        cv2.putText(image, 'X', (x_end_x - 5, x_end_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Y-axis (left direction) - GREEN arrow pointing left in image (left in robot frame)
        y_end_x = center_x - arrow_length  # Left in image = left in robot frame
        y_end_y = center_y
        cv2.arrowedLine(image, (center_x, center_y), (y_end_x, y_end_y), 
                       (0, 255, 0), arrow_thickness, tipLength=0.3)  # Green for Y
        cv2.putText(image, 'Y', (y_end_x - 15, y_end_y + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Z-axis (up direction) - BLUE circle since Z is out of the BEV plane
        z_radius = 12
        cv2.circle(image, (center_x + arrow_length//2, center_y - arrow_length//2), 
                  z_radius, (255, 0, 0), 2)  # Blue circle for Z
        cv2.putText(image, 'Z', (center_x + arrow_length//2 - 5, center_y - arrow_length//2 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Add coordinate system legend in corner
        legend_x, legend_y = 20, 30
        cv2.putText(image, 'Robot Coordinate System:', (legend_x, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(image, 'X-Forward (Red, 0-4m)', (legend_x, legend_y + 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(image, 'Y-Left (Green, ±1.5m)', (legend_x, legend_y + 33), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(image, 'Z-Up (Blue, ±2m)', (legend_x, legend_y + 48), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Add color encoding explanation
        color_legend_x, color_legend_y = 20, 100
        cv2.putText(image, 'Color Encoding:', (color_legend_x, color_legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(image, 'Hue = Height (Blue=Low, Red=High)', (color_legend_x, color_legend_y + 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(image, 'Brightness = Distance (Bright=Close)', (color_legend_x, color_legend_y + 33), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(image, 'Dark = Far, Bright = Near', (color_legend_x, color_legend_y + 48), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        
        # Add edge labels for better orientation understanding with distance markers
        height, width = image.shape[:2]
        
        # Top edge - FORWARD direction with distance marker
        cv2.putText(image, f'FORWARD ({self.bev_x_range:.0f}m)', (width//2 - 50, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Bottom edge - ROBOT position (X=0)
        cv2.putText(image, 'ROBOT (X=0)', (width//2 - 50, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Left edge - LEFT direction with distance
        left_distance = self.bev_y_range / 2
        cv2.putText(image, f'LEFT', (5, height//2 - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f'(-{left_distance:.0f}m)', (5, height//2 + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Right edge - RIGHT direction with distance
        right_distance = self.bev_y_range / 2
        cv2.putText(image, f'RIGHT', (width - 55, height//2 - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f'(+{right_distance:.0f}m)', (width - 55, height//2 + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

def main(args=None):
    rclpy.init(args=args)
    node = D435IBEVProjection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

