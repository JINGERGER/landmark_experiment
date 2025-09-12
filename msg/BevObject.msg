# BEV Object message - represents a single detected object in BEV coordinate system
string class_name               # Object class name (e.g., "person", "car", "bottle")
float32 confidence             # Detection confidence [0.0, 1.0]

geometry_msgs/Point camera_3d_position    # 3D position in camera coordinate system (meters)
geometry_msgs/Point bev_position          # Position in BEV coordinate system (x: forward, y: left, z: unused)

geometry_msgs/Point bev_pixel_position    # Pixel position in BEV image (x, y pixels)
float32 equivalent_diameter_m             # Object equivalent diameter in meters
float32 depth_m                          # Object depth/distance from camera in meters
int32 area_pixels                        # Object area in pixels
