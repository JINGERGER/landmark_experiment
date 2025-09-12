# BEV Objects Array message - contains all detected objects in BEV coordinate system
std_msgs/Header header          # Timestamp and frame_id (should be "camera_bev")
BevObject[] objects             # Array of detected objects
int32 total_objects            # Total number of detected objects
geometry_msgs/Point bev_range   # BEV detection range (x: forward_max, y: lateral_max)
