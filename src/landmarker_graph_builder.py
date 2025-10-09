#!/usr/bin/env python3
"""
Landmark Graph Builder Node
订阅BEV物体检测结果和里程计数据，构建全局地标图

Subscribes to: 
    /yolo11/bev/objects - BEV物体检测结果 (BevObjectArray)
    /odom - 里程计数据 (Odometry)
    /camera/camera/color/image_raw - 彩色图像 (Image)
    
Publishes to:
    /landmark_graph/markers - 地标图可视化 (MarkerArray)
    /landmark_graph/data - 地标图数据 (String - JSON格式)
"""

import rclpy
from rclpy.node import Node
from bev.msg import BevObjectArray, BevObject
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point, Quaternion, Pose, Vector3
from sensor_msgs.msg import Image
import json
import numpy as np
import math
from datetime import datetime
import time
import cv2
from cv_bridge import CvBridge

class LandmarkGraphBuilder(Node):
    def __init__(self):
        super().__init__('landmark_graph_builder')
        
        # Declare parameters
        self.declare_parameter('bev_objects_topic', '/yolo11/bev/objects')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('landmark_markers_topic', '/landmark_graph/markers')
        self.declare_parameter('landmark_data_topic', '/landmark_graph/data')
        self.declare_parameter('association_distance_threshold', 1.0)  # 物体关联距离阈值(米)
        self.declare_parameter('landmark_confidence_threshold', 0.6)   # 地标置信度阈值
        self.declare_parameter('min_observations', 2)                 # 最小观测次数
        self.declare_parameter('max_landmark_age', 30.0)              # 地标最大存活时间(秒)
        self.declare_parameter('publish_rate', 2.0)                   # 发布频率(Hz)
        self.declare_parameter('enable_vlm_association', False)       # 启用VLM关联
        self.declare_parameter('vlm_model_name', 'gpt-4-vision-preview')  # VLM模型名称
        self.declare_parameter('vlm_confidence_threshold', 0.7)       # VLM关联置信度阈值
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        
        # Get parameters
        self.bev_objects_topic = self.get_parameter('bev_objects_topic').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.landmark_markers_topic = self.get_parameter('landmark_markers_topic').get_parameter_value().string_value
        self.landmark_data_topic = self.get_parameter('landmark_data_topic').get_parameter_value().string_value
        self.association_threshold = self.get_parameter('association_distance_threshold').get_parameter_value().double_value
        self.confidence_threshold = self.get_parameter('landmark_confidence_threshold').get_parameter_value().double_value
        self.min_observations = self.get_parameter('min_observations').get_parameter_value().integer_value
        self.max_age = self.get_parameter('max_landmark_age').get_parameter_value().double_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.vlm_enabled = self.get_parameter('enable_vlm_association').get_parameter_value().bool_value
        self.vlm_model = self.get_parameter('vlm_model_name').get_parameter_value().string_value
        self.vlm_confidence_threshold = self.get_parameter('vlm_confidence_threshold').get_parameter_value().double_value
        # 订阅彩色图像话题（用于VLM多模态）
        self.color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
         
        # 地标图数据结构
        self.landmarks = {}  # landmark_id: LandmarkData
        self.next_landmark_id = 1
        
        # 机器人当前位姿
        self.robot_pose = None
        self.robot_orientation = None
        
        # 最新的BEV检测数据
        self.latest_bev_objects = None
        self.last_bev_timestamp = None
        
        # 彩色图像处理
        self.bridge = CvBridge()
        self.latest_color_image_path = None
        
        # Create subscribers
        self.bev_sub = self.create_subscription(
            BevObjectArray,
            self.bev_objects_topic,
            self.bev_objects_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )
        
        self.color_sub = self.create_subscription(
            Image,
            self.color_topic,
            self.color_image_callback,
            5
        )
        
        # Create publishers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            self.landmark_markers_topic,
            10
        )
        
        self.data_pub = self.create_publisher(
            String,
            self.landmark_data_topic,
            10
        )
        
        # Timer for periodic publishing
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_landmarks)
        
        self.get_logger().info(f"🚀 Landmark Graph Builder initialized")
        self.get_logger().info(f"📥 Subscribing to BEV objects: {self.bev_objects_topic}")
        self.get_logger().info(f"📥 Subscribing to odometry: {self.odom_topic}")
        self.get_logger().info(f"📥 Subscribing to color images: {self.color_topic}")
        self.get_logger().info(f"📤 Publishing markers to: {self.landmark_markers_topic}")
        self.get_logger().info(f"📤 Publishing data to: {self.landmark_data_topic}")
        self.get_logger().info(f"🔧 Association threshold: {self.association_threshold}m")
        self.get_logger().info(f"🔧 Confidence threshold: {self.confidence_threshold}")
        self.get_logger().info(f"🔧 Min observations: {self.min_observations}")

    def odom_callback(self, msg):
        """处理里程计数据，更新机器人位姿"""
        self.robot_pose = msg.pose.pose.position
        self.robot_orientation = msg.pose.pose.orientation
        
        # 处理最新的BEV检测数据（如果有的话）
        if self.latest_bev_objects is not None:
            self.process_bev_objects_with_odom(self.latest_bev_objects)
            self.latest_bev_objects = None

    def bev_objects_callback(self, msg):
        """处理BEV物体检测数据"""
        self.latest_bev_objects = msg
        self.last_bev_timestamp = msg.header.stamp
        
        # 如果已有机器人位姿，立即处理
        if self.robot_pose is not None:
            self.process_bev_objects_with_odom(msg)
            self.latest_bev_objects = None

    def color_image_callback(self, msg):
        """保存最近一帧彩色图像到本地，供VLM使用"""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            save_path = f"/tmp/vlm_latest_color.jpg"
            cv2.imwrite(save_path, cv_img)
            self.latest_color_image_path = save_path
        except Exception as e:
            self.get_logger().warn(f"Failed to save color image: {e}")

    def process_bev_objects_with_odom(self, bev_msg):
        """将BEV检测结果与里程计数据关联，更新地标图"""
        if self.robot_pose is None:
            return
        
        current_time = time.time()
        
        # 转换机器人朝向为欧拉角
        robot_yaw = self.quaternion_to_yaw(self.robot_orientation)
        
        # 处理每个检测到的物体
        for bev_obj in bev_msg.objects:
            if bev_obj.confidence < self.confidence_threshold:
                continue
                
            # 将BEV坐标转换为全局坐标
            global_pos = self.bev_to_global(
                bev_obj.bev_position, 
                self.robot_pose, 
                robot_yaw
            )
            
            # 查找或创建地标
            landmark_id = self.associate_or_create_landmark(
                global_pos, 
                bev_obj.class_name, 
                bev_obj.confidence,
                bev_obj.equivalent_diameter_m,
                current_time,
                bev_obj  # 传递完整的BEV对象给VLM使用
            )
            
            self.get_logger().debug(
                f"🎯 Object {bev_obj.class_name} at global ({global_pos[0]:.2f}, {global_pos[1]:.2f}) "
                f"-> landmark {landmark_id}"
            )
        
        # 清理过期地标
        self.cleanup_old_landmarks(current_time)
        
        self.get_logger().info(f"🗺️ Landmark graph updated: {len(self.landmarks)} landmarks")

    def bev_to_global(self, bev_position, robot_position, robot_yaw):
        """将BEV相对坐标转换为全局坐标"""
        # BEV坐标系：x-前进，y-左侧
        bev_x = bev_position.x  # 前进距离
        bev_y = bev_position.y  # 左侧距离
        
        # 旋转到全局坐标系
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        
        # 全局坐标
        global_x = robot_position.x + cos_yaw * bev_x - sin_yaw * bev_y
        global_y = robot_position.y + sin_yaw * bev_x + cos_yaw * bev_y
        
        return np.array([global_x, global_y])

    def associate_or_create_landmark(self, global_pos, class_name, confidence, size, timestamp, bev_obj=None):
        """关联或创建新地标 - 支持VLM增强"""
        # 查找距离最近的同类地标
        closest_landmark_id = None
        min_distance = float('inf')
        
        candidates = []  # 候选地标列表，用于VLM验证
        
        for lid, landmark in self.landmarks.items():
            if landmark['class_name'] == class_name:
                distance = np.linalg.norm(global_pos - landmark['position'])
                if distance < min_distance:
                    min_distance = distance
                    closest_landmark_id = lid
                
                # 收集所有在阈值范围内的候选者
                if distance < self.association_threshold * 2:  # 扩大搜索范围
                    candidates.append({
                        'id': lid,
                        'distance': distance,
                        'landmark': landmark
                    })
        
        # 如果有多个候选者且有图像数据，使用VLM验证
        if len(candidates) > 1 and bev_obj is not None and hasattr(self, 'vlm_enabled') and self.vlm_enabled:
            best_match = self.vlm_associate_landmark(bev_obj, candidates, global_pos)
            if best_match is not None:
                self.update_landmark(best_match['id'], global_pos, confidence, size, timestamp, bev_obj)
                return best_match['id']
        
        # 传统方法：如果找到足够近的地标，更新它
        if closest_landmark_id is not None and min_distance < self.association_threshold:
            self.update_landmark(closest_landmark_id, global_pos, confidence, size, timestamp, bev_obj)
            return closest_landmark_id
        else:
            # 创建新地标
            return self.create_new_landmark(global_pos, class_name, confidence, size, timestamp, bev_obj)

    def create_new_landmark(self, global_pos, class_name, confidence, size, timestamp):
        """创建新地标"""
        landmark_id = self.next_landmark_id
        self.next_landmark_id += 1
        
        self.landmarks[landmark_id] = {
            'id': landmark_id,
            'class_name': class_name,
            'position': global_pos.copy(),
            'confidence': confidence,
            'size': size,
            'observations': 1,
            'created_time': timestamp,
            'last_seen': timestamp,
            'position_history': [global_pos.copy()],
            'confidence_history': [confidence]
        }
        
        self.get_logger().info(f"✨ Created new landmark {landmark_id}: {class_name} at ({global_pos[0]:.2f}, {global_pos[1]:.2f})")
        return landmark_id

    def update_landmark(self, landmark_id, global_pos, confidence, size, timestamp, bev_obj=None):
        """更新现有地标"""
        landmark = self.landmarks[landmark_id]
        
        # 增加观测次数
        landmark['observations'] += 1
        landmark['last_seen'] = timestamp
        
        # 更新位置（使用加权平均）
        alpha = 0.3  # 新观测的权重
        landmark['position'] = (1 - alpha) * landmark['position'] + alpha * global_pos
        
        # 更新置信度（使用最大值或加权平均）
        landmark['confidence'] = max(landmark['confidence'], confidence)
        
        # 更新尺寸
        landmark['size'] = (landmark['size'] + size) / 2
        
        # 记录历史
        landmark['position_history'].append(global_pos.copy())
        landmark['confidence_history'].append(confidence)
        
        # 限制历史记录长度
        if len(landmark['position_history']) > 10:
            landmark['position_history'] = landmark['position_history'][-10:]
            landmark['confidence_history'] = landmark['confidence_history'][-10:]

    def cleanup_old_landmarks(self, current_time):
        """清理过期地标"""
        landmarks_to_remove = []
        
        for lid, landmark in self.landmarks.items():
            age = current_time - landmark['last_seen']
            if age > self.max_age:
                landmarks_to_remove.append(lid)
        
        for lid in landmarks_to_remove:
            removed_landmark = self.landmarks.pop(lid)
            self.get_logger().info(
                f"🗑️ Removed expired landmark {lid}: {removed_landmark['class_name']} "
                f"(age: {current_time - removed_landmark['last_seen']:.1f}s)"
            )

    def extract_vlm_features(self, bev_obj):
        """从BEV对象中提取VLM可用的特征描述"""
        if not bev_obj:
            return None
            
        features = {
            'description': f"A {bev_obj.class_name} object",
            'size_category': self.categorize_size(bev_obj.equivalent_diameter_m),
            'position_description': self.describe_position(bev_obj.bev_position),
            'confidence_level': self.categorize_confidence(bev_obj.confidence),
            'timestamp': time.time()
        }
        
        return features
    
    def merge_vlm_features(self, old_features, new_features):
        """合并VLM特征"""
        if not old_features:
            return new_features
            
        if not new_features:
            return old_features
        
        # 简单的特征合并策略
        merged = old_features.copy()
        
        # 更新时间戳
        merged['timestamp'] = new_features['timestamp']
        
        # 如果置信度提高，更新描述
        if new_features.get('confidence_level', 'low') > merged.get('confidence_level', 'low'):
            merged.update(new_features)
            
        return merged
    
    def categorize_size(self, diameter_m):
        """将尺寸分类为可理解的描述"""
        if diameter_m < 0.3:
            return "small"
        elif diameter_m < 1.0:
            return "medium"
        elif diameter_m < 2.0:
            return "large"
        else:
            return "very_large"
    
    def describe_position(self, bev_position):
        """描述BEV位置"""
        x, y = bev_position.x, bev_position.y
        
        # 距离描述
        distance = math.sqrt(x*x + y*y)
        if distance < 1.0:
            dist_desc = "very close"
        elif distance < 2.0:
            dist_desc = "close"
        elif distance < 4.0:
            dist_desc = "moderate distance"
        else:
            dist_desc = "far"
            
        # 方向描述
        if abs(y) < 0.5:  # 基本直前方
            dir_desc = "directly ahead"
        elif y > 0:
            dir_desc = "to the left"
        else:
            dir_desc = "to the right"
            
        return f"{dist_desc}, {dir_desc}"
    
    def categorize_confidence(self, confidence):
        """将置信度分类"""
        if confidence < 0.5:
            return "low"
        elif confidence < 0.7:
            return "medium"
        elif confidence < 0.9:
            return "high"
        else:
            return "very_high"

    def publish_landmarks(self):
        """发布地标图数据和可视化"""
        if not self.landmarks:
            return
        
        # 发布可视化标记
        self.publish_landmark_markers()
        
        # 发布地标数据
        self.publish_landmark_data()

    def publish_landmark_markers(self):
        """发布地标可视化标记"""
        marker_array = MarkerArray()
        current_time = self.get_clock().now().to_msg()
        
        for lid, landmark in self.landmarks.items():
            # 只发布有足够观测次数的地标
            if landmark['observations'] < self.min_observations:
                continue
                
            # 创建标记
            marker = Marker()
            marker.header.frame_id = "odom"  # 使用里程计坐标系
            marker.header.stamp = current_time
            marker.ns = "landmarks"
            marker.id = lid
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # 设置位置
            marker.pose.position.x = landmark['position'][0]
            marker.pose.position.y = landmark['position'][1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # 设置尺寸（基于物体大小）
            size = max(0.2, min(landmark['size'], 2.0))  # 限制在0.2-2.0米之间
            marker.scale.x = size
            marker.scale.y = size
            marker.scale.z = 0.5
            
            # 设置颜色（基于物体类别和置信度）
            color = self.get_landmark_color(landmark['class_name'])
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = min(1.0, landmark['confidence'] + 0.3)
            
            marker.lifetime.sec = int(2.0 / self.publish_rate)  # 标记生存时间
            
            marker_array.markers.append(marker)
            
            # 创建文本标签
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "landmark_labels"
            text_marker.id = lid
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = landmark['position'][0]
            text_marker.pose.position.y = landmark['position'][1]
            text_marker.pose.position.z = 1.0
            text_marker.pose.orientation.w = 1.0
            
            text_marker.scale.z = 0.3
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.text = f"{landmark['class_name']}\nID:{lid}\nObs:{landmark['observations']}"
            text_marker.lifetime.sec = int(2.0 / self.publish_rate)
            
            marker_array.markers.append(text_marker)
        
        self.marker_pub.publish(marker_array)

    def publish_landmark_data(self):
        """发布地标图数据"""
        landmark_data = {
            'timestamp': time.time(),
            'total_landmarks': len(self.landmarks),
            'active_landmarks': sum(1 for l in self.landmarks.values() if l['observations'] >= self.min_observations),
            'robot_position': {
                'x': float(self.robot_pose.x) if self.robot_pose else 0.0,
                'y': float(self.robot_pose.y) if self.robot_pose else 0.0
            },
            'landmarks': []
        }
        
        for lid, landmark in self.landmarks.items():
            if landmark['observations'] >= self.min_observations:
                landmark_info = {
                    'id': lid,
                    'class_name': landmark['class_name'],
                    'position': {
                        'x': float(landmark['position'][0]),
                        'y': float(landmark['position'][1])
                    },
                    'confidence': float(landmark['confidence']),
                    'size': float(landmark['size']),
                    'observations': landmark['observations'],
                    'age': time.time() - landmark['created_time']
                }
                landmark_data['landmarks'].append(landmark_info)
        
        # 发布JSON数据
        data_msg = String()
        data_msg.data = json.dumps(landmark_data, indent=2)
        self.data_pub.publish(data_msg)

    def get_landmark_color(self, class_name):
        """根据物体类别获取颜色"""
        colors = {
            'person': [0.0, 1.0, 0.0],      # 绿色
            'car': [1.0, 0.0, 0.0],         # 红色
            'truck': [1.0, 0.5, 0.0],       # 橙色
            'bus': [1.0, 1.0, 0.0],         # 黄色
            'bicycle': [0.0, 0.0, 1.0],     # 蓝色
            'motorcycle': [1.0, 0.0, 1.0],  # 紫色
            'bottle': [0.0, 1.0, 1.0],      # 青色
            'chair': [0.5, 0.5, 0.5],       # 灰色
            'table': [0.8, 0.4, 0.2],       # 棕色
        }
        return colors.get(class_name, [0.5, 0.5, 0.5])  # 默认灰色

    def quaternion_to_yaw(self, quaternion):
        """将四元数转换为偏航角"""
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def vlm_associate_landmark(self, bev_obj, candidates, global_pos):
        """使用VLM进行地标关联验证"""
        try:
            # 构建VLM查询
            query = self.build_vlm_query(bev_obj, candidates)
            
            # 调用VLM API（这里是伪代码，需要根据具体VLM实现）
            vlm_response = self.query_vlm(query)
            
            # 解析VLM响应
            best_match = self.parse_vlm_response(vlm_response, candidates)
            
            if best_match and best_match.get('confidence', 0) > self.vlm_confidence_threshold:
                self.get_logger().info(
                    f"🤖 VLM matched {bev_obj.class_name} with landmark {best_match['id']} "
                    f"(confidence: {best_match['confidence']:.2f})"
                )
                return best_match
            
        except Exception as e:
            self.get_logger().warn(f"⚠️ VLM association failed: {e}, falling back to traditional method")
        
        return None
    
    def build_vlm_query(self, bev_obj, candidates):
        """构建VLM查询"""
        # 构建描述当前检测物体的文本
        current_description = {
            'class': bev_obj.class_name,
            'confidence': float(bev_obj.confidence),
            'size': f"{bev_obj.equivalent_diameter_m:.2f}m diameter",
            'position': f"({bev_obj.bev_position.x:.2f}, {bev_obj.bev_position.y:.2f})",
        }
        
        # 构建候选地标的描述
        candidate_descriptions = []
        for candidate in candidates:
            landmark = candidate['landmark']
            desc = {
                'id': candidate['id'],
                'class': landmark['class_name'],
                'confidence': landmark['confidence'],
                'size': f"{landmark['size']:.2f}m",
                'observations': landmark['observations'],
                'age': f"{time.time() - landmark['created_time']:.1f}s",
                'distance': f"{candidate['distance']:.2f}m"
            }
            candidate_descriptions.append(desc)
        
        # 构建VLM提示
        prompt = f"""
        I am performing landmark association for robot navigation. A new {current_description['class']} object has been detected:
        - Confidence: {current_description['confidence']:.2f}
        - Size: {current_description['size']}
        - BEV Position: {current_description['position']}
        
        Existing landmark candidates are:
        """
        
        for i, desc in enumerate(candidate_descriptions):
            prompt += f"""
        Candidate {i+1} (ID: {desc['id']}):
        - Class: {desc['class']}
        - Confidence: {desc['confidence']:.2f}
        - Size: {desc['size']}
        - Observations: {desc['observations']}
        - Age: {desc['age']}
        - Distance: {desc['distance']}
        """
        
        prompt += """
        Please determine which existing landmark the newly detected object most likely corresponds to based on:
        1. Object class must match
        2. Size similarity
        3. Spatial proximity (closer distance means more likely to be the same object)
        4. Temporal continuity (recently observed landmarks are more likely to be the same)
        
        Please respond in JSON format:
        {
            "best_match_id": <landmark ID or null>,
            "confidence": <confidence score 0.0-1.0>,
            "reasoning": "<brief reasoning process>"
        }
        
        If no suitable match exists, return best_match_id as null.
        """
        
        query = {
            'prompt': prompt,
            'current_object': current_description,
            'candidates': candidate_descriptions
        }
        # 如果有最新彩色图像，加入image_url字段（本地文件路径）
        if self.latest_color_image_path:
            query['image_url'] = f"file://{self.latest_color_image_path}"
        return query
    
    def query_vlm(self, query):
        """查询VLM API"""
        # 这里是VLM API调用的接口
        # 可以集成OpenAI GPT-4V, Google Gemini Vision, Qwen, 或本地部署的VLM
        if self.vlm_model.startswith('gpt-4'):
            return self.query_openai_vlm(query)
        elif self.vlm_model.startswith('gemini'):
            return self.query_gemini_vlm(query)
        elif self.vlm_model.startswith('llava'):
            return self.query_local_vlm(query)
        elif self.vlm_model.startswith('qwen'):
            return self.query_qwen_vlm(query)
        else:
            raise ValueError(f"Unsupported VLM model: {self.vlm_model}")
    
    def query_openai_vlm(self, query):
        """调用OpenAI GPT-4V API"""
        # 伪代码 - 需要安装 openai 包
        try:
            import openai
            
            response = openai.ChatCompletion.create(
                model=self.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": query['prompt']
                    }
                ],
                max_tokens=500,
                temperature=0.1  # 低温度以获得更一致的结果
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            self.get_logger().warn("OpenAI package not installed, VLM disabled")
            return None
        except Exception as e:
            self.get_logger().error(f"OpenAI API error: {e}")
            return None
    
    def query_gemini_vlm(self, query):
        """调用Google Gemini Vision API"""
        # 伪代码 - 需要安装 google-generativeai 包
        try:
            import google.generativeai as genai
            
            model = genai.GenerativeModel(self.vlm_model)
            response = model.generate_content(query['prompt'])
            
            return response.text
            
        except ImportError:
            self.get_logger().warn("Google GenerativeAI package not installed, VLM disabled")
            return None
        except Exception as e:
            self.get_logger().error(f"Gemini API error: {e}")
            return None
    
    def query_local_vlm(self, query):
        """调用本地VLM模型（如LLaVA）"""
        # 伪代码 - 可以通过HTTP API调用本地部署的模型
        try:
            import requests
            
            # 假设本地VLM服务运行在localhost:8000
            response = requests.post(
                "http://localhost:8000/generate",
                json={
                    "prompt": query['prompt'],
                    "max_tokens": 500,
                    "temperature": 0.1
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()['text']
            else:
                raise Exception(f"Local VLM API returned {response.status_code}")
                
        except ImportError:
            self.get_logger().warn("Requests package not installed, local VLM disabled")
            return None
        except Exception as e:
            self.get_logger().error(f"Local VLM error: {e}")
            return None
    
    def query_qwen_vlm(self, query):
        """调用Qwen VLM API (Aliyun 百炼)，支持多模态（图文）输入"""
        try:
            import os
            from openai import OpenAI
            client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            # 判断是否有图片URL
            image_url = query.get('image_url', None)
            if image_url:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": query['prompt']},
                    ]},
                ]
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query['prompt']},
                ]
            response = client.chat.completions.create(
                model=self.vlm_model,  # e.g., "qwen3-vl-plus"
                messages=messages,
                max_tokens=500,
                temperature=0.1
            )
            return response.choices[0].message.content
        except ImportError:
            self.get_logger().warn("openai package not installed, Qwen VLM disabled")
            return None
        except Exception as e:
            self.get_logger().error(f"Qwen API error: {e}")
            return None

    def parse_vlm_response(self, response_text, candidates):
        """解析VLM响应"""
        if not response_text:
            return None
            
        try:
            import json
            
            # 尝试解析JSON响应
            # 处理可能的markdown格式包装
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            else:
                json_str = response_text.strip()
            
            result = json.loads(json_str)
            
            # 验证响应格式
            if 'best_match_id' in result and 'confidence' in result:
                match_id = result['best_match_id']
                confidence = float(result['confidence'])
                reasoning = result.get('reasoning', 'No reasoning provided')
                
                # 如果有匹配，验证ID是否在候选列表中
                if match_id is not None:
                    for candidate in candidates:
                        if candidate['id'] == match_id:
                            return {
                                'id': match_id,
                                'confidence': confidence,
                                'reasoning': reasoning,
                                'candidate': candidate
                            }
                
                # 没有匹配的情况
                return None
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.get_logger().warn(f"Failed to parse VLM response: {e}")
            self.get_logger().debug(f"Response text: {response_text}")
        
        return None

def main(args=None):
    rclpy.init(args=args)
    node = LandmarkGraphBuilder()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Landmark Graph Builder shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
