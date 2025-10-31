#!/usr/bin/env python3

import math
import json
from collections import deque
from typing import Dict, Tuple, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from bev.msg import BevObjectArray


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
	"""Extract yaw (rotation around Z) from quaternion."""
	# yaw (z-axis rotation)
	siny_cosp = 2.0 * (w * z + x * y)
	cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
	return math.atan2(siny_cosp, cosy_cosp)


def parse_xy(text: str, default: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
	try:
		parts = [p.strip() for p in text.split(",")]
		if len(parts) >= 2:
			return float(parts[0]), float(parts[1])
	except Exception:
		pass
	return default


def hash_color(name: str) -> Tuple[float, float, float]:
	"""Deterministic pseudo-random color in RGB [0,1] for a class name."""
	h = abs(hash(name))
	r = ((h >> 0) & 0xFF) / 255.0
	g = ((h >> 8) & 0xFF) / 255.0
	b = ((h >> 16) & 0xFF) / 255.0
	# ensure not too dark
	s = 0.25
	return (s + (1 - s) * r, s + (1 - s) * g, s + (1 - s) * b)


class LandmakerWeaverNode(Node):
	"""
	Subscribes to BEV detections and Odometry, transforms detections into a global frame,
	clusters them into persistent landmarks, and publishes a simple landmark graph as RViz markers
	and JSON summary.

	Conventions:
	- Input BEV coords: x forward (meters), y left (meters), frame_id ~ "camera_bev"
	- Odometry: pose of base in global_frame (default 'odom')
	- We assume camera frame is rigidly attached to base with small configurable offset
	"""

	def __init__(self):
		super().__init__('landmaker_weaver')

		# Params
		self.declare_parameter('bev_topic', '/yolo11/bev/objects')
		self.declare_parameter('odom_topic', '/odom')
		self.declare_parameter('global_frame', 'odom')
		self.declare_parameter('camera_yaw_offset_deg', 0.0)
		self.declare_parameter('camera_xy_offset_m', '0.0,0.0')  # (x forward, y left)
		self.declare_parameter('match_radius_m', 0.5)
		self.declare_parameter('match_radius_gain_per_meter', 0.05)  # widen radius with distance
		self.declare_parameter('max_node_age_sec', 5.0)               # only match to recently seen nodes
		self.declare_parameter('strict_class_match', True)            # require same class for primary match
		self.declare_parameter('fallback_class_agnostic', True)       # allow fallback ignoring class (less strict)
		self.declare_parameter('fallback_extra_radius_m', 0.3)        # extra radius for fallback
		# Strict association / promotion
		self.declare_parameter('min_hits_to_confirm', 2)
		self.declare_parameter('require_consecutive_hits', False)
		self.declare_parameter('consecutive_window_sec', 1.0)
		# Matching with tentative nodes
		self.declare_parameter('tentative_radius_scale', 0.7)
		self.declare_parameter('ema_alpha', 0.3)
		self.declare_parameter('max_odom_dt_sec', 0.25)
		self.declare_parameter('min_confidence', 0.3)
		self.declare_parameter('publish_markers_topic', '/landgraph/markers')
		self.declare_parameter('publish_graph_topic', '/landgraph/graph')
		self.declare_parameter('node_diameter_m', 0.2)
		self.declare_parameter('label_scale_m', 0.15)
		self.declare_parameter('line_width_m', 0.02)

		self.bev_topic = self.get_parameter('bev_topic').value
		self.odom_topic = self.get_parameter('odom_topic').value
		self.global_frame = self.get_parameter('global_frame').value
		self.cam_yaw_offset = math.radians(self.get_parameter('camera_yaw_offset_deg').value)
		self.cam_offset_xy = parse_xy(self.get_parameter('camera_xy_offset_m').value, (0.0, 0.0))
		self.match_radius = float(self.get_parameter('match_radius_m').value)
		self.match_radius_gain = float(self.get_parameter('match_radius_gain_per_meter').value)
		self.max_node_age = float(self.get_parameter('max_node_age_sec').value)
		self.strict_class = bool(self.get_parameter('strict_class_match').value)
		self.fallback_class_agnostic = bool(self.get_parameter('fallback_class_agnostic').value)
		self.fallback_extra_radius = float(self.get_parameter('fallback_extra_radius_m').value)
		self.min_hits_to_confirm = int(self.get_parameter('min_hits_to_confirm').value)
		self.require_consecutive_hits = bool(self.get_parameter('require_consecutive_hits').value)
		self.consecutive_window = float(self.get_parameter('consecutive_window_sec').value)
		self.tentative_radius_scale = float(self.get_parameter('tentative_radius_scale').value)
		self.ema_alpha = float(self.get_parameter('ema_alpha').value)
		self.max_odom_dt = float(self.get_parameter('max_odom_dt_sec').value)
		self.min_conf = float(self.get_parameter('min_confidence').value)
		self.markers_topic = self.get_parameter('publish_markers_topic').value
		self.graph_topic = self.get_parameter('publish_graph_topic').value
		self.node_diameter = float(self.get_parameter('node_diameter_m').value)
		self.label_scale = float(self.get_parameter('label_scale_m').value)
		self.line_width = float(self.get_parameter('line_width_m').value)

		# State
		self.odom_buffer: deque = deque(maxlen=500)
		self.nodes: Dict[int, Dict] = {}
		self.next_node_id: int = 1

		# Publishers/Subscribers
		self.marker_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)
		self.graph_pub = self.create_publisher(String, self.graph_topic, 10)
		self.bev_sub = self.create_subscription(BevObjectArray, self.bev_topic, self.bev_callback, 50)
		self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 100)

		# Periodic republish markers
		self.timer = self.create_timer(0.5, self.publish_markers)

		self.get_logger().info(
			f"LandmakerWeaver started. bev_topic={self.bev_topic}, odom_topic={self.odom_topic}, frame={self.global_frame}"
		)

	# ---------------- Odometry ----------------
	def odom_callback(self, msg: Odometry):
		p = msg.pose.pose.position
		q = msg.pose.pose.orientation
		yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
		stamp = self._stamp_to_time(msg.header.stamp)
		self.odom_buffer.append((stamp, float(p.x), float(p.y), float(yaw)))

	def _nearest_odom(self, t: float) -> Optional[Tuple[float, float, float]]:
		if not self.odom_buffer:
			return None
		# Find by absolute time difference
		best = None
		best_dt = float('inf')
		for (ts, x, y, yaw) in self.odom_buffer:
			dt = abs(ts - t)
			if dt < best_dt:
				best_dt = dt
				best = (x, y, yaw)
		if best is None or best_dt > self.max_odom_dt:
			return None
		return best

	def _stamp_to_time(self, stamp) -> float:
		# rclpy builtin_interfaces/Time: sec, nanosec
		return float(stamp.sec) + float(stamp.nanosec) * 1e-9

	# --------------- Transform ----------------
	def bev_local_to_global(self, bev_x: float, bev_y: float, odom_pose: Tuple[float, float, float]) -> Tuple[float, float]:
		"""
		Transform BEV local (x forward, y left) to global frame using odom pose
		and static camera offset (x forward, y left) and yaw offset.
		"""
		ox, oy, oyaw = odom_pose

		# First apply camera offset in robot frame (still x forward, y left)
		cx, cy = self.cam_offset_xy
		# rotate camera offset by robot yaw to global
		cos_y = math.cos(oyaw)
		sin_y = math.sin(oyaw)
		cam_off_x = cos_y * cx - sin_y * cy
		cam_off_y = sin_y * cx + cos_y * cy

		# Apply camera yaw offset to local BEV measurement
		yaw = oyaw + self.cam_yaw_offset
		c = math.cos(yaw)
		s = math.sin(yaw)
		gx = ox + cam_off_x + c * bev_x - s * bev_y
		gy = oy + cam_off_y + s * bev_x + c * bev_y
		return gx, gy

	# --------------- Association -------------
	def _dynamic_radius(self, dist_to_robot: float, extra: float = 0.0) -> float:
		return self.match_radius + self.match_radius_gain * max(0.0, dist_to_robot) + extra

	def _update_node(self, nid: int, x: float, y: float, t: float):
		n = self.nodes[nid]
		a = self.ema_alpha
		n['x'] = a * x + (1.0 - a) * n['x']
		n['y'] = a * y + (1.0 - a) * n['y']
		n['count'] += 1
		# hit counters for confirmation
		n['hits'] = n.get('hits', 0) + 1
		# consecutive logic
		dt = t - n.get('last', t)
		if dt <= self.consecutive_window:
			n['consecutive'] = n.get('consecutive', 0) + 1
		else:
			n['consecutive'] = 1
		# promote to confirmed if thresholds met
		if self.require_consecutive_hits:
			if n['consecutive'] >= self.min_hits_to_confirm:
				n['confirmed'] = True
		else:
			if n['hits'] >= self.min_hits_to_confirm:
				n['confirmed'] = True
		
		n['last'] = t
		hist = n['history']
		hist.append((n['x'], n['y'], t))
		if len(hist) > 50:
			del hist[0]

	def _create_node(self, cls: str, x: float, y: float, t: float) -> int:
		nid = self.next_node_id
		self.next_node_id += 1
		color = hash_color(cls)
		self.nodes[nid] = {
			'id': nid,
			'class': cls,
			'x': x,
			'y': y,
			'first': t,
			'last': t,
			'count': 1,
			'hits': 1,
			'consecutive': 1,
			'confirmed': False,
			'history': [(x, y, t)],
			'color': color,
		}
		return nid

	# ----------------- Callbacks -------------
	def bev_callback(self, msg: BevObjectArray):
		if msg.total_objects <= 0 or len(msg.objects) == 0:
			return

		t = self._stamp_to_time(msg.header.stamp)
		od = self._nearest_odom(t)
		if od is None:
			self.get_logger().warn('No odometry close to BEV timestamp; skipping this frame')
			return

		# Prepare detections transformed to global
		ox, oy, oyaw = od
		dets = []  # list of dicts: idx, cls, gx, gy, dist
		for i, obj in enumerate(msg.objects):
			try:
				conf = float(obj.confidence)
			except Exception:
				conf = 1.0
			if conf < self.min_conf:
				continue
			cls = obj.class_name if obj.class_name else 'unknown'
			bx = float(obj.bev_position.x)
			by = float(obj.bev_position.y)
			gx, gy = self.bev_local_to_global(bx, by, od)
			dist = math.hypot(gx - ox, gy - oy)
			dets.append({'idx': i, 'class': cls, 'gx': gx, 'gy': gy, 'dist': dist})

		if not dets:
			return

		# Active nodes (recently seen) - only confirmed nodes for stricter matching
		active_nodes = {nid: n for nid, n in self.nodes.items() if n.get('confirmed', False) and (t - n['last']) <= self.max_node_age}

		# Greedy one-to-one assignment by ascending distance
		used_nodes = set()
		assigned = {}  # det_index -> node_id

		def candidates(dets_list, nodes_dict, class_sensitive: bool, extra_radius: float = 0.0, radius_scale: float = 1.0):
			cands = []  # (d2, di, nid)
			for di, d in enumerate(dets_list):
				for nid, n in nodes_dict.items():
					if nid in used_nodes:
						continue
					if class_sensitive and n['class'] != d['class']:
						continue
					dx = d['gx'] - n['x']
					dy = d['gy'] - n['y']
					d2 = dx * dx + dy * dy
					thr = radius_scale * self._dynamic_radius(d['dist'], extra_radius)
					if d2 <= thr * thr:
						cands.append((d2, di, nid))
			cands.sort(key=lambda x: x[0])
			return cands

		# Split nodes into confirmed and tentative (recent only)
		confirmed_nodes = active_nodes
		tentative_nodes = {nid: n for nid, n in self.nodes.items() if (not n.get('confirmed', False)) and (t - n['last']) <= self.max_node_age}

		# Pass 1: strict class match on confirmed
		if self.strict_class and active_nodes:
			for d2, di, nid in candidates(dets, confirmed_nodes, class_sensitive=True, extra_radius=0.0, radius_scale=1.0):
				if di in assigned or nid in used_nodes:
					continue
				assigned[di] = nid
				used_nodes.add(nid)

		# Pass 2: fallback class-agnostic on confirmed
		if self.fallback_class_agnostic and confirmed_nodes:
			for d2, di, nid in candidates(dets, confirmed_nodes, class_sensitive=False, extra_radius=self.fallback_extra_radius, radius_scale=1.0):
				if di in assigned or nid in used_nodes:
					continue
				assigned[di] = nid
				used_nodes.add(nid)

		# Pass 3: strict class match on tentative (tighter radius)
		if self.strict_class and tentative_nodes:
			for d2, di, nid in candidates(dets, tentative_nodes, class_sensitive=True, extra_radius=0.0, radius_scale=self.tentative_radius_scale):
				if di in assigned or nid in used_nodes:
					continue
				assigned[di] = nid
				used_nodes.add(nid)

		# Pass 4: fallback class-agnostic on tentative (tighter radius)
		if self.fallback_class_agnostic and tentative_nodes:
			for d2, di, nid in candidates(dets, tentative_nodes, class_sensitive=False, extra_radius=self.fallback_extra_radius, radius_scale=self.tentative_radius_scale):
				if di in assigned or nid in used_nodes:
					continue
				assigned[di] = nid
				used_nodes.add(nid)

		# Apply updates / create new nodes
		for di, d in enumerate(dets):
			if di in assigned:
				self._update_node(assigned[di], d['gx'], d['gy'], t)
			else:
				self._create_node(d['class'], d['gx'], d['gy'], t)

		# After processing a frame, publish graph JSON snapshot
		self.publish_graph()

	# --------------- Publishing --------------
	def publish_markers(self):
		ma = MarkerArray()
		now = self.get_clock().now().to_msg()

		# Nodes as spheres and labels; tracks as line strips
		for nid, n in self.nodes.items():
			# Only render confirmed nodes
			if not n.get('confirmed', False):
				continue
			r, g, b = n['color']

			# Node marker
			m = Marker()
			m.header.frame_id = self.global_frame
			m.header.stamp = now
			m.ns = 'nodes'
			m.id = nid
			m.type = Marker.SPHERE
			m.action = Marker.ADD
			m.pose.position.x = float(n['x'])
			m.pose.position.y = float(n['y'])
			m.pose.position.z = 0.0
			m.pose.orientation.w = 1.0
			m.scale.x = self.node_diameter
			m.scale.y = self.node_diameter
			m.scale.z = self.node_diameter
			m.color.r = r
			m.color.g = g
			m.color.b = b
			# tentative nodes render more transparent
			if not n.get('confirmed', False):
				m.color.a = 0.35
			else:
				m.color.a = 0.9
			ma.markers.append(m)

			# Label
			t = Marker()
			t.header.frame_id = self.global_frame
			t.header.stamp = now
			t.ns = 'labels'
			t.id = 10000 + nid
			t.type = Marker.TEXT_VIEW_FACING
			t.action = Marker.ADD
			t.pose.position.x = float(n['x'])
			t.pose.position.y = float(n['y'])
			t.pose.position.z = max(0.05, self.node_diameter)
			t.pose.orientation.w = 1.0
			t.scale.z = self.label_scale
			# lighter label for tentative
			if not n.get('confirmed', False):
				t.color.r = 1.0
				t.color.g = 1.0
				t.color.b = 1.0
				t.color.a = 0.6
			else:
				t.color.r = 1.0
				t.color.g = 1.0
				t.color.b = 1.0
				t.color.a = 0.9
			t.text = f"{n['id']}:{n['class']} x{n['count']}"
			ma.markers.append(t)

			# Track line strip
			if len(n['history']) >= 2:
				l = Marker()
				l.header.frame_id = self.global_frame
				l.header.stamp = now
				l.ns = 'tracks'
				l.id = 20000 + nid
				l.type = Marker.LINE_STRIP
				l.action = Marker.ADD
				l.scale.x = self.line_width
				l.color.r = r
				l.color.g = g
				l.color.b = b
				l.color.a = 0.7
				for (hx, hy, _ht) in n['history']:
					pt = Point()
					pt.x = float(hx)
					pt.y = float(hy)
					pt.z = 0.0
					l.points.append(pt)
				ma.markers.append(l)

		if len(ma.markers) > 0:
			self.marker_pub.publish(ma)

	def publish_graph(self):
		# Minimal JSON summary
		nodes = [
			{
				'id': nid,
				'class': n['class'],
				'x': n['x'],
				'y': n['y'],
				'count': n['count'],
				'first': n['first'],
				'last': n['last'],
			}
			for nid, n in self.nodes.items()
		]
		msg = String()
		msg.data = json.dumps({'frame': self.global_frame, 'nodes': nodes})
		self.graph_pub.publish(msg)


def main():
	rclpy.init()
	node = LandmakerWeaverNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()

